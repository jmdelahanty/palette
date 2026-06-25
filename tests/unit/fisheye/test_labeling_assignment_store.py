from __future__ import annotations

import csv
import hashlib
import json
import threading
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import fisheye.labeling.web as labeling_web_module
from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.web import (
    IDENTITY_PERSONAL_QUEUE_EVIDENCE_STATUS_VALUES,
    _BROWSER_MUTATION_STATUS_JS,
    ServerConfig,
    _add_work_summary_fields,
    _admin_html,
    _admin_recording_html,
    _admin_recording_payload,
    _admin_summary_payload,
    _admin_user_html,
    _admin_user_payload,
    _admin_promotion_retry_preflight_error,
    _assignment_ownership_policy,
    _browser_mutation_failure_metadata,
    _browser_mutation_write_policy,
    _browser_mutation_response_metadata,
    _browser_smoke_evidence_template,
    _browser_smoke_personalized_route_contract,
    _browser_response_security_contract_policy,
    _browser_response_security_evidence_template,
    _record_browser_response_security_evidence,
    _record_browser_smoke_evidence,
    _labeler_promotion_retry_operator_support_payload,
    _browser_response_security_policy,
    _handoff_browser_response_security_fields,
    _browser_signed_link_policy,
    _browser_task_state_policy,
    _dashboard_html,
    _dashboard_dataset_queue_counts,
    _dashboard_invite_actions,
    _dashboard_invite_reason_counts,
    _dashboard_operator_validation_fields,
    _dashboard_roster_rows,
    _datasets_html,
    _mutation_audit_policy,
    _operator_validation_visibility_policy,
    _operator_validation_public_fields,
    _queue_first_entry_contract_policy,
    _zarr_backup_policy,
    _handoff_ready_to_send,
    _handoff_sendability_actions,
    _handoff_sendability_reasons,
    _identity_probe_html,
    _identity_probe_payload,
    _inspection_failure_actions,
    _inspection_operator_repair_commands,
    _inspect_handoff_launch_evidence_execution_checklist,
    _inspect_handoff_package,
    _inspect_handoff_operator_evidence_commands,
    _launch_evidence_execution_checklist_public_summary,
    _operator_evidence_template_status,
    _operator_evidence_commands_public_summary,
    _inspect_handoff_validation_checklist,
    _keypoint_session_html,
    _labeler_safety_policy,
    _labeler_route_authorization_policy,
    _session_guard_policy,
    _store_consistency_report,
    _signed_task_link_token,
    _task_open_preflight_error,
    _validation_checklist_gate_summary,
    _work_dataset_queue_task,
    _write_directory_checksums,
    ServerState,
    _make_handler,
)
from fisheye.utils import labeling_work


def _store(tmp_path):
    store = LabelingStore(tmp_path / "labeling_work.sqlite")
    store.initialize()
    return store


@contextmanager
def _running_labeling_server(
    store: LabelingStore,
    *,
    user: str,
    link_secret: str | None = None,
    require_operator_validation_for_start: bool = False,
    validation_checklist_path: Path | None = None,
):
    config = ServerConfig(
        store_path=store.path,
        host="127.0.0.1",
        port=0,
        fixed_user=user,
        auth_header="X-Forwarded-User",
        session_ttl_seconds=600,
        admin_users=(user,),
        link_secret=link_secret,
        validation_checklist_path=validation_checklist_path,
        require_operator_validation_for_start=require_operator_validation_for_start,
    )
    state = ServerState(store=store, config=config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _http_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: object | None = None,
) -> tuple[int, str, bytes]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return (
                response.status,
                str(response.headers.get("Content-Type") or ""),
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return (
            exc.code,
            str(exc.headers.get("Content-Type") or ""),
            exc.read(),
        )


def _sendability_ready_baseline_fields() -> dict[str, object]:
    return {
        "single_owner_policy": _assignment_ownership_policy(),
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "unique_active_recording_count": 1,
            "duplicate_active_owner_count": 0,
            "recording_id_primary_key": True,
            "schema_enforced_recording_primary_key": True,
            "primary_key_columns": ["recording_id"],
            "schema_integrity_source": "test_fixture",
        },
        "labeler_route_authorization_checklist": {
            "ready": True,
            "single_owner_store_proof_ready": True,
            "assignment_ownership_integrity_ok": True,
            "duplicate_active_owner_count": 0,
            "browser_mutation_target_resolved_server_side": True,
            "labelers_mutate_assigned_training_zarrs": True,
            "labelers_mutate_intermediate_csvs": False,
        },
        "dataset_queue_blocks_labeler_start": False,
    }


def test_dashboard_dataset_queue_counts_derives_personal_queue_matches_from_urls():
    counts = _dashboard_dataset_queue_counts(
        [
            {
                "user": "alice",
                "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                "personal_dataset_queue_link_role": "preferred_queue",
                "dataset_queue_link_role": "canonical_queue_fallback",
                "expected_user_personal_dataset_queue_url": "/my-datasets?expected_user=alice",
                "preferred_labeler_entry_url": "/my-datasets?expected_user=alice",
                "personalized_labeler_entry_url": "/my-datasets?expected_user=alice",
                "dataset_queue_preview_url": "/my-datasets?expected_user=alice",
                "canonical_dataset_queue_preview_url": "/datasets?expected_user=alice",
            },
            {
                "user": "bob",
                "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                "personal_dataset_queue_link_role": "preferred_queue",
                "dataset_queue_link_role": "canonical_queue_fallback",
                "expected_user_personal_dataset_queue_url": "/my-datasets?expected_user=bob",
                "preferred_labeler_entry_url": "/my-datasets?expected_user=bob",
                "personalized_labeler_entry_url": "/my-datasets?expected_user=bob",
                "dataset_queue_preview_url": "",
                "canonical_dataset_queue_preview_url": "/datasets?expected_user=bob",
                "preferred_labeler_entry_url_matches_personal_dataset_queue": "False",
                "personalized_labeler_entry_url_matches_personal_dataset_queue": "False",
            },
        ]
    )

    assert counts["personalized_dataset_queue_preview_users"] == ["alice"]
    assert counts["missing_personalized_dataset_queue_preview_users"] == ["bob"]
    assert counts["all_users_have_personalized_dataset_queue_preview"] is False
    assert counts["preferred_personal_queue_match_users"] == ["alice", "bob"]
    assert counts["missing_preferred_personal_queue_match_users"] == []
    assert counts["all_users_have_preferred_personal_queue_match"] is True
    assert counts["personalized_personal_queue_match_users"] == ["alice", "bob"]
    assert counts["missing_personalized_personal_queue_match_users"] == []
    assert counts["all_users_have_personalized_personal_queue_match"] is True


def test_dashboard_roster_blocks_invitation_when_preferred_personal_queue_is_missing(
    tmp_path,
    monkeypatch,
):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
        )
        monkeypatch.setattr(
            labeling_web_module,
            "_personal_dataset_queue_url_for_dashboard",
            lambda dashboard_url, user: "",
        )

        rows = _dashboard_roster_rows(
            store,
            dashboard_url="https://labeling.example.org/work",
            operator_validation_fields={
                "operator_validation_required_before_invite": True,
                "operator_validation_all_complete": True,
                "operator_validation_status": "passed",
                "operator_validation_source": "validation_checklist",
            },
        )
    finally:
        store.close()

    assert len(rows) == 1
    row = rows[0]
    assert row["ready_to_invite"] is False
    assert row["copy_intent"] == "diagnostic_note"
    assert row["invite_reasons"] == ["preferred_personal_queue_mismatch"]
    assert row["preferred_labeler_entry_url_matches_dataset_queue"] is True
    assert row["preferred_labeler_entry_url_matches_personal_dataset_queue"] is False
    assert row["personalized_labeler_entry_url_matches_personal_dataset_queue"] is False
    assert "preferred_personal_queue_mismatch" in row["invitation_message"]
    assert "/my-datasets?expected_user=<user>" in row["invite_actions"][0]
    assert _dashboard_invite_reason_counts(rows) == {
        "preferred_personal_queue_mismatch": 1
    }
    assert _dashboard_invite_actions(_dashboard_invite_reason_counts(rows)) == [
        "Regenerate roster or handoff links so Start here uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of a canonical fallback."
    ]


def test_dashboard_invite_actions_explain_preferred_personal_queue_mismatch():
    actions = _dashboard_invite_actions(["preferred_personal_queue_mismatch"])

    assert actions == [
        "Regenerate roster or handoff links so Start here uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of a canonical fallback."
    ]


def test_handoff_sendability_blocks_preferred_personal_queue_mismatch():
    manifest = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "counts": {"tasks": 1, "signed_links": 1, "ready_to_share_links": 1},
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "recording_id_primary_key": True,
            "schema_enforced_recording_primary_key": True,
            "schema_integrity_source": "store_pragma",
            "primary_key_columns": ["recording_id"],
            "one_row_per_recording_enforced": True,
            "active_assignment_count": 1,
            "unique_active_recording_count": 1,
            "duplicate_active_owner_count": 0,
            "duplicate_active_owners": [],
        },
        "files": {
            "html_index": "index.html",
            "message": "message.txt",
            "quickstart": "quickstart.txt",
            "dataset_queue": "dataset-queue.json",
            "manifest": "manifest.json",
        },
        "labeler_safety_policy_present": True,
        "labeler_safety_ready": True,
        "labeler_route_authorization_policy_present": True,
        "labeler_route_authorization_ready": True,
        "signed_link_policy_present": True,
        "signed_link_policy_ready": True,
        "session_guard_policy_present": True,
        "session_guard_policy_ready": True,
        "task_state_policy_present": True,
        "task_state_policy_ready": True,
        "zarr_backup_policy_present": True,
        "zarr_backup_ready": True,
        "mutation_audit_policy_present": True,
        "mutation_audit_ready": True,
        "browser_response_security_policy_present": True,
        "browser_response_security_ready": True,
        "browser_mutation_write_policy_present": True,
        "browser_mutation_write_ready": True,
    }

    reasons = _handoff_sendability_reasons(manifest)
    actions = _handoff_sendability_actions(["preferred_personal_queue_mismatch"])

    assert _handoff_ready_to_send(manifest) is False
    assert "preferred_personal_queue_mismatch" in reasons
    assert actions == [
        "Regenerate the handoff so Start here uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of a canonical /datasets fallback."
    ]


def test_inspection_failure_actions_explain_preferred_personal_queue_mismatch():
    actions = _inspection_failure_actions(
        ["preferred_personal_queue_mismatch"],
        validation_checklist={},
    )
    repair_command_rows = {
        row["id"]: row for row in _inspection_operator_repair_commands(actions)
    }

    assert actions == [
        "Regenerate the handoff package so every labeler Start here link uses the guarded /my-datasets?expected_user=<user> preferred queue URL instead of canonical /datasets fallback."
    ]
    assert "regenerate_handoffs_with_personal_dataset_queue" in repair_command_rows
    repair_command = repair_command_rows[
        "regenerate_handoffs_with_personal_dataset_queue"
    ]
    assert repair_command["category"] == "handoff_regeneration"
    assert repair_command["reason_ids"] == ["preferred_personal_queue_mismatch"]
    assert "export-user-handoffs" in repair_command["command"]
    assert "--base-url DEPLOYED_LABELING_URL" in repair_command["command"]
    assert repair_command["requires_checksum_refresh_after_run"] is False


def test_operator_validation_public_fields_default_pending_launch_gates():
    fields = _operator_validation_public_fields(
        {
            "operator_validation_required_before_invite": True,
            "operator_validation_status": "pending_operator_evidence",
        }
    )

    assert fields["operator_validation_gate_count"] == 6
    assert fields["operator_validation_required_pending_gate_count"] == 6
    assert fields["operator_validation_required_missing_evidence_gate_count"] == 6
    assert fields["operator_validation_pending_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert fields["operator_validation_required_missing_evidence_gate_ids"] == fields[
        "operator_validation_pending_gate_ids"
    ]
    assert fields["operator_validation_source"] == "none"
    assert fields["operator_validation_external_evidence_required"] is True
    assert fields["operator_validation_external_evidence_required_gate_count"] == 5
    assert fields["operator_validation_external_evidence_required_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert fields["operator_validation_external_evidence_template_fields_by_gate_id"][
        "browser_smoke"
    ] == "browser_smoke_evidence_template"
    assert fields["operator_validation_external_evidence_template_paths_by_gate_id"][
        "browser_smoke"
    ] == "browser-smoke-evidence-template.json"
    assert fields["operator_validation_checklist_only_required_gate_ids"] == [
        "operator_recovery_contract"
    ]
    assert fields["operator_validation_checklist_only_required_gate_count"] == 1
    assert "mutable_zarr_backup_confirmation" in fields["operator_validation_operator_action"]
    assert "operator_recovery_contract" in fields["operator_validation_operator_action"]


def test_default_operator_validation_public_fields_hide_checklist_path():
    fields = _operator_validation_public_fields(_dashboard_operator_validation_fields())

    assert "operator_validation_checklist_path" not in fields
    assert fields["operator_validation_source"] == "none"
    assert fields["operator_validation_gate_count"] == 6
    assert fields["operator_validation_pending_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert fields["identity_personal_queue_evidence_status"] == "missing"
    assert fields["identity_personal_queue_evidence_ready_count"] == 0
    assert fields["identity_personal_queue_evidence_missing_count"] == 0
    assert fields["identity_personal_queue_evidence_ready_users"] == []
    assert fields["identity_personal_queue_evidence_missing_users"] == []
    assert fields["identity_personal_queue_evidence_missing_fields_by_user"] == {}
    assert fields["identity_all_users_have_personal_queue_evidence"] is False
    assert fields["operator_validation_external_evidence_required"] is True
    assert fields["operator_validation_external_evidence_required_gate_count"] == 5
    assert fields["operator_validation_checklist_only_required_gate_ids"] == [
        "operator_recovery_contract"
    ]


def test_operator_validation_public_fields_normalize_partial_pending_gate_lists():
    fields = _operator_validation_public_fields(
        {
            "operator_validation_required_before_invite": True,
            "operator_validation_status": "pending_operator_evidence",
            "operator_validation_source": "none",
            "operator_validation_pending_gate_ids": ["browser_smoke"],
        }
    )

    assert fields["operator_validation_gate_count"] == 1
    assert fields["operator_validation_required_pending_gate_count"] == 1
    assert fields["operator_validation_required_missing_evidence_gate_count"] == 1
    assert fields["operator_validation_pending_gate_ids"] == ["browser_smoke"]
    assert fields["operator_validation_required_missing_evidence_gate_ids"] == [
        "browser_smoke"
    ]
    assert fields["operator_validation_external_evidence_required"] is True
    assert fields["operator_validation_external_evidence_required_gate_ids"] == [
        "browser_smoke"
    ]
    assert fields["operator_validation_external_evidence_required_gate_count"] == 1
    assert fields["operator_validation_checklist_only_required_gate_ids"] == []
    assert "browser_smoke" in fields["operator_validation_operator_action"]


def test_safe_share_next_action_summary_fails_closed_for_operator_validation_only_payload():
    text = labeling_web_module._safe_share_next_action_summary_text(
        {
            "operator_validation_required_before_invite": True,
            "operator_validation_status": "pending_operator_evidence",
            "operator_validation_pending_gate_ids": ["browser_smoke"],
            "operator_validation_required_missing_evidence_gate_ids": ["browser_smoke"],
        }
    )

    assert "Safe-share next actions: 6" in text
    assert "browser_smoke=missing_evidence" in text
    assert "mutable_zarr_backup_confirmation=unknown" in text


def test_safe_share_exact_field_values_preserve_partial_next_actions():
    fields = labeling_web_module._safe_share_checklist_field_values(
        {
            "safe_share_launch_blocking_next_actions": [
                {
                    "gate_id": "browser_smoke",
                    "status": "missing_evidence",
                    "operator_only": True,
                    "blocks_share": True,
                    "action": "Attach approved browser smoke evidence.",
                }
            ]
        }
    )

    assert fields["safe_share_launch_blocking_gate_statuses"] == {
        "browser_smoke": "missing_evidence"
    }
    assert fields["safe_share_launch_blocking_missing_evidence_gate_ids"] == [
        "browser_smoke"
    ]
    assert fields["safe_share_launch_blocking_unsatisfied_gate_ids"] == ["browser_smoke"]
    assert fields["safe_share_launch_blocking_next_action_count"] == 1
    assert fields["safe_share_launch_blocking_next_action_detail_fields"] == [
        "gate_id",
        "status",
        "operator_only",
        "blocks_share",
        "action",
        "operator_validation_command_template_schema",
        "operator_validation_command_ids",
        "operator_validation_record_command_ids",
        "operator_validation_apply_command_id",
        "operator_validation_apply_required_after_approval",
        "operator_validation_evidence_template_field",
        "operator_validation_evidence_template_path",
    ]
    assert fields["safe_share_launch_blocking_next_action_command_fields"] == [
        "operator_validation_command_template_schema",
        "operator_validation_command_ids",
        "operator_validation_record_command_ids",
        "operator_validation_apply_command_id",
        "operator_validation_apply_required_after_approval",
        "operator_validation_evidence_template_field",
        "operator_validation_evidence_template_path",
    ]
    action = fields["safe_share_launch_blocking_next_actions"][0]
    assert action["operator_validation_command_template_schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert action["operator_validation_record_command_ids"] == [
        "record_browser_smoke_evidence"
    ]
    assert action["operator_validation_apply_command_id"] == (
        "apply_operator_evidence_templates"
    )
    assert action["operator_validation_apply_required_after_approval"] is True
    assert action["operator_validation_evidence_template_path"] == (
        "browser-smoke-evidence-template.json"
    )
    assert "browser_smoke=missing_evidence" in fields["safe_share_next_action_summary"]
    assert fields["safe_share_external_launch_evidence_gap_gate_ids"] == [
        "browser_smoke"
    ]
    assert fields["safe_share_external_launch_evidence_gap_count"] == 1
    assert fields["safe_share_external_launch_evidence_gap_statuses"] == {
        "browser_smoke": "missing_evidence"
    }
    assert fields["safe_share_external_launch_evidence_gap_template_paths_by_gate_id"] == {
        "browser_smoke": "browser-smoke-evidence-template.json"
    }
    assert fields[
        "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id"
    ] == {"browser_smoke": ["record_browser_smoke_evidence"]}
    assert fields["safe_share_external_launch_evidence_gap_todo_count"] == 1
    assert fields["safe_share_external_launch_evidence_gap_todo_fields"] == [
        "gate_id",
        "status",
        "operator_only",
        "blocks_share",
        "action",
        "operator_validation_command_template_schema",
        "operator_validation_record_command_ids",
        "operator_validation_apply_command_id",
        "operator_validation_apply_required_after_approval",
        "operator_validation_evidence_template_field",
        "operator_validation_evidence_template_path",
    ]
    assert fields["safe_share_external_launch_evidence_gap_todos"][0][
        "gate_id"
    ] == "browser_smoke"
    assert fields["safe_share_external_launch_evidence_gap_todos"][0][
        "operator_validation_evidence_template_path"
    ] == "browser-smoke-evidence-template.json"


def test_safe_share_projection_preserves_summary_only_artifacts():
    fields = labeling_web_module._safe_share_checklist_gate_status_fields_from_operator_validation(
        {
            "safe_share_next_action_summary": (
                "Safe-share next actions: 1; browser_smoke=missing_evidence"
            )
        },
        safe_share_gate=labeling_web_module._safe_share_gate_policy(),
    )

    assert fields["safe_share_next_action_summary"] == (
        "Safe-share next actions: 1; browser_smoke=missing_evidence"
    )
    assert fields["safe_share_launch_blocking_gate_statuses"] == {}
    assert fields["safe_share_launch_blocking_next_actions"] == []
    assert fields["safe_share_launch_blocking_next_action_count"] == 0
    assert fields["safe_share_external_launch_evidence_gap_gate_ids"] == []
    assert fields["safe_share_external_launch_evidence_gap_action_required"] is False


def test_safe_share_projection_preserves_external_gap_todo_only_artifacts():
    fields = labeling_web_module._safe_share_checklist_field_values(
        {
            "safe_share_external_launch_evidence_gap_todos": [
                {
                    "gate_id": "identity_probe_verification",
                    "status": "pending_operator_evidence",
                    "operator_only": True,
                    "blocks_share": True,
                    "action": "Record deployed identity-source evidence.",
                    "operator_validation_record_command_ids": [
                        "record_identity_source_evidence"
                    ],
                    "operator_validation_apply_command_id": (
                        "apply_operator_evidence_templates"
                    ),
                    "operator_validation_apply_required_after_approval": True,
                    "operator_validation_evidence_template_field": (
                        "identity_source_evidence_template"
                    ),
                    "operator_validation_evidence_template_path": (
                        "identity-source-evidence-template.json"
                    ),
                }
            ]
        }
    )

    assert fields["safe_share_launch_blocking_gate_statuses"] == {
        "identity_probe_verification": "pending_operator_evidence"
    }
    assert fields["safe_share_launch_blocking_pending_gate_ids"] == [
        "identity_probe_verification"
    ]
    assert fields["safe_share_launch_blocking_unsatisfied_gate_ids"] == [
        "identity_probe_verification"
    ]
    assert fields["safe_share_external_launch_evidence_gap_gate_ids"] == [
        "identity_probe_verification"
    ]
    assert fields["safe_share_external_launch_evidence_gap_todo_count"] == 1
    assert fields["safe_share_external_launch_evidence_gap_todos"][0][
        "operator_validation_record_command_ids"
    ] == ["record_identity_source_evidence"]
    assert fields["safe_share_external_launch_evidence_gap_todos"][0][
        "operator_validation_evidence_template_path"
    ] == "identity-source-evidence-template.json"


def test_personalized_launch_readiness_derives_gap_fields_from_todo_only_artifacts():
    readiness = labeling_web_module._personalized_launch_readiness_summary(
        {
            "personalized_labeler_entry_url": "/my-datasets?expected_user=alice",
            "browser_mutation_browser_label_write_target": "training_zarr",
            "browser_mutation_browser_writes_csv_or_handoff_files": "False",
            "browser_mutation_browser_has_direct_zarr_write_authority": "False",
            "queue_first_entry_contract": {
                "ready": "False",
                "personalized_labeler_entry_url_matches_personal_dataset_queue": (
                    "False"
                ),
            },
            "personalized_labeler_entry_url_matches_personal_dataset_queue": (
                "False"
            ),
            "labeler_start_ready": "False",
            "safe_share_required_inspection_value": "False",
            "safe_share_checklist_gate_evidence_complete": "False",
            "safe_share_external_launch_evidence_gap_todos": json.dumps(
                [
                    {
                        "gate_id": "identity_probe_verification",
                        "status": "pending_operator_evidence",
                        "operator_only": True,
                        "blocks_share": True,
                        "action": "Record deployed identity-source evidence.",
                        "operator_validation_record_command_ids": [
                            "record_identity_source_evidence"
                        ],
                        "operator_validation_apply_command_id": (
                            "apply_operator_evidence_templates"
                        ),
                        "operator_validation_apply_required_after_approval": True,
                        "operator_validation_evidence_template_field": (
                            "identity_source_evidence_template"
                        ),
                        "operator_validation_evidence_template_path": (
                            "identity-source-evidence-template.json"
                        ),
                    }
                ]
            ),
        }
    )

    assert readiness["external_launch_evidence_gap_action_required"] is True
    assert readiness["queue_first_entry_contract_ready"] is False
    assert (
        readiness["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        is False
    )
    assert readiness["labeler_start_ready"] is False
    assert readiness["safe_share_required_inspection_value"] is False
    assert readiness["safe_share_checklist_gate_evidence_complete"] is False
    assert readiness["external_launch_evidence_gap_count"] == 1
    assert readiness["external_launch_evidence_gap_gate_ids"] == [
        "identity_probe_verification"
    ]
    assert readiness["external_launch_evidence_gap_statuses"] == {
        "identity_probe_verification": "pending_operator_evidence"
    }
    assert readiness["external_launch_evidence_gap_todo_count"] == 1
    assert "operator_validation_evidence_template_path" in readiness[
        "external_launch_evidence_gap_todo_fields"
    ]
    assert readiness[
        "external_launch_evidence_gap_template_paths_by_gate_id"
    ] == {
        "identity_probe_verification": "identity-source-evidence-template.json"
    }
    assert readiness[
        "external_launch_evidence_gap_record_command_ids_by_gate_id"
    ] == {
        "identity_probe_verification": ["record_identity_source_evidence"]
    }
    assert readiness["browser_label_write_target"] == "training_zarr"
    assert readiness["browser_writes_csv_or_handoff_files"] is False
    assert readiness["browser_has_direct_zarr_write_authority"] is False


def test_operator_validation_public_fields_preserve_safe_share_summary_without_checklist_path():
    fields = _operator_validation_public_fields(
        {
            "operator_validation_checklist_path": "/operator/private/validation-checklist.json",
            "safe_share_next_action_summary": (
                "Safe-share next actions: 1; browser_smoke=missing_evidence"
            ),
        }
    )

    assert "operator_validation_checklist_path" not in fields
    assert fields["safe_share_next_action_summary"] == (
        "Safe-share next actions: 1; browser_smoke=missing_evidence"
    )


def test_operator_validation_public_fields_expose_identity_personal_queue_evidence():
    fields = _operator_validation_public_fields(
        {
            "identity_personal_queue_evidence_ready_count": 1,
            "identity_personal_queue_evidence_missing_count": 1,
            "identity_personal_queue_evidence_ready_users": ["alice"],
            "identity_personal_queue_evidence_missing_users": ["bob"],
            "identity_personal_queue_evidence_missing_fields_by_user": {
                "bob": ["preferred_labeler_entry_url_matches_personal_dataset_queue"],
            },
            "identity_all_users_have_personal_queue_evidence": False,
        }
    )

    assert fields["identity_personal_queue_evidence_status"] == "incomplete"
    assert fields["identity_personal_queue_evidence_ready_count"] == 1
    assert fields["identity_personal_queue_evidence_missing_count"] == 1
    assert fields["identity_personal_queue_evidence_ready_users"] == ["alice"]
    assert fields["identity_personal_queue_evidence_missing_users"] == ["bob"]
    assert fields["identity_personal_queue_evidence_missing_fields_by_user"] == {
        "bob": ["preferred_labeler_entry_url_matches_personal_dataset_queue"],
    }
    assert fields["identity_all_users_have_personal_queue_evidence"] is False


def test_operator_validation_public_fields_normalize_identity_personal_queue_status():
    ready_fields = _operator_validation_public_fields(
        {
            "identity_personal_queue_evidence_status": "stale_ready",
            "identity_personal_queue_evidence_ready_users": ["alice"],
            "identity_all_users_have_personal_queue_evidence": True,
        }
    )
    conflicting_fields = _operator_validation_public_fields(
        {
            "identity_personal_queue_evidence_ready_users": ["alice"],
            "identity_personal_queue_evidence_missing_users": ["bob"],
            "identity_all_users_have_personal_queue_evidence": True,
        }
    )
    stale_all_users_without_ready_evidence_fields = _operator_validation_public_fields(
        {"identity_all_users_have_personal_queue_evidence": True}
    )
    incomplete_fields = _operator_validation_public_fields(
        {
            "identity_personal_queue_evidence_status": "stale_incomplete",
            "identity_personal_queue_evidence_missing_users": ["alice"],
        }
    )
    missing_fields = _operator_validation_public_fields(
        {"identity_personal_queue_evidence_status": "stale_missing"}
    )
    stale_ready_without_proof_fields = _operator_validation_public_fields(
        {"identity_personal_queue_evidence_status": " READY "}
    )

    assert ready_fields["identity_personal_queue_evidence_status"] == "ready"
    assert conflicting_fields["identity_personal_queue_evidence_status"] == "incomplete"
    assert (
        stale_all_users_without_ready_evidence_fields[
            "identity_personal_queue_evidence_status"
        ]
        == "missing"
    )
    assert incomplete_fields["identity_personal_queue_evidence_status"] == "incomplete"
    assert missing_fields["identity_personal_queue_evidence_status"] == "missing"
    assert (
        stale_ready_without_proof_fields["identity_personal_queue_evidence_status"]
        == "missing"
    )


def test_operator_validation_visibility_policy_marks_checklist_path_operator_only():
    policy = _operator_validation_visibility_policy()

    assert policy["operator_only_fields"] == ["operator_validation_checklist_path"]
    assert policy["operator_action_fields"] == ["operator_validation_command_templates"]
    assert policy["operator_action_fields_are_labeler_instructions"] is False
    assert policy["labeler_visible_payloads_may_include_operator_action_fields_for_support"] is True
    assert policy["labeler_visible_payloads_include_operator_only_fields"] is False
    assert policy["per_user_payloads_use_public_fields_only"] is True
    assert "operator_validation_status" in policy["public_fields"]
    assert "operator_validation_required_missing_evidence_gate_ids" in policy["public_fields"]
    assert "operator_validation_external_evidence_required" in policy["public_fields"]
    assert "operator_validation_external_evidence_required_gate_ids" in policy["public_fields"]
    assert (
        "operator_validation_external_evidence_template_paths_by_gate_id"
        in policy["public_fields"]
    )
    assert "operator_validation_checklist_only_required_gate_ids" in policy["public_fields"]
    assert "identity_personal_queue_evidence_status" in policy["public_fields"]
    assert "identity_personal_queue_evidence_ready_count" in policy["public_fields"]
    assert "identity_personal_queue_evidence_missing_count" in policy["public_fields"]
    assert "identity_personal_queue_evidence_ready_users" in policy["public_fields"]
    assert "identity_personal_queue_evidence_missing_users" in policy["public_fields"]
    assert "identity_personal_queue_evidence_missing_fields_by_user" in policy["public_fields"]
    assert "identity_all_users_have_personal_queue_evidence" in policy["public_fields"]
    assert policy["identity_personal_queue_evidence_status_values"] == [
        "missing",
        "incomplete",
        "ready",
    ]
    assert policy["operator_validation_gate_status_values"] == [
        "unknown",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]
    assert policy["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert policy["operator_validation_gate_flat_field_suffixes"] == [
        "status",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]


def test_queue_first_entry_contract_requires_expected_user_guarded_personal_queue():
    common = {
        "labeler_safety": _labeler_safety_policy(),
        "labeler_landing_page_path": "/",
        "labeler_landing_url": "/",
        "expected_user_labeler_landing_url": "/?expected_user=alice",
        "dataset_queue_page_path": "/datasets",
        "dataset_queue_url": "/datasets",
        "expected_user_dataset_queue_url": "/datasets?expected_user=alice",
        "dashboard_url": "/work",
        "expected_user_dashboard_url": "/work?expected_user=alice",
        "personal_dataset_queue_page_path": "/my-datasets",
        "personal_dataset_queue_url": "/my-datasets",
        "personal_work_page_path": "/my-work",
        "personal_work_url": "/my-work",
        "expected_user_personal_work_url": "/my-work?expected_user=alice",
    }

    guarded = _queue_first_entry_contract_policy(
        **common,
        expected_user_personal_dataset_queue_url="/my-datasets?expected_user=alice",
    )
    unguarded = _queue_first_entry_contract_policy(
        **common,
        expected_user_personal_dataset_queue_url="",
    )

    assert guarded["ready"] is True
    assert guarded["preferred_labeler_entry_url_is_expected_user_guarded"] is True
    assert guarded["personalized_labeler_entry_url_is_expected_user_guarded"] is True
    assert unguarded["ready"] is False
    assert unguarded["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert (
        unguarded["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        is True
    )
    assert unguarded["preferred_labeler_entry_url_is_expected_user_guarded"] is False
    assert unguarded["personalized_labeler_entry_url_is_expected_user_guarded"] is False


def test_admin_summary_payload_exposes_dataset_queue_direct_start_policy(tmp_path):
    store = _store(tmp_path)
    config = ServerConfig(
        store_path=tmp_path / "labeling_work.sqlite",
        host="127.0.0.1",
        port=0,
        fixed_user=None,
        auth_header="X-User",
        session_ttl_seconds=3600,
        trust_auth_header=True,
        admin_users=("admin@example.org",),
        production=True,
    )

    payload = _admin_summary_payload(store, config=config)

    assert payload["labeler_landing_page_path"] == "/"
    assert payload["dashboard_path"] == "/work"
    assert payload["dataset_queue_page_path"] == "/datasets"
    assert payload["personal_work_page_path"] == "/my-work"
    assert payload["personal_dataset_queue_page_path"] == "/my-datasets"
    assert payload["personal_work_alias_for"] == "/work"
    assert payload["personal_dataset_queue_alias_for"] == "/datasets"
    assert payload["preferred_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert payload["preferred_labeler_entry_path"] == "/my-datasets"
    assert payload["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert payload["personalized_labeler_entry_path"] == "/my-datasets"
    assert payload["preflight"]["personal_work_page_path"] == "/my-work"
    assert payload["preflight"]["personal_dataset_queue_page_path"] == "/my-datasets"
    assert payload["preflight"]["personalized_labeler_entrypoint"] == (
        "personal_datasets_waiting_queue"
    )
    assert payload["preflight"]["personalized_labeler_entry_path"] == "/my-datasets"
    dashboard_counts = payload["dashboard_user_counts"]
    assert "personalized_dataset_queue_preview_users" in dashboard_counts
    assert "canonical_dataset_queue_preview_users" in dashboard_counts
    assert "missing_personalized_dataset_queue_preview_users" in dashboard_counts
    assert "all_users_have_personalized_dataset_queue_preview" in dashboard_counts
    assert "preferred_personal_queue_match_users" in dashboard_counts
    assert "missing_preferred_personal_queue_match_users" in dashboard_counts
    assert "all_users_have_preferred_personal_queue_match" in dashboard_counts
    assert "personalized_personal_queue_match_users" in dashboard_counts
    assert "missing_personalized_personal_queue_match_users" in dashboard_counts
    assert "all_users_have_personalized_personal_queue_match" in dashboard_counts
    assert "dataset_queue_preferred_entrypoint_counts" in dashboard_counts
    assert "dataset_queue_link_role_counts" in dashboard_counts
    policy = payload["dataset_queue_direct_start_policy"]
    assert policy["enabled"] is True
    assert policy["method"] == "POST"
    assert policy["endpoint_route_template"] == "/api/tasks/{task_id}/open"
    assert policy["same_origin_only"] is True
    assert policy["exact_route_required"] is True
    assert policy["endpoint_task_segment_must_match_row_task_id"] is True
    assert policy["expected_user_guard_required"] is True
    assert policy["post_body_expected_user_required"] is True
    assert policy["post_body_expected_user_field"] == "expected_user"
    assert policy["denied_start_returns_task_open_authorization_contract"] is True
    assert policy["denied_start_support_preserves_task_open_authorization_contract"] is True
    assert policy["denied_start_support_includes_authorization_context"] is True
    assert policy["denied_start_contract_reports_no_session_created"] is True
    assert policy["denied_start_contract_reports_server_authorizes_open_false"] is True
    assert policy["startable_task_states"] == ["pending", "in_progress"]
    assert policy["label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert policy["browser_label_write_target"] == "training_zarr"
    assert policy["csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert policy["csv_handoff_artifacts_are_label_write_targets"] is False
    assert policy["handoff_csv_artifacts_are_label_write_targets"] is False
    assert policy["intermediate_csv_artifacts_are_label_write_targets"] is False
    assert policy["browser_writes_csv_or_handoff_files"] is False
    assert policy["browser_writes_handoff_csv"] is False
    assert policy["browser_writes_intermediate_csv"] is False
    assert policy["browser_receives_zarr_write_authority"] is False
    assert policy["browser_has_direct_zarr_write_authority"] is False
    assert payload["browser_mutation_write_policy"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["browser_mutation_write_policy"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert payload["browser_mutation_write_policy"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert payload["browser_mutation_write_policy"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_policy"][
        "handoff_csv_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_policy"][
        "intermediate_csv_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert payload["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert payload["browser_mutation_write_checklist"]["ready"] is True
    assert payload["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["browser_mutation_write_checklist"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert payload["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert payload["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_checklist"][
        "handoff_csv_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_checklist"][
        "intermediate_csv_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_checklist"]["browser_writes_handoff_csv"] is False
    assert payload["browser_mutation_write_checklist"]["browser_writes_intermediate_csv"] is False
    assert payload["preflight"]["browser_mutation_write_checklist"] == payload[
        "browser_mutation_write_checklist"
    ]
    assert payload["preflight"]["dataset_queue_direct_start_policy"] == policy
    assert payload["session_guard_policy"]["recovery_entrypoints"] == [
        "/",
        "/my-datasets",
        "/datasets",
        "/my-work",
        "/work",
    ]
    response_policy = payload["browser_response_security_policy"]
    assert response_policy["protected_labeler_paths"] == [
        "/",
        "/me",
        "/labeling",
        "/my-datasets",
        "/datasets",
        "/my-work",
        "/work",
        "/identity",
        "/api/me/identity",
        "/api/me/tasks",
        "/api/me/datasets",
    ]
    assert response_policy["personalized_alias_paths"] == [
        "/my-datasets",
        "/my-work",
    ]
    assert response_policy["canonical_fallback_paths"] == ["/datasets", "/work"]
    assert response_policy["personal_api_paths"] == [
        "/api/me/identity",
        "/api/me/tasks",
        "/api/me/datasets",
    ]
    assert response_policy["personalized_alias_headers_must_match_canonical"] is True
    response_contract = _browser_response_security_contract_policy(response_policy)
    assert response_contract["ready"] is True
    assert response_contract["protected_labeler_paths_ready"] is True
    assert response_contract["personalized_alias_paths_ready"] is True
    assert response_contract["canonical_fallback_paths_ready"] is True
    assert response_contract["personal_api_paths_ready"] is True
    stale_response_policy = {
        **response_policy,
        "protected_labeler_paths": ["/", "/datasets", "/work", "/api/me/tasks"],
        "personalized_alias_paths": [],
    }
    stale_response_contract = _browser_response_security_contract_policy(
        stale_response_policy
    )
    assert stale_response_contract["ready"] is False
    assert stale_response_contract["protected_labeler_paths_ready"] is False
    assert stale_response_contract["personalized_alias_paths_ready"] is False
    handoff_response_fields = _handoff_browser_response_security_fields(
        {"browser_response_security_policy": response_policy}
    )
    assert handoff_response_fields["browser_response_security_ready"] is True
    assert handoff_response_fields[
        "browser_response_security_protected_labeler_paths"
    ] == [
        "/",
        "/me",
        "/labeling",
        "/my-datasets",
        "/datasets",
        "/my-work",
        "/work",
        "/identity",
        "/api/me/identity",
        "/api/me/tasks",
        "/api/me/datasets",
    ]
    assert handoff_response_fields[
        "browser_response_security_protected_labeler_paths_ready"
    ] is True
    assert handoff_response_fields[
        "browser_response_security_personalized_alias_paths"
    ] == [
        "/my-datasets",
        "/my-work",
    ]
    assert handoff_response_fields[
        "browser_response_security_personalized_alias_paths_ready"
    ] is True
    assert handoff_response_fields[
        "browser_response_security_canonical_fallback_paths"
    ] == ["/datasets", "/work"]
    assert handoff_response_fields[
        "browser_response_security_canonical_fallback_paths_ready"
    ] is True
    assert handoff_response_fields["browser_response_security_personal_api_paths"] == [
        "/api/me/identity",
        "/api/me/tasks",
        "/api/me/datasets",
    ]
    assert handoff_response_fields[
        "browser_response_security_personal_api_paths_ready"
    ] is True
    assert handoff_response_fields[
        "browser_response_security_personalized_alias_headers_must_match_canonical"
    ] is True
    command_templates = payload["operator_validation_command_templates"]
    assert command_templates["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert command_templates["commands_are_operator_only"] is True
    assert command_templates["commands_are_labeler_instructions"] is False
    assert command_templates["labelers_must_not_run_commands"] is True
    assert command_templates["operator_authorization_required"] is True
    assert command_templates["gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert "record_zarr_backup_evidence" in command_templates["command_ids"]
    assert "record_browser_response_security_evidence" in command_templates["command_ids"]
    assert "record_identity_source_evidence" in command_templates["command_ids"]
    assert "identity-source and personal queue evidence" in json.dumps(command_templates)
    assert "DEPLOYED_IDENTITY_PROBE_AND_PERSONAL_MY_DATASETS_URL_VERIFIED" in json.dumps(
        command_templates
    )
    assert "record_browser_smoke_evidence" in command_templates["command_ids"]
    assert "record_disposable_zarr_mutation_smoke_evidence" in command_templates["command_ids"]
    assert "record_operator_recovery_contract_gate" in command_templates["command_ids"]
    assert "apply_operator_evidence_templates" in command_templates["command_ids"]
    assert command_templates["template_backed_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert command_templates["validation_checklist_gate_ids"] == [
        "operator_recovery_contract"
    ]
    assert command_templates["apply_required_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert command_templates["evidence_template_fields_by_gate_id"][
        "browser_smoke"
    ] == "browser_smoke_evidence_template"
    assert command_templates["evidence_template_paths_by_gate_id"][
        "browser_smoke"
    ] == "browser-smoke-evidence-template.json"
    assert command_templates["evidence_templates_by_gate_id"]["browser_smoke"] == {
        "gate_id": "browser_smoke",
        "template_field": "browser_smoke_evidence_template",
        "template_path": "browser-smoke-evidence-template.json",
        "record_command_id": "record_browser_smoke_evidence",
        "apply_required_after_approval": True,
        "apply_command_id": "apply_operator_evidence_templates",
    }
    assert command_templates["commands_by_gate_id"]["browser_smoke"] == [
        "record_browser_smoke_evidence",
        "apply_operator_evidence_templates",
    ]
    assert command_templates["commands_by_gate_id"]["operator_recovery_contract"] == [
        "record_operator_recovery_contract_gate"
    ]
    assert command_templates["missing_command_gate_ids"] == []
    assert command_templates["launch_evidence_collection_plan_schema"] == (
        "palette.web_labeling_launch_evidence_collection_plan.v1"
    )
    assert command_templates["launch_evidence_collection_step_count"] == 6
    assert command_templates["launch_evidence_collection_gate_ids"] == (
        command_templates["gate_ids"]
    )
    assert "record_browser_smoke_evidence" in command_templates[
        "launch_evidence_collection_record_command_ids"
    ]
    assert command_templates["launch_evidence_collection_operator_only"] is True
    assert command_templates["launch_evidence_collection_required_final_field"] == (
        "labeler_links_safe_to_share"
    )
    assert command_templates["launch_evidence_collection_required_final_value"] is True
    assert command_templates["launch_evidence_collection_final_inspection_command"] == (
        "inspect-handoff --path PACKAGE --require-shareable"
    )
    collection_plan = command_templates["launch_evidence_collection_plan"]
    assert collection_plan["schema"] == (
        "palette.web_labeling_launch_evidence_collection_plan.v1"
    )
    assert collection_plan["operator_only"] is True
    assert collection_plan["commands_are_labeler_instructions"] is False
    assert collection_plan["labelers_must_not_run_commands"] is True
    assert collection_plan["safe_to_share_blocked_until_plan_complete"] is True
    assert collection_plan["step_count"] == 6
    assert collection_plan["required_final_field"] == "labeler_links_safe_to_share"
    assert collection_plan["required_final_value"] is True
    assert collection_plan["final_inspection_command"] == (
        "inspect-handoff --path PACKAGE --require-shareable"
    )
    assert collection_plan["steps_by_gate_id"]["browser_smoke"][
        "record_command_id"
    ] == "record_browser_smoke_evidence"
    assert collection_plan["steps_by_gate_id"]["browser_smoke"][
        "template_backed"
    ] is True
    assert collection_plan["steps_by_gate_id"]["browser_smoke"][
        "apply_required_after_approval"
    ] is True
    assert collection_plan["steps_by_gate_id"]["operator_recovery_contract"][
        "template_backed"
    ] is False
    assert collection_plan["steps_by_gate_id"]["operator_recovery_contract"][
        "apply_required_after_approval"
    ] is False
    assert "record-browser-smoke-evidence" in json.dumps(command_templates)
    assert "--personalized-dataset-queue-verified" in json.dumps(command_templates)
    assert (
        "--preferred-labeler-entry-url-matches-personal-dataset-queue"
        in json.dumps(command_templates)
    )
    assert (
        "--personalized-labeler-entry-url-matches-personal-dataset-queue"
        in json.dumps(command_templates)
    )
    assert "--personalized-work-dashboard-verified" in json.dumps(command_templates)
    assert "DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER" in json.dumps(
        command_templates
    )
    assert "SAME_USER_AS_EXPECTED_USER" in json.dumps(command_templates)
    assert payload["preflight"]["operator_validation_command_templates"] == command_templates
    assert payload["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke=missing_evidence" in payload["safe_share_next_action_summary"]
    assert payload["preflight"]["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke=missing_evidence" in payload["preflight"][
        "safe_share_next_action_summary"
    ]
    assert payload["operator_validation"]["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert payload["operator_validation"]["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert payload["operator_validation"]["operator_validation_gate_flat_field_suffixes"] == [
        "status",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]
    assert payload["preflight"]["operator_validation"]["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert payload["preflight"]["operator_validation"]["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
        assert payload["operator_validation"][f"operator_validation_gate_{gate_id}_status"] == (
            "missing_evidence"
        )
        assert payload["operator_validation"][f"operator_validation_gate_{gate_id}_pending"] is True
        assert payload["operator_validation"][f"operator_validation_gate_{gate_id}_missing_evidence"] is True
        assert payload["operator_validation"][f"operator_validation_gate_{gate_id}_needs_review"] is False
        assert payload["operator_validation"][f"operator_validation_gate_{gate_id}_passed"] is False
        assert payload["preflight"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_status"
        ] == "missing_evidence"
        assert payload["preflight"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_pending"
        ] is True


def test_browser_response_security_evidence_template_prefers_personal_alias_capture():
    payload = _browser_response_security_evidence_template(
        base_url="https://labeling.example.org",
    )

    assert payload["preferred_capture_path"] == "/my-datasets"
    assert payload["preferred_capture_url"] == "https://labeling.example.org/my-datasets"
    assert payload["expected_user_capture_query_required"] is True
    assert payload["required_capture_contract"] == {
        "schema": "palette.web_labeling_browser_response_security_capture_contract.v1",
        "preferred_capture_path": "/my-datasets",
        "preferred_capture_url": "https://labeling.example.org/my-datasets",
        "expected_user_query_required": True,
        "authenticated_test_user_required": True,
        "authenticated_test_user_must_match_expected_user": True,
        "capture_path_must_match_declared_sample_path": True,
        "personalized_alias_headers_must_match_canonical": True,
        "labeler_entrypoint": "personal_datasets_waiting_queue",
    }
    assert payload["sample_capture_paths"] == [
        "/my-datasets",
        "/labeling",
        "/datasets",
        "/my-work",
        "/work",
        "/api/me/tasks",
        "/api/me/datasets",
    ]
    assert payload["sample_capture_urls"] == [
        "https://labeling.example.org/my-datasets",
        "https://labeling.example.org/labeling",
        "https://labeling.example.org/datasets",
        "https://labeling.example.org/my-work",
        "https://labeling.example.org/work",
        "https://labeling.example.org/api/me/tasks",
        "https://labeling.example.org/api/me/datasets",
    ]
    assert payload["sample_expected_user_capture_urls"] == [
        "https://labeling.example.org/my-datasets?expected_user=USER",
        "https://labeling.example.org/labeling?expected_user=USER",
        "https://labeling.example.org/datasets?expected_user=USER",
        "https://labeling.example.org/my-work?expected_user=USER",
        "https://labeling.example.org/work?expected_user=USER",
        "https://labeling.example.org/api/me/tasks?expected_user=USER",
        "https://labeling.example.org/api/me/datasets?expected_user=USER",
    ]
    assert payload["personalized_alias_capture_paths"] == [
        "/my-datasets",
        "/my-work",
    ]
    assert payload["human_readable_queue_alias_capture_paths"] == ["/labeling"]
    assert payload["canonical_fallback_capture_paths"] == ["/datasets", "/work"]
    assert payload["api_capture_paths"] == ["/api/me/tasks", "/api/me/datasets"]
    assert payload["personalized_alias_headers_must_match_canonical"] is True
    assert payload["checks"]["expected_user_capture_query_present"] is False
    assert payload["checks"]["authenticated_test_user_present"] is False
    assert payload["checks"]["authenticated_test_user_matches_expected_user"] is False
    assert payload["checks"]["capture_url_matches_preferred_path"] is False
    assert payload["checks"]["capture_url_matches_sample_path"] is False
    assert payload["checks"]["capture_url_contract_ready"] is False
    assert payload["checks"]["authenticated_test_user_contract_ready"] is False
    assert "/my-datasets?expected_user=<user>" in "\n".join(payload["instructions"])
    assert "same user named by the expected_user query" in "\n".join(
        payload["instructions"]
    )
    assert "Use /labeling" in "\n".join(payload["instructions"])


def test_record_browser_response_security_requires_expected_user_capture_url(tmp_path):
    evidence_path = tmp_path / "browser-response-security-evidence-template.json"
    template = _browser_response_security_evidence_template(
        base_url="https://labeling.example.org",
    )
    evidence_path.write_text(json.dumps(template), encoding="utf-8")
    expected_headers = {
        str(key): str(value)
        for key, value in template["expected_headers"].items()
    }

    missing_query_report = _record_browser_response_security_evidence(
        evidence_path=evidence_path,
        headers=expected_headers,
        operator="operator",
        capture_url="https://labeling.example.org/my-datasets",
        authenticated_test_user="alice",
    )
    missing_query_payload = json.loads(evidence_path.read_text())

    assert missing_query_report["ok"] is False
    assert missing_query_report["capture_url_contract_ready"] is False
    assert missing_query_report["required_capture_contract"] == template[
        "required_capture_contract"
    ]
    assert missing_query_report["expected_user_capture_query_required"] is True
    assert missing_query_report["expected_user_query_present"] is False
    assert missing_query_report["expected_user_query_value"] == ""
    assert missing_query_report["authenticated_test_user_present"] is True
    assert missing_query_report["authenticated_test_user_matches_expected_user"] is False
    assert missing_query_report["authenticated_test_user_contract_ready"] is False
    assert missing_query_report["capture_url_matches_preferred_path"] is True
    assert missing_query_report["capture_url_matches_sample_path"] is True
    assert missing_query_report["operator_approval_status"] == "pending_operator_confirmation"
    assert "expected_user_capture_query_missing" in {
        error["error"] for error in missing_query_report["errors"]
    }
    assert missing_query_payload["checks"]["expected_user_capture_query_present"] is False
    assert missing_query_payload["checks"]["authenticated_test_user_present"] is True
    assert missing_query_payload["checks"]["authenticated_test_user_matches_expected_user"] is False
    assert missing_query_payload["checks"]["capture_url_matches_preferred_path"] is True
    assert missing_query_payload["checks"]["capture_url_contract_ready"] is False

    no_user_evidence_path = tmp_path / "browser-response-security-no-user.json"
    no_user_evidence_path.write_text(json.dumps(template), encoding="utf-8")
    no_user_report = _record_browser_response_security_evidence(
        evidence_path=no_user_evidence_path,
        headers=expected_headers,
        operator="operator",
        capture_url="https://labeling.example.org/my-datasets?expected_user=alice",
    )
    no_user_payload = json.loads(no_user_evidence_path.read_text())

    assert no_user_report["ok"] is False
    assert no_user_report["capture_url_contract_ready"] is True
    assert no_user_report["authenticated_test_user_present"] is False
    assert no_user_report["authenticated_test_user_matches_expected_user"] is False
    assert no_user_report["authenticated_test_user_contract_ready"] is False
    assert no_user_report["operator_approval_status"] == "pending_operator_confirmation"
    assert "authenticated_test_user_missing" in {
        error["error"] for error in no_user_report["errors"]
    }
    assert no_user_payload["checks"]["expected_user_capture_query_present"] is True
    assert no_user_payload["checks"]["authenticated_test_user_present"] is False
    assert no_user_payload["checks"]["authenticated_test_user_matches_expected_user"] is False
    assert no_user_payload["checks"]["authenticated_test_user_contract_ready"] is False
    no_user_status = _operator_evidence_template_status(
        gate_id="browser_response_security_headers",
        template_path=str(no_user_evidence_path),
        template=no_user_payload,
        present=True,
        valid=True,
    )
    assert no_user_status["ready"] is False
    assert no_user_status["capture_context_required"] is True
    assert no_user_status["capture_context_ready"] is False
    assert no_user_status["required_capture_contract"] == template[
        "required_capture_contract"
    ]
    assert "capture.authenticated_test_user" in no_user_status[
        "capture_context_missing_fields"
    ]

    wrong_user_evidence_path = tmp_path / "browser-response-security-wrong-user.json"
    wrong_user_evidence_path.write_text(json.dumps(template), encoding="utf-8")
    wrong_user_report = _record_browser_response_security_evidence(
        evidence_path=wrong_user_evidence_path,
        headers=expected_headers,
        operator="operator",
        capture_url="https://labeling.example.org/my-datasets?expected_user=alice",
        authenticated_test_user="bob",
    )
    wrong_user_payload = json.loads(wrong_user_evidence_path.read_text())
    wrong_user_status = _operator_evidence_template_status(
        gate_id="browser_response_security_headers",
        template_path=str(wrong_user_evidence_path),
        template=wrong_user_payload,
        present=True,
        valid=True,
    )

    assert wrong_user_report["ok"] is False
    assert wrong_user_report["expected_user_query_value"] == "alice"
    assert wrong_user_report["authenticated_test_user_present"] is True
    assert wrong_user_report["authenticated_test_user_matches_expected_user"] is False
    assert wrong_user_report["authenticated_test_user_contract_ready"] is False
    assert "authenticated_test_user_expected_user_mismatch" in {
        error["error"] for error in wrong_user_report["errors"]
    }
    assert wrong_user_payload["checks"]["authenticated_test_user_matches_expected_user"] is False
    assert wrong_user_status["ready"] is False
    assert wrong_user_status["capture_context_ready"] is False
    assert wrong_user_status["required_capture_contract"] == template[
        "required_capture_contract"
    ]
    assert "capture.authenticated_test_user_matches_expected_user" in wrong_user_status[
        "capture_context_missing_fields"
    ]

    path_only_evidence_path = tmp_path / "browser-response-security-path-only.json"
    path_only_template = _browser_response_security_evidence_template()
    path_only_evidence_path.write_text(json.dumps(path_only_template), encoding="utf-8")
    path_only_report = _record_browser_response_security_evidence(
        evidence_path=path_only_evidence_path,
        headers={
            str(key): str(value)
            for key, value in path_only_template["expected_headers"].items()
        },
        operator="operator",
        capture_url="https://labeling.example.org/my-datasets?expected_user=alice",
        authenticated_test_user="alice",
    )

    assert path_only_report["ok"] is True
    assert path_only_report["capture_url_contract_ready"] is True
    assert path_only_report["capture_url_matches_preferred_path"] is True
    assert path_only_report["capture_url_matches_sample_path"] is True

    approved_report = _record_browser_response_security_evidence(
        evidence_path=evidence_path,
        headers=expected_headers,
        operator="operator",
        capture_url="https://labeling.example.org/my-datasets?expected_user=alice",
        authenticated_test_user="alice",
    )
    approved_payload = json.loads(evidence_path.read_text())

    assert approved_report["ok"] is True
    assert approved_report["capture_url_contract_ready"] is True
    assert approved_report["required_capture_contract"] == template[
        "required_capture_contract"
    ]
    assert approved_report["expected_user_capture_query_required"] is True
    assert approved_report["expected_user_query_present"] is True
    assert approved_report["expected_user_query_value"] == "alice"
    assert approved_report["authenticated_test_user_present"] is True
    assert approved_report["authenticated_test_user_matches_expected_user"] is True
    assert approved_report["authenticated_test_user_contract_ready"] is True
    assert approved_report["capture_url_matches_preferred_path"] is True
    assert approved_report["capture_url_matches_sample_path"] is True
    assert approved_report["operator_approval_status"] == "operator_approved"
    assert approved_payload["checks"]["expected_user_capture_query_present"] is True
    assert approved_payload["checks"]["authenticated_test_user_present"] is True
    assert approved_payload["checks"]["authenticated_test_user_matches_expected_user"] is True
    assert approved_payload["checks"]["authenticated_test_user_contract_ready"] is True
    assert approved_payload["checks"]["capture_url_matches_preferred_path"] is True
    assert approved_payload["checks"]["capture_url_contract_ready"] is True
    approved_status = _operator_evidence_template_status(
        gate_id="browser_response_security_headers",
        template_path=str(evidence_path),
        template=approved_payload,
        present=True,
        valid=True,
    )
    assert approved_status["ready"] is True
    assert approved_status["capture_context_required"] is True
    assert approved_status["capture_context_ready"] is True
    assert approved_status["required_capture_contract"] == template[
        "required_capture_contract"
    ]
    assert approved_status["capture_url_expected_user"] == "alice"
    assert approved_status["authenticated_test_user_matches_expected_user"] is True
    assert approved_status["capture_context_missing_fields"] == []


def test_dataset_queue_page_renders_labeler_start_contract_fields():
    html = _datasets_html().decode("utf-8")

    assert "labeler_start_ready" in html
    assert "labeler_start_status" in html
    assert "labeler_action" in html
    assert "reassignment-session-safety" in html
    assert "page_context=dataset_queue_reassignment_session_safety" in html
    assert "reassignment_session_safety_ok" in html
    assert "reassignment_session_safety_blocks_labeler_mutation" in html
    assert "reassignment_session_safety_active_session_assignment_mismatch_count" in html
    assert "reassignment_session_safety_active_session_assignment_mismatch_recording_ids" in html
    assert "reassignment_session_safety_operator_action" in html
    assert "Copy reassignment safety" in html
    assert "data_plane_write_target" in html
    assert "mutable_label_data_plane" in html
    assert "browser_mutation_target_contract_met" in html
    assert "browser_mutation_target_mismatch_count" in html
    assert "direct_browser_start_contract_met" in html
    assert "direct_browser_start_mismatch_count" in html
    assert "single_owner_policy_contract_met" in html
    assert "browser_mutation_write_checklist_ready" in html
    assert "browser_mutation_write_checklist_label_mutation_target_kind" in html
    assert "browser_mutation_write_checklist_csv_handoff_artifact_role" in html
    assert "browser_mutation_write_checklist_csv_handoff_artifacts_are_label_write_targets" in html
    assert "browser_mutation_write_checklist_handoff_csv_artifacts_are_label_write_targets" in html
    assert "browser_mutation_write_checklist_intermediate_csv_artifacts_are_label_write_targets" in html
    assert "handoff_ready_to_send" in html
    assert "handoff_sendability_reasons" in html
    assert "operator_validation_source" in html
    assert "operator_validation_gate_count" in html
    assert "operator_validation_pending_gate_ids" in html
    assert "operator_validation_required_missing_evidence_gate_ids" in html
    assert "safe_share_next_action_summary" not in html
    assert "operator_validation_public_fields" in html
    assert "operator_validation_operator_only_fields" in html
    assert "operator_validation_operator_action_fields" in html
    assert "operator_validation_operator_action_fields_are_labeler_instructions" in html
    assert "operator_validation_labeler_visible_payloads_may_include_operator_action_fields_for_support" in html
    assert "operator_validation_labeler_visible_payloads_include_operator_only_fields" in html
    assert "operator_validation_per_user_payloads_use_public_fields_only" in html
    assert "operator_validation_top_level_operator_reports_may_include_operator_only_fields" in html
    assert "operator_validation_required_pending_gate_count" in html
    assert "operator_validation_command_template_schema" in html
    assert "operator_validation_command_template_command_count" in html
    assert "operator_validation_command_template_command_ids" in html
    assert "operator_validation_command_template_missing_command_gate_ids" in html
    assert "operator_validation_command_template_launch_evidence_collection_plan_schema" in html
    assert "operator_validation_command_template_launch_evidence_collection_step_count" in html
    assert "operator_validation_command_template_launch_evidence_collection_required_final_field" in html
    assert "safe_share_external_launch_evidence_gap_gate_ids" in html
    assert "safe_share_external_launch_evidence_gap_summary" in html
    assert "safe_share_external_launch_evidence_gap_todos" in html
    assert "safe_share_external_launch_evidence_gap_todo_fields" in html
    assert "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id" in html
    assert "operator_validation_command_template_commands_are_operator_only" in html
    assert "operator_validation_command_template_commands_are_labeler_instructions" in html
    assert "operator_validation_command_template_labelers_must_not_run_commands" in html
    assert "operator_validation_needs_review_gate_count" in html
    assert "record-browser-smoke-evidence --evidence" not in html
    assert "apply-operator-evidence-templates --path" not in html
    assert "record-zarr-backup-evidence --evidence" not in html
    assert "operator_validation_required_missing_evidence_gate_count" in html
    assert "operator_validation_operator_action" in html
    assert "links_expire_at_utc" in html
    assert "labeler_route_authorization_checklist_schema" in html
    assert "labeler_route_authorization_checklist_ready" in html
    assert "labeler_route_authorization_expected_user_must_match_resolved_user" in html
    assert "labeler_route_authorization_expected_user_matches_resolved_user" in html
    assert "labeler_route_authorization_known_assignment_store_user_required" in html
    assert "labeler_route_authorization_active_assignment_required" in html
    assert "labeler_route_authorization_active_assignment_count" in html
    assert "labeler_route_authorization_has_active_assignment" in html
    assert "labeler_route_authorization_task_open_requires_active_assignment" in html
    assert "labeler_route_authorization_task_open_requires_task_assigned_to_resolved_user" in html
    assert "labeler_route_authorization_task_open_requires_startable_task_state" in html
    assert "labeler_route_authorization_mutation_requires_current_session" in html
    assert "labeler_route_authorization_mutation_requires_current_target_token" in html
    assert "labeler_route_authorization_signed_links_are_entry_hints_not_authorization" in html
    assert "labeler_route_authorization_forwarded_expected_user_links_recheck_identity" in html
    assert "labeler_route_authorization_forwarded_signed_links_recheck_runtime_operator_validation_start_gate" in html
    assert "direct_start_policy_enabled" in html
    assert "direct_start_policy_endpoint_route_template" in html
    assert "direct_start_policy_endpoint_task_segment_must_match_row_task_id" in html
    assert "direct_start_policy_handoff_csv_artifacts_are_label_write_targets" in html
    assert "direct_start_policy_intermediate_csv_artifacts_are_label_write_targets" in html
    assert "direct_start_policy_browser_writes_csv_or_handoff_files" in html
    assert "direct_start_policy_browser_writes_handoff_csv" in html
    assert "direct_start_policy_browser_writes_intermediate_csv" in html
    assert "direct_start_policy_browser_has_direct_zarr_write_authority" in html
    assert "direct_browser_start_authorization_contract_ready" in html
    assert "direct_browser_start_expected_user_guard_required" in html
    assert "direct_browser_start_expected_user_guard_enforced_by_api" in html
    assert "direct_browser_start_server_rechecks_on_post" in html
    assert "direct_browser_start_not_ready_reason" in html
    assert "direct_browser_start_not_ready_reasons" in html
    assert "direct_browser_start_operator_action" in html
    assert "label_mutation_target_kind" in html
    assert "browser_label_write_target" in html
    assert "training_zarr_mutations_are_server_owned" in html
    assert "handoff_artifacts_are_metadata_only" in html
    assert "csv_handoff_artifact_role" in html
    assert "csv_handoff_artifacts_are_label_write_targets" in html
    assert "browser_writes_csv_or_handoff_files" in html
    assert "browser_writes_handoff_csv" in html
    assert "browser_writes_intermediate_csv" in html
    assert "page_context=dataset_queue" in html
    assert "page_context=dataset_queue_empty" in html
    assert "page_context=dataset_queue_blocked_recordings" in html
    assert "page_context=dataset_queue_session_guard_policy" in html
    assert "preferred_labeler_entrypoint" in html
    assert "preferred_labeler_entry_url" in html
    assert "preferred_labeler_entry_url_matches_dataset_queue" in html
    assert "preferred_labeler_entry_url_matches_personal_dataset_queue" in html
    assert "personalized_labeler_entry_url_matches_personal_dataset_queue" in html
    assert "queue_first_entry_contract_schema" in html
    assert "queue_first_entry_contract_ready" in html
    assert "queue_first_entry_contract_preferred_labeler_entrypoint" in html
    assert "queue_first_entry_contract_preferred_labeler_entry_url" in html
    assert "queue_first_entry_contract_personalized_labeler_entrypoint" in html
    assert "queue_first_entry_contract_personalized_labeler_entry_url" in html
    assert "queue_first_entry_contract_personalized_entry_required" in html
    assert (
        "queue_first_entry_contract_personalized_labeler_entry_url_matches_personal_dataset_queue"
        in html
    )
    assert (
        "queue_first_entry_contract_preferred_labeler_entry_url_matches_personal_dataset_queue"
        in html
    )
    assert (
        "queue_first_entry_contract_preferred_labeler_entry_url_is_expected_user_guarded"
        in html
    )
    assert (
        "queue_first_entry_contract_personalized_labeler_entry_url_is_expected_user_guarded"
        in html
    )
    assert "queue_first_entry_contract_landing_ready" in html
    assert "queue_first_entry_contract_dataset_queue_ready" in html
    assert "queue_first_entry_contract_personal_dataset_queue_ready" in html
    assert "queue_first_entry_contract_personal_work_ready" in html
    assert "queue_first_entry_contract_queue_first_paths_ready" in html
    assert "queue_first_entry_contract_datasets_waiting_aliases_ready" in html
    assert "queue_first_entry_contract_expected_user_landing_guard" in html
    assert "queue_first_entry_contract_expected_user_queue_guard" in html
    assert "queue_first_entry_contract_expected_user_dashboard_guard" in html
    assert "personal_dataset_queue_link_role" in html
    assert "dataset_queue_link_role" in html
    assert "canonical_dataset_queue_link_role" in html
    assert "personalized_dataset_queue_preview_users" in html
    assert "canonical_dataset_queue_preview_users" in html
    assert "missing_personalized_dataset_queue_preview_users" in html
    assert "all_users_have_personalized_dataset_queue_preview" in html
    assert "preferred_personal_queue_match_users" in html
    assert "missing_preferred_personal_queue_match_users" in html
    assert "all_users_have_preferred_personal_queue_match" in html
    assert "personalized_personal_queue_match_users" in html
    assert "missing_personalized_personal_queue_match_users" in html
    assert "all_users_have_personalized_personal_queue_match" in html
    assert "dataset_queue_preferred_entrypoint_counts" in html
    assert "dataset_queue_link_role_counts" in html
    assert "dataset_queue_preview_url" in html
    assert "canonical_dataset_queue_preview_url" in html
    assert "dashboard_link_role" in html
    assert "identity_probe_link_role" in html
    assert "task_links_role" in html
    assert "expected_user_dataset_queue_url" in html
    assert "expected_user_dashboard_url" in html
    assert "expected_user_personal_dataset_queue_url" in html
    assert "expected_user_personal_work_url" in html
    assert "personalized_labeler_entry_url" in html
    assert "startDatasetQueueTask" in html
    assert "startDatasetQueueTask(this)" in html
    assert "Start browser task" in html
    assert "startable queue tasks" in html
    assert "startable /" in html
    assert "Task is not startable from the queue" in html
    assert "Open dashboard fallback" in html
    assert "data-open-endpoint" in html
    assert "missing_task_open_endpoint" in html
    assert "invalid_task_open_endpoint" in html
    assert "^\\/api\\/tasks\\/([^/?#]+)\\/open$" in html
    assert "exact same-origin /api/tasks/{task_id}/open route" in html
    assert "openEndpointMatch" in html
    assert "decodeURIComponent(openEndpointMatch[1])" in html
    assert "endpoint task segment was not valid percent-encoding" in html
    assert "task_open_endpoint_mismatch" in html
    assert "did not match the task_id on this row" in html
    assert "direct_browser_start_endpoint" in html
    assert "direct_browser_start_method" in html
    assert "direct_browser_start_uses_existing_task_open_api" in html
    assert "direct_browser_start_requires_expected_user_guard" in html
    assert "non_startable_task_count" in html
    assert "dataset_queue_direct_start_policy" in html
    assert "handoff_ready_to_send" in html
    assert "handoff_sendability_reasons" in html
    assert "operator_validation_source" in html
    assert "operator_validation_gate_count" in html
    assert "operator_validation_pending_gate_ids" in html
    assert "operator_validation_required_missing_evidence_gate_ids" in html
    assert "safe_share_next_action_summary" not in html
    assert "operator_validation_public_fields" in html
    assert "operator_validation_operator_only_fields" in html
    assert "operator_validation_labeler_visible_payloads_include_operator_only_fields" in html
    assert "operator_validation_per_user_payloads_use_public_fields_only" in html
    assert "operator_validation_top_level_operator_reports_may_include_operator_only_fields" in html
    assert "operator_validation_required_pending_gate_count" in html
    assert "operator_validation_needs_review_gate_count" in html
    assert "operator_validation_required_missing_evidence_gate_count" in html
    assert "operator_validation_operator_action" in html
    assert "operator_validation_command_template_schema" in html
    assert "operator_validation_command_template_command_count" in html
    assert "operator_validation_command_template_command_ids" in html
    assert "operator_validation_command_template_missing_command_gate_ids" in html
    assert "operator_validation_command_template_launch_evidence_collection_plan_schema" in html
    assert "operator_validation_command_template_launch_evidence_collection_step_count" in html
    assert "operator_validation_command_template_launch_evidence_collection_required_final_field" in html
    assert "safe_share_external_launch_evidence_gap_gate_ids" in html
    assert "safe_share_external_launch_evidence_gap_summary" in html
    assert "safe_share_external_launch_evidence_gap_todos" in html
    assert "safe_share_external_launch_evidence_gap_todo_fields" in html
    assert "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id" in html
    assert "operator_validation_command_template_commands_are_operator_only" in html
    assert "operator_validation_command_template_commands_are_labeler_instructions" in html
    assert "operator_validation_command_template_labelers_must_not_run_commands" in html
    assert "links_expire_at_utc" in html
    assert "record-browser-smoke-evidence --evidence" not in html
    assert "apply-operator-evidence-templates --path" not in html
    assert "record-zarr-backup-evidence --evidence" not in html
    assert "direct_start_policy_enabled" in html
    assert "direct_start_policy_endpoint_route_template" in html
    assert "direct_start_policy_endpoint_task_segment_must_match_row_task_id" in html
    assert "direct_start_policy_startable_task_states" in html
    assert "direct_start_policy_handoff_csv_artifacts_are_label_write_targets" in html
    assert "direct_start_policy_intermediate_csv_artifacts_are_label_write_targets" in html
    assert "direct_start_policy_browser_writes_csv_or_handoff_files" in html
    assert "direct_start_policy_browser_writes_handoff_csv" in html
    assert "direct_start_policy_browser_writes_intermediate_csv" in html
    assert "direct_start_policy_browser_has_direct_zarr_write_authority" in html
    assert "labeler_route_authorization_checklist_schema" in html
    assert "labeler_route_authorization_checklist_ready" in html
    assert "labeler_route_authorization_expected_user_must_match_resolved_user" in html
    assert "labeler_route_authorization_expected_user_matches_resolved_user" in html
    assert "labeler_route_authorization_known_assignment_store_user_required" in html
    assert "labeler_route_authorization_active_assignment_required" in html
    assert "labeler_route_authorization_active_assignment_count" in html
    assert "labeler_route_authorization_has_active_assignment" in html
    assert "labeler_route_authorization_task_open_requires_active_assignment" in html
    assert "labeler_route_authorization_task_open_requires_task_assigned_to_resolved_user" in html
    assert "labeler_route_authorization_task_open_requires_startable_task_state" in html
    assert "labeler_route_authorization_mutation_requires_current_session" in html
    assert "labeler_route_authorization_mutation_requires_current_target_token" in html
    assert "labeler_route_authorization_signed_links_are_entry_hints_not_authorization" in html
    assert "labeler_route_authorization_forwarded_expected_user_links_recheck_identity" in html
    assert "labeler_route_authorization_forwarded_signed_links_recheck_runtime_operator_validation_start_gate" in html
    assert 'direct_browser_start_endpoint=${row.direct_browser_start_endpoint || ""}' in html
    assert 'const directStartEndpoint = task.direct_browser_start_endpoint || "";' in html
    assert "const directStartContractReady = task.direct_browser_start_authorization_contract_ready === true" in html
    assert "const directStartNotReadyReason = operatorValidationBlocksStart" in html
    assert "const directStartOperatorAction = operatorValidationBlocksStart" in html
    assert "const operatorValidationStartGate = payload.operator_validation_start_gate || {}" in html
    assert "const operatorValidationBlocksStart = operatorValidationStartGate.blocks_task_open === true" in html
    assert "operator_validation_start_gate_blocks_task_open" in html
    assert "operator_validation_mutation_gate_required" in html
    assert "operator_validation_mutation_gate_ready" in html
    assert "operator_validation_mutation_gate_blocks_browser_mutation" in html
    assert "operator_validation_mutation_gate_not_ready_reason" in html
    assert "operator_validation_mutation_gate_pending_gate_ids" in html
    assert "operator_validation_mutation_gate_required_missing_evidence_gate_ids" in html
    assert "runtime_operator_validation_gate_cli_policy_preferred_require_flag" in html
    assert "runtime_operator_validation_gate_cli_policy_protects_browser_mutations" in html
    assert "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write" in html
    assert "const canStartTask = !operatorValidationBlocksStart && directStartContractReady" in html
    assert "Start is waiting for operator validation" in html
    assert "Direct start authorization contract:" in html
    assert "Direct start not-ready reason:" in html
    assert "Direct start operator action:" in html
    assert (
        "body: JSON.stringify({client_label: navigator.userAgent, expected_user: expectedUserGuardParam || \"\"})"
        in html
    )
    assert "const startableTaskStates = new Set" in html
    assert "startableTaskStates.has(String(task.state || \"\"))" in html
    assert '"/api/tasks/" + encodeURIComponent(taskId) + "/open"' not in html
    assert '"/api/tasks/" + encodeURIComponent(row.task_id) + "/open"' not in html
    assert "Labeler action:" in html


def test_personalized_dataset_queue_http_routes_scope_to_expected_user(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            dataset_id="dataset-a",
            state="pending",
        )
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(
            task_id="task-b",
            recording_id="rec-b",
            workflow_kind="keypoints",
            dataset_id="dataset-b",
            state="pending",
        )
        link_secret = "test-link-secret"
        alice_token = _signed_task_link_token(
            task_id="task-a",
            secret=link_secret,
            ttl_seconds=600,
            expected_user="alice",
        )
        signed_link_contract = labeling_web_module._signed_link_contract_policy(
            _browser_signed_link_policy()
        )
        assert signed_link_contract["ready"] is True
        assert (
            signed_link_contract["runtime_operator_validation_start_gate_enforced"]
            is True
        )
        assert (
            signed_link_contract[
                "operator_validation_start_gate_checked_before_session_create"
            ]
            is True
        )

        with _running_labeling_server(
            store,
            user="alice",
            link_secret=link_secret,
        ) as base_url:
            ok_status, ok_content_type, ok_body = _http_request(
                base_url,
                "/api/me/datasets?expected_user=alice",
            )
            ok_payload = json.loads(ok_body.decode("utf-8"))
            recording_ids = {
                str(recording.get("recording_id") or "")
                for dataset in ok_payload["datasets"]
                for recording in dataset.get("recordings", [])
            }

            assert ok_status == 200
            assert "application/json" in ok_content_type
            assert ok_payload["ok"] is True
            assert ok_payload["user"] == "alice"
            assert ok_payload["expected_user"] == "alice"
            assert ok_payload["preferred_labeler_entry_url"] == (
                "/my-datasets?expected_user=alice"
            )
            assert ok_payload[
                "personalized_labeler_entry_url_matches_personal_dataset_queue"
            ] is True
            assert recording_ids == {"rec-a"}
            assert ok_payload["labeler_route_authorization_checklist"][
                "expected_user_matches_resolved_user"
            ] is True
            assert ok_payload["labeler_route_authorization_checklist"][
                "has_active_assignment"
            ] is True
            assert ok_payload["labeler_route_authorization_checklist"][
                "forwarded_signed_links_recheck_runtime_operator_validation_start_gate"
            ] is True

            work_status, work_content_type, work_body = _http_request(
                base_url,
                "/api/me/tasks?expected_user=alice",
            )
            work_payload = json.loads(work_body.decode("utf-8"))
            work_recording_ids = {
                str(recording.get("recording_id") or "")
                for recording in work_payload["work"]["recordings"]
            }

            assert work_status == 200
            assert "application/json" in work_content_type
            assert work_payload["ok"] is True
            assert work_payload["work"]["user"] == "alice"
            assert work_payload["work"]["expected_user"] == "alice"
            assert work_recording_ids == {"rec-a"}
            assert work_payload["work"][
                "personalized_labeler_entry_url_matches_personal_dataset_queue"
            ] is True
            assert work_payload["work"]["labeler_route_authorization_checklist"][
                "expected_user_matches_resolved_user"
            ] is True
            assert work_payload["work"]["labeler_route_authorization_checklist"][
                "has_active_assignment"
            ] is True
            assert work_payload["work"]["labeler_route_authorization_checklist"][
                "forwarded_signed_links_recheck_runtime_operator_validation_start_gate"
            ] is True

            for assigned_shell_path in (
                "/?expected_user=alice",
                "/me?expected_user=alice",
                "/my-datasets?expected_user=alice",
                "/labeling?expected_user=alice",
                "/my-work?expected_user=alice",
                "/datasets?expected_user=alice",
                "/work?expected_user=alice",
            ):
                assigned_shell_status, assigned_shell_content_type, assigned_shell_body = (
                    _http_request(base_url, assigned_shell_path)
                )
                assigned_shell_text = assigned_shell_body.decode("utf-8")

                assert assigned_shell_status == 200
                assert "text/html" in assigned_shell_content_type
                assert "unknown_labeling_user" not in assigned_shell_text
                assert "dashboard_user_mismatch" not in assigned_shell_text

            missing_expected_identity_status, missing_expected_identity_content_type, missing_expected_identity_body = (
                _http_request(
                    base_url,
                    "/identity",
                )
            )
            missing_expected_identity_text = missing_expected_identity_body.decode("utf-8")

            assert missing_expected_identity_status == 403
            assert "text/html" in missing_expected_identity_content_type
            assert "identity_expected_user_required" in missing_expected_identity_text
            assert "Expected user required: stop before labeling" in missing_expected_identity_text
            assert "Do not open labeling work from this browser identity" in missing_expected_identity_text
            assert "Open your personalized dataset queue" not in missing_expected_identity_text
            assert "Open your datasets-waiting landing page" not in missing_expected_identity_text
            assert "Open your full personalized work dashboard" not in missing_expected_identity_text
            assert "Open canonical dataset queue fallback" not in missing_expected_identity_text
            assert "Open canonical work dashboard fallback" not in missing_expected_identity_text

            missing_expected_identity_api_status, missing_expected_identity_api_content_type, missing_expected_identity_api_body = (
                _http_request(
                    base_url,
                    "/api/me/identity",
                )
            )
            missing_expected_identity_api_payload = json.loads(
                missing_expected_identity_api_body.decode("utf-8")
            )
            missing_expected_identity_api_identity = missing_expected_identity_api_payload[
                "identity"
            ]

            assert missing_expected_identity_api_status == 403
            assert "application/json" in missing_expected_identity_api_content_type
            assert missing_expected_identity_api_payload["ok"] is False
            assert (
                missing_expected_identity_api_payload["error"]
                == "identity_expected_user_required"
            )
            assert (
                missing_expected_identity_api_identity["expected_user_guard_present"]
                is False
            )
            assert (
                missing_expected_identity_api_payload[
                    "identity_probe_expected_user_guard_required"
                ]
                is True
            )
            assert (
                missing_expected_identity_api_identity[
                    "identity_probe_expected_user_guard_required"
                ]
                is True
            )
            assert (
                missing_expected_identity_api_payload[
                    "identity_probe_launch_ctas_rendered"
                ]
                is False
            )
            assert (
                missing_expected_identity_api_payload[
                    "identity_probe_launch_ctas_suppressed"
                ]
                is True
            )
            assert (
                missing_expected_identity_api_payload[
                    "identity_probe_failed_support_urls_diagnostic_only"
                ]
                is True
            )
            assert (
                "expected-user guarded identity probe"
                in missing_expected_identity_api_identity["operator_action"]
            )
            assert (
                "open the guarded dataset queue first"
                not in missing_expected_identity_api_identity["operator_action"]
            )

            mismatch_identity_status, mismatch_identity_content_type, mismatch_identity_body = (
                _http_request(
                    base_url,
                    "/identity?expected_user=bob",
                )
            )
            mismatch_identity_text = mismatch_identity_body.decode("utf-8")

            assert mismatch_identity_status == 403
            assert "text/html" in mismatch_identity_content_type
            assert "identity_user_mismatch" in mismatch_identity_text
            assert "Stop before labeling" in mismatch_identity_text
            assert "Do not open labeling work from this browser identity" in mismatch_identity_text
            assert "Open your personalized dataset queue" not in mismatch_identity_text
            assert "Open your datasets-waiting landing page" not in mismatch_identity_text
            assert "Open your full personalized work dashboard" not in mismatch_identity_text
            assert "Open canonical dataset queue fallback" not in mismatch_identity_text
            assert "Open canonical work dashboard fallback" not in mismatch_identity_text

            mismatch_identity_api_status, mismatch_identity_api_content_type, mismatch_identity_api_body = (
                _http_request(
                    base_url,
                    "/api/me/identity?expected_user=bob",
                )
            )
            mismatch_identity_api_payload = json.loads(
                mismatch_identity_api_body.decode("utf-8")
            )
            mismatch_identity_api_identity = mismatch_identity_api_payload["identity"]

            assert mismatch_identity_api_status == 403
            assert "application/json" in mismatch_identity_api_content_type
            assert mismatch_identity_api_payload["ok"] is False
            assert mismatch_identity_api_payload["error"] == "identity_user_mismatch"
            assert (
                mismatch_identity_api_payload[
                    "identity_probe_expected_user_guard_required"
                ]
                is True
            )
            assert (
                mismatch_identity_api_payload[
                    "identity_probe_launch_ctas_rendered"
                ]
                is False
            )
            assert (
                mismatch_identity_api_payload[
                    "identity_probe_launch_ctas_suppressed"
                ]
                is True
            )
            assert (
                mismatch_identity_api_payload[
                    "identity_probe_failed_support_urls_diagnostic_only"
                ]
                is True
            )
            assert "Stop before labeling" in mismatch_identity_api_identity["operator_action"]
            assert (
                "open the guarded dataset queue first"
                not in mismatch_identity_api_identity["operator_action"]
            )
            assert mismatch_identity_api_identity["matches_expected_user"] is False
            assert (
                mismatch_identity_api_identity["identity_probe_diagnostic_only"]
                is True
            )
            assert (
                mismatch_identity_api_identity[
                    "identity_probe_expected_user_guard_required"
                ]
                is True
            )
            assert (
                mismatch_identity_api_identity[
                    "identity_probe_does_not_authorize_work"
                ]
                is True
            )
            assert (
                mismatch_identity_api_identity[
                    "identity_probe_launch_ctas_rendered"
                ]
                is False
            )
            assert (
                mismatch_identity_api_identity[
                    "identity_probe_launch_ctas_suppressed"
                ]
                is True
            )
            assert (
                mismatch_identity_api_identity[
                    "identity_probe_failed_support_urls_diagnostic_only"
                ]
                is True
            )

            work_mismatch_status, work_mismatch_content_type, work_mismatch_body = (
                _http_request(
                    base_url,
                    "/api/me/tasks?expected_user=bob",
                )
            )
            work_mismatch_payload = json.loads(
                work_mismatch_body.decode("utf-8")
            )

            assert work_mismatch_status == 403
            assert "application/json" in work_mismatch_content_type
            assert work_mismatch_payload["ok"] is False
            assert work_mismatch_payload["error"] == "dashboard_user_mismatch"
            assert work_mismatch_payload["expected_user"] == "bob"
            assert work_mismatch_payload["labeler_read_authorization_contract"][
                "personal_work_reads_filtered_by_resolved_user"
            ] is True
            assert work_mismatch_payload["labeler_read_authorization_contract"][
                "expected_user_matches_resolved_user"
            ] is False
            assert work_mismatch_payload["labeler_read_authorization_contract"][
                "server_authorizes_read"
            ] is False

            signed_status, signed_content_type, signed_body = _http_request(
                base_url,
                f"/t/{alice_token}",
            )
            signed_text = signed_body.decode("utf-8")

            assert signed_status == 200
            assert "text/html" in signed_content_type
            assert "Scoped keypoint session" in signed_text
            assert "sessionId" in signed_text
            assert "/api/sessions/${encodeURIComponent(sessionId)}/complete" in signed_text

            with _running_labeling_server(
                store,
                user="bob",
                link_secret=link_secret,
            ) as bob_signed_base_url:
                bob_signed_status, bob_signed_content_type, bob_signed_body = (
                    _http_request(
                        bob_signed_base_url,
                        f"/t/{alice_token}",
                    )
                )
                bob_signed_text = bob_signed_body.decode("utf-8")

                assert bob_signed_status == 403
                assert "text/html" in bob_signed_content_type
                assert "signed_link_user_mismatch" in bob_signed_text
                assert "expected_user=alice" in bob_signed_text
                assert "task_open_authorization_contract" in bob_signed_text

            with _running_labeling_server(
                store,
                user="alice",
                link_secret=link_secret,
                require_operator_validation_for_start=True,
            ) as gated_signed_base_url:
                gated_signed_status, gated_signed_content_type, gated_signed_body = (
                    _http_request(
                        gated_signed_base_url,
                        f"/t/{alice_token}",
                    )
                )
                gated_signed_text = gated_signed_body.decode("utf-8")

                assert gated_signed_status == 409
                assert "text/html" in gated_signed_content_type
                assert "operator_validation_start_blocked" in gated_signed_text
                assert "task_open_authorization_contract" in gated_signed_text
                assert "operator_validation_start_gate_blocks_task_open" in gated_signed_text

            open_status, open_content_type, open_body = _http_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            open_payload = json.loads(open_body.decode("utf-8"))

            assert open_status == 200
            assert "application/json" in open_content_type
            assert open_payload["ok"] is True
            assert open_payload["session"]["task_id"] == "task-a"
            assert open_payload["session"]["recording_id"] == "rec-a"
            assert open_payload["session"]["user"] == "alice"
            assert open_payload["task_open_authorization_contract"]["ready"] is True
            assert open_payload["task_open_authorization_contract"][
                "expected_user_matches_resolved_user"
            ] is True
            assert open_payload["task_open_authorization_contract"][
                "task_assigned_to_resolved_user"
            ] is True
            assert open_payload["task_open_authorization_contract"][
                "server_authorizes_open"
            ] is True
            session_id = str(open_payload["session"]["session_id"])

            with _running_labeling_server(store, user="bob") as bob_base_url:
                bob_session_completion_status, bob_session_completion_content_type, bob_session_completion_body = (
                    _http_request(
                        bob_base_url,
                        f"/api/sessions/{session_id}/complete",
                        method="POST",
                    )
                )
                bob_session_completion_payload = json.loads(
                    bob_session_completion_body.decode("utf-8")
                )

                assert bob_session_completion_status == 403
                assert "application/json" in bob_session_completion_content_type
                assert bob_session_completion_payload["ok"] is False
                assert bob_session_completion_payload["error"] == "session_user_mismatch"
                assert bob_session_completion_payload[
                    "task_completion_authorization_contract"
                ]["current_session_required"] is True
                assert bob_session_completion_payload[
                    "task_completion_authorization_contract"
                ]["current_session_present"] is True
                assert bob_session_completion_payload[
                    "task_completion_authorization_contract"
                ]["session_owned_by_resolved_user"] is False
                assert bob_session_completion_payload[
                    "task_completion_authorization_contract"
                ]["server_authorizes_completion"] is False

                bob_session_save_status, bob_session_save_content_type, bob_session_save_body = (
                    _http_request(
                        bob_base_url,
                        f"/api/sessions/{session_id}/keypoints/save",
                        method="POST",
                        payload={"points": [], "target_token": "stale-or-wrong-user"},
                    )
                )
                bob_session_save_payload = json.loads(
                    bob_session_save_body.decode("utf-8")
                )

                assert bob_session_save_status == 403
                assert "application/json" in bob_session_save_content_type
                assert bob_session_save_payload["ok"] is False
                assert bob_session_save_payload["error"] == "session_user_mismatch"
                assert bob_session_save_payload[
                    "mutation_authorization_contract"
                ]["session_owned_by_resolved_user"] is False
                assert bob_session_save_payload[
                    "mutation_authorization_contract"
                ]["server_authorizes_mutation"] is False

            second_open_status, second_open_content_type, second_open_body = _http_request(
                base_url,
                "/api/tasks/task-a/open",
                method="POST",
                payload={"expected_user": "alice"},
            )
            second_open_payload = json.loads(second_open_body.decode("utf-8"))

            assert second_open_status == 200
            assert "application/json" in second_open_content_type
            assert second_open_payload["ok"] is True
            assert second_open_payload["session"]["task_id"] == "task-a"
            assert second_open_payload["session"]["superseded_session_ids"] == [
                session_id
            ]
            active_session_id = str(second_open_payload["session"]["session_id"])

            stale_completion_status, stale_completion_content_type, stale_completion_body = (
                _http_request(
                    base_url,
                    f"/api/sessions/{session_id}/complete",
                    method="POST",
                )
            )
            stale_completion_payload = json.loads(
                stale_completion_body.decode("utf-8")
            )

            assert stale_completion_status == 409
            assert "application/json" in stale_completion_content_type
            assert stale_completion_payload["ok"] is False
            assert stale_completion_payload["error"] == "session_superseded"
            assert stale_completion_payload[
                "task_completion_authorization_contract"
            ]["current_session_required"] is True
            assert stale_completion_payload[
                "task_completion_authorization_contract"
            ]["current_session_present"] is True
            assert stale_completion_payload[
                "task_completion_authorization_contract"
            ]["session_owned_by_resolved_user"] is True
            assert stale_completion_payload[
                "task_completion_authorization_contract"
            ]["server_authorizes_completion"] is False

            stale_save_status, stale_save_content_type, stale_save_body = (
                _http_request(
                    base_url,
                    f"/api/sessions/{session_id}/keypoints/save",
                    method="POST",
                    payload={"points": [], "target_token": "superseded-session"},
                )
            )
            stale_save_payload = json.loads(stale_save_body.decode("utf-8"))

            assert stale_save_status == 409
            assert "application/json" in stale_save_content_type
            assert stale_save_payload["ok"] is False
            assert stale_save_payload["error"] == "session_superseded"
            assert stale_save_payload["mutation_authorization_contract"][
                "session_owned_by_resolved_user"
            ] is True
            assert stale_save_payload["mutation_authorization_contract"][
                "current_session_result"
            ] == "superseded"
            assert stale_save_payload["mutation_authorization_contract"][
                "server_authorizes_mutation"
            ] is False

            completion_mismatch_status, completion_mismatch_content_type, completion_mismatch_body = (
                _http_request(
                    base_url,
                    "/api/tasks/task-a/complete",
                    method="POST",
                    payload={"expected_user": "bob", "session_id": session_id},
                )
            )
            completion_mismatch_payload = json.loads(
                completion_mismatch_body.decode("utf-8")
            )

            assert completion_mismatch_status == 403
            assert "application/json" in completion_mismatch_content_type
            assert completion_mismatch_payload["ok"] is False
            assert completion_mismatch_payload["error"] == "task_complete_user_mismatch"
            assert completion_mismatch_payload[
                "task_completion_authorization_contract"
            ]["expected_user_matches_resolved_user"] is False
            assert completion_mismatch_payload[
                "task_completion_authorization_contract"
            ]["server_authorizes_completion"] is False

            completion_missing_session_status, completion_missing_session_content_type, completion_missing_session_body = (
                _http_request(
                    base_url,
                    "/api/tasks/task-a/complete",
                    method="POST",
                    payload={"expected_user": "alice"},
                )
            )
            completion_missing_session_payload = json.loads(
                completion_missing_session_body.decode("utf-8")
            )

            assert completion_missing_session_status == 400
            assert "application/json" in completion_missing_session_content_type
            assert completion_missing_session_payload["ok"] is False
            assert completion_missing_session_payload["error"] == "session_required"
            assert completion_missing_session_payload[
                "task_completion_authorization_contract"
            ]["current_session_required"] is True
            assert completion_missing_session_payload[
                "task_completion_authorization_contract"
            ]["current_session_present"] is False
            assert completion_missing_session_payload[
                "task_completion_authorization_contract"
            ]["task_assigned_to_resolved_user"] is True
            assert completion_missing_session_payload[
                "task_completion_authorization_contract"
            ]["server_authorizes_completion"] is False

            completion_cross_assignee_status, completion_cross_assignee_content_type, completion_cross_assignee_body = (
                _http_request(
                    base_url,
                    "/api/tasks/task-b/complete",
                    method="POST",
                    payload={"expected_user": "alice", "session_id": session_id},
                )
            )
            completion_cross_assignee_payload = json.loads(
                completion_cross_assignee_body.decode("utf-8")
            )

            assert completion_cross_assignee_status == 403
            assert "application/json" in completion_cross_assignee_content_type
            assert completion_cross_assignee_payload["ok"] is False
            assert completion_cross_assignee_payload["error"] == "not_assigned"
            assert completion_cross_assignee_payload[
                "task_completion_authorization_contract"
            ]["task_assigned_to_resolved_user"] is False
            assert completion_cross_assignee_payload[
                "task_completion_authorization_contract"
            ]["server_authorizes_completion"] is False

            open_mismatch_status, open_mismatch_content_type, open_mismatch_body = (
                _http_request(
                    base_url,
                    "/api/tasks/task-a/open",
                    method="POST",
                    payload={"expected_user": "bob"},
                )
            )
            open_mismatch_payload = json.loads(
                open_mismatch_body.decode("utf-8")
            )

            assert open_mismatch_status == 403
            assert "application/json" in open_mismatch_content_type
            assert open_mismatch_payload["ok"] is False
            assert open_mismatch_payload["error"] == "task_open_user_mismatch"
            assert open_mismatch_payload["expected_user"] == "bob"
            assert open_mismatch_payload["task_open_authorization_contract"][
                "ready"
            ] is False
            assert open_mismatch_payload["task_open_authorization_contract"][
                "expected_user_matches_resolved_user"
            ] is False
            assert open_mismatch_payload["task_open_authorization_contract"][
                "server_authorizes_open"
            ] is False

            cross_assignee_status, cross_assignee_content_type, cross_assignee_body = (
                _http_request(
                    base_url,
                    "/api/tasks/task-b/open",
                    method="POST",
                    payload={"expected_user": "alice"},
                )
            )
            cross_assignee_payload = json.loads(
                cross_assignee_body.decode("utf-8")
            )

            assert cross_assignee_status == 403
            assert "application/json" in cross_assignee_content_type
            assert cross_assignee_payload["ok"] is False
            assert cross_assignee_payload["error"] == "not_assigned"
            assert cross_assignee_payload["task_open_authorization_contract"][
                "task_assigned_to_resolved_user"
            ] is False
            assert cross_assignee_payload["task_open_authorization_contract"][
                "server_authorizes_open"
            ] is False

            mismatch_status, mismatch_content_type, mismatch_body = _http_request(
                base_url,
                "/api/me/datasets?expected_user=bob",
            )
            mismatch_payload = json.loads(mismatch_body.decode("utf-8"))

            assert mismatch_status == 403
            assert "application/json" in mismatch_content_type
            assert mismatch_payload["ok"] is False
            assert mismatch_payload["error"] == "dashboard_user_mismatch"
            assert mismatch_payload["expected_user"] == "bob"
            read_contract = mismatch_payload["labeler_read_authorization_contract"]
            assert read_contract[
                "expected_user_matches_resolved_user"
            ] is False
            assert read_contract["dataset_queue_reads_filtered_by_resolved_user"] is True
            assert read_contract["labeler_visible_scope"] == (
                "assigned_recordings_for_resolved_user"
            )
            assert read_contract["server_authorizes_read"] is False
            assert read_contract["server_authorizes_task_open"] is False
            assert read_contract["server_authorizes_mutation"] is False

            html_status, html_content_type, html_body = _http_request(
                base_url,
                "/my-datasets?expected_user=bob",
            )
            html_text = html_body.decode("utf-8")

            assert html_status == 403
            assert "text/html" in html_content_type
            assert "dashboard_user_mismatch" in html_text
            assert "expected_user=bob" in html_text
            assert "labeler_read_authorization_contract" in html_text

            with _running_labeling_server(store, user="charlie") as charlie_base_url:
                unknown_status, unknown_content_type, unknown_body = _http_request(
                    charlie_base_url,
                    "/api/me/datasets?expected_user=charlie",
                )
                unknown_payload = json.loads(unknown_body.decode("utf-8"))

                assert unknown_status == 403
                assert "application/json" in unknown_content_type
                assert unknown_payload["ok"] is False
                assert unknown_payload["error"] == "unknown_labeling_user"
                assert unknown_payload["expected_user"] == "charlie"
                assert unknown_payload["known_user_status"][
                    "is_known_labeler"
                ] is False
                assert unknown_payload[
                    "expected_user_personal_dataset_queue_url"
                ] == "/my-datasets?expected_user=charlie"
                assert unknown_payload["preferred_labeler_entry_url"] == (
                    "/my-datasets?expected_user=charlie"
                )
                assert unknown_payload["labeler_route_authorization_checklist"][
                    "known_assignment_store_user"
                ] is False
                assert unknown_payload["labeler_route_authorization_checklist"][
                    "has_active_assignment"
                ] is False
                assert unknown_payload["labeler_route_authorization_checklist"][
                    "ready"
                ] is False

                unknown_work_status, unknown_work_content_type, unknown_work_body = (
                    _http_request(
                        charlie_base_url,
                        "/api/me/tasks?expected_user=charlie",
                    )
                )
                unknown_work_payload = json.loads(
                    unknown_work_body.decode("utf-8")
                )

                assert unknown_work_status == 403
                assert "application/json" in unknown_work_content_type
                assert unknown_work_payload["ok"] is False
                assert unknown_work_payload["error"] == "unknown_labeling_user"
                assert unknown_work_payload["expected_user"] == "charlie"
                assert unknown_work_payload["known_user_status"][
                    "is_known_labeler"
                ] is False
                assert unknown_work_payload["labeler_route_authorization_checklist"][
                    "known_assignment_store_user"
                ] is False
                assert unknown_work_payload["labeler_route_authorization_checklist"][
                    "ready"
                ] is False

                unknown_identity_status, unknown_identity_content_type, unknown_identity_body = (
                    _http_request(
                        charlie_base_url,
                        "/identity?expected_user=charlie",
                    )
                )
                unknown_identity_text = unknown_identity_body.decode("utf-8")

                assert unknown_identity_status == 403
                assert "text/html" in unknown_identity_content_type
                assert "unknown_labeling_user" in unknown_identity_text
                assert "/my-datasets?expected_user=charlie" in unknown_identity_text
                assert "Stop before labeling" in unknown_identity_text
                assert "assigned any active labeling recording" in unknown_identity_text
                assert "Do not open labeling work from this browser identity" in unknown_identity_text
                assert "Open your personalized dataset queue" not in unknown_identity_text
                assert "Open your datasets-waiting landing page" not in unknown_identity_text
                assert "Open your full personalized work dashboard" not in unknown_identity_text
                assert "Open canonical dataset queue fallback" not in unknown_identity_text
                assert "Open canonical work dashboard fallback" not in unknown_identity_text
                assert "open the guarded dataset queue first" not in unknown_identity_text

                unknown_identity_api_status, unknown_identity_api_content_type, unknown_identity_api_body = (
                    _http_request(
                        charlie_base_url,
                        "/api/me/identity?expected_user=charlie",
                    )
                )
                unknown_identity_api_payload = json.loads(
                    unknown_identity_api_body.decode("utf-8")
                )

                assert unknown_identity_api_status == 403
                assert "application/json" in unknown_identity_api_content_type
                assert unknown_identity_api_payload["ok"] is False
                assert unknown_identity_api_payload["error"] == "unknown_labeling_user"
                assert unknown_identity_api_payload["identity_probe_diagnostic_only"] is True
                assert unknown_identity_api_payload["identity_probe_does_not_authorize_work"] is True
                assert unknown_identity_api_payload[
                    "identity_probe_unknown_user_blocks_work_surfaces"
                ] is True
                assert (
                    unknown_identity_api_payload[
                        "identity_probe_expected_user_guard_required"
                    ]
                    is True
                )
                assert (
                    unknown_identity_api_payload[
                        "identity_probe_launch_ctas_rendered"
                    ]
                    is False
                )
                assert (
                    unknown_identity_api_payload[
                        "identity_probe_launch_ctas_suppressed"
                    ]
                    is True
                )
                assert (
                    unknown_identity_api_payload[
                        "identity_probe_failed_support_urls_diagnostic_only"
                    ]
                    is True
                )
                unknown_identity = unknown_identity_api_payload["identity"]
                assert "Stop before labeling" in unknown_identity["operator_action"]
                assert (
                    "assigned any active labeling recording"
                    in unknown_identity["operator_action"]
                )
                assert (
                    "open the guarded dataset queue first"
                    not in unknown_identity["operator_action"]
                )
                assert unknown_identity["matches_known_labeling_user"] is False
                assert unknown_identity["known_assignment_store_user"] is False
                assert unknown_identity["identity_probe_diagnostic_only"] is True
                assert (
                    unknown_identity["identity_probe_expected_user_guard_required"]
                    is True
                )
                assert unknown_identity["identity_probe_does_not_authorize_work"] is True
                assert (
                    unknown_identity[
                        "identity_probe_unknown_user_blocks_work_surfaces"
                    ]
                    is True
                )
                assert unknown_identity["identity_probe_launch_ctas_rendered"] is False
                assert unknown_identity["identity_probe_launch_ctas_suppressed"] is True
                assert (
                    unknown_identity[
                        "identity_probe_failed_support_urls_diagnostic_only"
                    ]
                    is True
                )

                unknown_html_status, unknown_html_content_type, unknown_html_body = (
                    _http_request(
                        charlie_base_url,
                        "/my-datasets?expected_user=charlie",
                    )
                )
                unknown_html_text = unknown_html_body.decode("utf-8")

                assert unknown_html_status == 403
                assert "text/html" in unknown_html_content_type
                assert "unknown_labeling_user" in unknown_html_text
                assert "expected_user=charlie" in unknown_html_text
                assert "labeler_read_authorization_contract" in unknown_html_text

                unknown_labeling_status, unknown_labeling_content_type, unknown_labeling_body = (
                    _http_request(
                        charlie_base_url,
                        "/labeling?expected_user=charlie",
                    )
                )
                unknown_labeling_text = unknown_labeling_body.decode("utf-8")

                assert unknown_labeling_status == 403
                assert "text/html" in unknown_labeling_content_type
                assert "unknown_labeling_user" in unknown_labeling_text
                assert "expected_user=charlie" in unknown_labeling_text
                assert "labeler_read_authorization_contract" in unknown_labeling_text

                unknown_my_work_status, unknown_my_work_content_type, unknown_my_work_body = (
                    _http_request(
                        charlie_base_url,
                        "/my-work?expected_user=charlie",
                    )
                )
                unknown_my_work_text = unknown_my_work_body.decode("utf-8")

                assert unknown_my_work_status == 403
                assert "text/html" in unknown_my_work_content_type
                assert "unknown_labeling_user" in unknown_my_work_text
                assert "expected_user=charlie" in unknown_my_work_text
                assert "labeler_read_authorization_contract" in unknown_my_work_text

                for unknown_shell_path in (
                    "/?expected_user=charlie",
                    "/me?expected_user=charlie",
                    "/datasets?expected_user=charlie",
                    "/work?expected_user=charlie",
                ):
                    unknown_shell_status, unknown_shell_content_type, unknown_shell_body = (
                        _http_request(charlie_base_url, unknown_shell_path)
                    )
                    unknown_shell_text = unknown_shell_body.decode("utf-8")

                    assert unknown_shell_status == 403
                    assert "text/html" in unknown_shell_content_type
                    assert "unknown_labeling_user" in unknown_shell_text
                    assert "expected_user=charlie" in unknown_shell_text
                    assert "labeler_read_authorization_contract" in unknown_shell_text

            session_completion_status, session_completion_content_type, session_completion_body = (
                _http_request(
                    base_url,
                    f"/api/sessions/{active_session_id}/complete",
                    method="POST",
                )
            )
            session_completion_payload = json.loads(
                session_completion_body.decode("utf-8")
            )

            assert session_completion_status == 200
            assert "application/json" in session_completion_content_type
            assert session_completion_payload["ok"] is True
            assert session_completion_payload["task"]["task_id"] == "task-a"
            assert session_completion_payload["task"]["state"] == "complete"
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["ready"] is True
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["expected_user_matches_resolved_user"] is True
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["current_session_required"] is True
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["current_session_present"] is True
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["session_owned_by_resolved_user"] is True
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["session_task_matches_requested_task"] is True
            assert session_completion_payload[
                "task_completion_authorization_contract"
            ]["server_authorizes_completion"] is True
            assert session_completion_payload[
                "post_completion_return_personal_dataset_queue_expected_user_guarded"
            ] is True
            assert session_completion_payload[
                "post_completion_return_personal_dataset_queue_url"
            ] == "/my-datasets?expected_user=alice"
    finally:
        store.close()


def test_work_dataset_queue_task_advertises_direct_start_only_for_startable_states():
    pending = _work_dataset_queue_task(
        {"task_id": "task-pending", "recording_id": "rec-a", "workflow_kind": "keypoints", "state": "pending"},
        expected_user="alice",
    )
    in_progress = _work_dataset_queue_task(
        {
            "task_id": "task-progress",
            "recording_id": "rec-a",
            "workflow_kind": "keypoints",
            "state": "in_progress",
        },
        expected_user="alice",
    )
    complete = _work_dataset_queue_task(
        {"task_id": "task-complete", "recording_id": "rec-a", "workflow_kind": "keypoints", "state": "complete"},
        expected_user="alice",
    )
    blocked = _work_dataset_queue_task(
        {"task_id": "task-blocked", "recording_id": "rec-a", "workflow_kind": "keypoints", "state": "blocked"},
        expected_user="alice",
    )
    stale_session_blocked = _work_dataset_queue_task(
        {"task_id": "task-stale", "recording_id": "rec-a", "workflow_kind": "keypoints", "state": "pending"},
        expected_user="alice",
        reassignment_session_safety_blocked=True,
    )

    assert pending["labeler_start_ready"] is True
    assert pending["direct_browser_start_endpoint"] == "/api/tasks/task-pending/open"
    assert pending["direct_browser_start_authorization_contract_ready"] is True
    assert pending["direct_browser_start_not_ready_reason"] == ""
    assert pending["direct_browser_start_not_ready_reasons"] == []
    assert pending["direct_browser_start_operator_action"] == ""
    assert pending["direct_browser_start_expected_user_guard_required"] is True
    assert pending["direct_browser_start_expected_user_guard_enforced_by_api"] is True
    assert pending["direct_browser_start_server_rechecks_on_post"] is True
    assert pending["direct_browser_start_authorization_contract"]["ready"] is True
    assert pending["direct_browser_start_authorization_contract"]["not_ready_reason"] == ""
    assert pending["direct_browser_start_authorization_contract"]["not_ready_reasons"] == []
    assert pending["direct_browser_start_authorization_contract"]["expected_user"] == "alice"
    assert pending["direct_browser_start_authorization_contract"][
        "expected_user_guard_enforced_by_api"
    ] is True
    assert pending["direct_browser_start_authorization_contract"][
        "task_assigned_to_expected_user"
    ] is True
    assert pending["direct_browser_start_authorization_contract"]["task_state_startable"] is True
    assert pending["direct_browser_start_authorization_contract"]["server_rechecks_on_post"] is True
    assert pending["direct_browser_start_authorization_contract"]["client_authorizes_open"] is False
    assert pending["direct_browser_start_authorization_contract"]["server_authorizes_open"] is True
    assert pending["direct_browser_start_authorization_contract"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert pending["direct_browser_start_authorization_contract"][
        "browser_has_direct_zarr_write_authority"
    ] is False
    assert in_progress["labeler_start_ready"] is True
    assert in_progress["direct_browser_start_endpoint"] == "/api/tasks/task-progress/open"
    assert in_progress["direct_browser_start_authorization_contract_ready"] is True
    assert in_progress["direct_browser_start_authorization_contract"]["ready"] is True
    assert complete["labeler_start_ready"] is False
    assert complete["direct_browser_start_endpoint"] == ""
    assert complete["direct_browser_start_authorization_contract_ready"] is False
    assert complete["direct_browser_start_not_ready_reason"] == "task_complete"
    assert complete["direct_browser_start_not_ready_reasons"] == ["task_complete"]
    assert complete["direct_browser_start_operator_action"] == "reopen_or_move_task_to_startable_state"
    assert complete["direct_browser_start_authorization_contract"]["ready"] is False
    assert complete["direct_browser_start_authorization_contract"]["not_ready_reason"] == "task_complete"
    assert complete["direct_browser_start_authorization_contract"]["not_ready_reasons"] == [
        "task_complete"
    ]
    assert complete["direct_browser_start_authorization_contract"]["task_state_startable"] is False
    assert blocked["labeler_start_ready"] is False
    assert blocked["direct_browser_start_endpoint"] == ""
    assert blocked["direct_browser_start_authorization_contract_ready"] is False
    assert blocked["direct_browser_start_not_ready_reason"] == "task_not_startable"
    assert blocked["direct_browser_start_not_ready_reasons"] == ["task_not_startable"]
    assert blocked["direct_browser_start_operator_action"] == "reopen_or_move_task_to_startable_state"
    assert blocked["direct_browser_start_authorization_contract"]["ready"] is False
    assert blocked["direct_browser_start_authorization_contract"]["not_ready_reason"] == "task_not_startable"
    assert blocked["direct_browser_start_authorization_contract"]["not_ready_reasons"] == [
        "task_not_startable"
    ]
    assert blocked["direct_browser_start_authorization_contract"]["task_state_startable"] is False
    assert stale_session_blocked["labeler_start_ready"] is False
    assert stale_session_blocked["direct_browser_start_endpoint"] == ""
    assert stale_session_blocked["direct_browser_start_not_ready_reason"] == (
        "reassignment_session_safety_failed"
    )
    assert stale_session_blocked["direct_browser_start_not_ready_reasons"] == [
        "reassignment_session_safety_failed"
    ]
    assert "repair-reassignment-sessions" in stale_session_blocked["direct_browser_start_operator_action"]
    assert stale_session_blocked["direct_browser_start_authorization_contract"][
        "reassignment_session_safety_passed"
    ] is False
    assert stale_session_blocked["direct_browser_start_authorization_contract"]["operator_action"] == (
        stale_session_blocked["direct_browser_start_operator_action"]
    )


def test_task_open_preflight_rejects_non_startable_task_state(tmp_path):
    task_state_policy = _browser_task_state_policy()
    assert task_state_policy["startable_task_states"] == ["pending", "in_progress"]
    assert task_state_policy["non_startable_task_open_requests"] == "reject_task_not_startable"
    assert task_state_policy["non_startable_task_save_requests"] == "reject_task_not_startable"
    session_guard_policy = _session_guard_policy()
    assert session_guard_policy["non_startable_task_sessions_rejected"] is True
    assert session_guard_policy["rejection_errors"]["task_not_startable"] == "task_not_startable"

    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-blocked",
            recording_id="rec-a",
            workflow_kind="keypoints",
            state="blocked",
        )

        error = _task_open_preflight_error(store, task_id="task-blocked", user="alice")

        assert error is not None
        code, details, status = error
        assert code == "task_not_startable"
        assert "state blocked" in str(details)
        assert status.value == 409
        with pytest.raises(PermissionError, match="startable labeling state"):
            store.create_session(task_id="task-blocked", user="alice", ttl_seconds=600)
    finally:
        store.close()


def test_task_open_preflight_and_session_creation_reject_reassignment_safety_issue(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(
            recording_id="rec-a",
            assignee_user="bob",
            allow_stale_open_sessions=True,
        )

        error = _task_open_preflight_error(store, task_id="task-a", user="bob")

        assert error is not None
        code, details, status = error
        assert code == "reassignment_session_safety_failed"
        assert "previous-owner sessions" in str(details)
        assert status.value == 409
        mismatched = store.active_assignment_mismatched_sessions_for_recording("rec-a")
        assert len(mismatched) == 1
        assert mismatched[0]["user"] == "alice"
        with pytest.raises(PermissionError, match="previous-owner sessions"):
            store.create_session(task_id="task-a", user="bob", ttl_seconds=600)

        work = store.task_summary_for_user("bob")
        check_report = _store_consistency_report(store)
        _add_work_summary_fields(
            work,
            reassignment_session_safety=check_report["reassignment_session_safety"],
        )
        queue_task = work["dataset_queue"][0]["recordings"][0]["tasks"][0]
        assert work["labeler_start_ready"] is False
        assert work["labeler_start_status"] == "reassignment_session_safety_failed"
        assert work["labeler_action"] == "wait_for_operator"
        assert work["dataset_queue_state"]["blocks_labeler_start"] is True
        assert work["reassignment_session_safety"]["ok"] is False
        assert work["reassignment_session_safety_ok"] is False
        assert work["reassignment_session_safety_blocks_labeler_mutation"] is True
        assert work["reassignment_session_safety_active_session_assignment_mismatch_count"] == 1
        assert work[
            "reassignment_session_safety_active_session_assignment_mismatch_recording_ids"
        ] == ["rec-a"]
        assert queue_task["labeler_start_ready"] is False
        assert queue_task["labeler_action"] == "wait_for_operator"
        assert queue_task["blocked_reason"] == "reassignment_session_safety_failed"
        assert queue_task["direct_browser_start_endpoint"] == ""
        assert queue_task["operator_support"]["blocked_reason"] == "reassignment_session_safety_failed"
    finally:
        store.close()


def test_non_startable_tasks_do_not_make_dataset_queue_ready(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-blocked",
            recording_id="rec-a",
            workflow_kind="keypoints",
            dataset_id="dataset-a",
            state="blocked",
        )

        work = store.task_summary_for_user("alice")
        _add_work_summary_fields(work)

        assert work["startable_task_count"] == 0
        assert work["incomplete_task_count"] == 1
        assert work["non_startable_task_count"] == 1
        workflow_state = work["recordings"][0]["workflow_state_counts"]["keypoints"]
        assert workflow_state["startable"] == 0
        assert workflow_state["non_startable"] == 1
        assert workflow_state["incomplete"] == 1
        assert work["empty_state"]["code"] == "no_open_tasks"
        assert work["progress_summary"]["open_task_count"] == 0
        assert work["progress_summary"]["waiting_recording_count"] == 0
        assert work["progress_summary"]["blocked_recording_count"] == 1
        assert work["progress_summary"]["blocked_recordings_by_reason"] == {
            "non_startable_task_state": 1
        }
        assert work["dataset_queue_summary"]["waiting_dataset_count"] == 0
        assert work["dataset_queue_summary"]["open_task_count"] == 0
        assert work["dataset_queue_summary"]["non_startable_task_count"] == 1
        assert work["dataset_queue_state"]["code"] == "assigned_recordings_need_operator_action"
        assert work["dataset_queue_state"]["has_open_dataset_work"] is False
        assert work["labeler_start_ready"] is False
        queue_dataset = work["dataset_queue"][0]
        queue_task = queue_dataset["recordings"][0]["tasks"][0]
        assert queue_dataset["labeler_start_ready"] is False
        assert queue_dataset["open_task_count"] == 0
        assert queue_dataset["non_startable_task_count"] == 1
        assert queue_task["labeler_start_ready"] is False
        assert queue_task["direct_browser_start_endpoint"] == ""
    finally:
        store.close()


def test_dashboard_page_renders_dataset_queue_start_contract_fields():
    html = _dashboard_html().decode("utf-8")

    assert "dashboardQueueSupportText" in html
    assert "labeler_start_ready" in html
    assert "labeler_action" in html
    assert "data_plane_write_target" in html
    assert "authoritative_label_state" in html
    assert "mutable_label_data_plane" in html
    assert "browser_mutation_target_contract_met" in html
    assert "browser_mutation_target_mismatch_count" in html
    assert "direct_browser_start_contract_met" in html
    assert "direct_browser_start_mismatch_count" in html
    assert "single_owner_policy_contract_met" in html
    assert "browser_mutation_write_checklist_ready" in html
    assert "browser_mutation_write_checklist_label_mutation_target_kind" in html
    assert "browser_mutation_write_checklist_browser_label_write_target" in html
    assert "browser_mutation_write_checklist_csv_handoff_artifact_role" in html
    assert "browser_mutation_write_checklist_csv_handoff_artifacts_are_label_write_targets" in html
    assert "browser_mutation_write_checklist_handoff_csv_artifacts_are_label_write_targets" in html
    assert "browser_mutation_write_checklist_intermediate_csv_artifacts_are_label_write_targets" in html
    assert "label_mutation_target_kind" in html
    assert "browser_label_write_target" in html
    assert "training_zarr_mutations_are_server_owned" in html
    assert "handoff_artifacts_are_metadata_only" in html
    assert "csv_handoff_artifact_role" in html
    assert "csv_handoff_artifacts_are_label_write_targets" in html
    assert "handoff_csv_artifacts_are_label_write_targets" in html
    assert "intermediate_csv_artifacts_are_label_write_targets" in html
    assert "browser_writes_csv_or_handoff_files" in html
    assert "browser_writes_handoff_csv" in html
    assert "browser_writes_intermediate_csv" in html
    assert "non_startable_task_count" in html
    assert "startable /" in html
    assert "non-startable" in html
    assert "startable / total tasks" in html
    assert "Task is not startable from the dashboard" in html
    assert "payload.task_state_policy" in html
    assert "startableTaskStates.has(taskState)" in html
    assert "taskOpenContractReadyKnown" in html
    assert "const operatorValidationStartGate = payload.operator_validation_start_gate || {}" in html
    assert "const operatorValidationBlocksStart = operatorValidationStartGate.blocks_task_open === true" in html
    assert "const taskOpenOperatorAction = operatorValidationBlocksStart" in html
    assert "canOpenTaskFromDashboard = !operatorValidationBlocksStart && startableTaskStates.has(taskState)" in html
    assert "Start is waiting for operator validation" in html
    assert "Task-open authorization contract:" in html
    assert "Task-open not-ready reason:" in html
    assert "Task-open operator action:" in html
    assert (
        "body: JSON.stringify({client_label: navigator.userAgent, expected_user: expectedUserGuardParam || \"\"})"
        in html
    )
    assert "page_context=work_dashboard_dataset_queue" in html
    assert "operator_validation_start_gate_required" in html
    assert "operator_validation_start_gate_ready" in html
    assert "operator_validation_start_gate_blocks_task_open" in html
    assert "operator_validation_start_gate_not_ready_reason" in html
    assert "operator_validation_mutation_gate_required" in html
    assert "operator_validation_mutation_gate_ready" in html
    assert "operator_validation_mutation_gate_blocks_browser_mutation" in html
    assert "operator_validation_mutation_gate_not_ready_reason" in html
    assert "operator_validation_mutation_gate_pending_gate_ids" in html
    assert "operator_validation_mutation_gate_required_missing_evidence_gate_ids" in html
    assert "runtime_operator_validation_gate_cli_policy_preferred_require_flag" in html
    assert "runtime_operator_validation_gate_cli_policy_protects_browser_mutations" in html
    assert "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write" in html
    assert "operator_validation_source" in html
    assert "operator_validation_gate_count" in html
    assert "operator_validation_pending_gate_ids" in html
    assert "operator_validation_required_missing_evidence_gate_ids" in html
    assert "safe_share_next_action_summary" not in html
    assert "operator_validation_public_fields" in html
    assert "operator_validation_operator_only_fields" in html
    assert "operator_validation_labeler_visible_payloads_include_operator_only_fields" in html
    assert "operator_validation_per_user_payloads_use_public_fields_only" in html
    assert "operator_validation_top_level_operator_reports_may_include_operator_only_fields" in html
    assert "operator_validation_required_pending_gate_count" in html
    assert "operator_validation_command_template_schema" in html
    assert "operator_validation_command_template_command_count" in html
    assert "operator_validation_command_template_command_ids" in html
    assert "operator_validation_command_template_missing_command_gate_ids" in html
    assert "operator_validation_command_template_launch_evidence_collection_plan_schema" in html
    assert "operator_validation_command_template_launch_evidence_collection_step_count" in html
    assert "operator_validation_command_template_launch_evidence_collection_required_final_field" in html
    assert "safe_share_external_launch_evidence_gap_gate_ids" in html
    assert "safe_share_external_launch_evidence_gap_summary" in html
    assert "safe_share_external_launch_evidence_gap_todos" in html
    assert "safe_share_external_launch_evidence_gap_todo_fields" in html
    assert "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id" in html
    assert "operator_validation_command_template_commands_are_operator_only" in html
    assert "operator_validation_command_template_commands_are_labeler_instructions" in html
    assert "operator_validation_command_template_labelers_must_not_run_commands" in html
    assert "operator_validation_needs_review_gate_count" in html
    assert "record-browser-smoke-evidence --evidence" not in html
    assert "apply-operator-evidence-templates --path" not in html
    assert "record-zarr-backup-evidence --evidence" not in html
    assert "operator_validation_required_missing_evidence_gate_count" in html
    assert "preferred_labeler_entrypoint" in html
    assert "preferred_labeler_entry_url" in html
    assert "preferred_labeler_entry_url_matches_dataset_queue" in html
    assert "preferred_labeler_entry_url_matches_personal_dataset_queue" in html
    assert "personalized_labeler_entry_url_matches_personal_dataset_queue" in html
    assert "personal_dataset_queue_link_role" in html
    assert "dataset_queue_link_role" in html
    assert "canonical_dataset_queue_link_role" in html
    assert "dataset_queue_preview_url" in html
    assert "canonical_dataset_queue_preview_url" in html
    assert "dashboard_link_role" in html
    assert "identity_probe_link_role" in html
    assert "task_links_role" in html
    assert "expected_user_dataset_queue_url" in html
    assert "expected_user_dashboard_url" in html
    assert "expected_user_personal_dataset_queue_url" in html
    assert "expected_user_personal_work_url" in html
    assert "personalized_labeler_entry_url" in html
    assert "Dataset support details" in html


def test_identity_probe_points_matched_users_to_dataset_queue_first():
    payload = _identity_probe_payload(
        user="alice",
        auth_source="header",
        expected_user="alice",
    )
    identity = payload["identity"]

    assert payload["ok"] is True
    assert identity["matches_expected_user"] is True
    assert identity["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
    assert identity["expected_user_dashboard_url"] == "/work?expected_user=alice"
    assert identity["personal_dataset_queue_page_path"] == "/my-datasets"
    assert identity["personal_work_page_path"] == "/my-work"
    assert identity["personal_dataset_queue_alias_for"] == "/datasets"
    assert identity["personal_work_alias_for"] == "/work"
    assert identity["identity_probe_diagnostic_only"] is True
    assert identity["identity_probe_does_not_authorize_work"] is True
    assert identity["identity_probe_unknown_user_blocks_work_surfaces"] is True
    assert payload["identity_probe_diagnostic_only"] is True
    assert payload["identity_probe_does_not_authorize_work"] is True
    assert payload["identity_probe_unknown_user_blocks_work_surfaces"] is True
    assert payload["identity_probe_expected_user_guard_required"] is True
    assert payload["identity_probe_launch_ctas_rendered"] is True
    assert payload["identity_probe_launch_ctas_suppressed"] is False
    assert payload["identity_probe_failed_support_urls_diagnostic_only"] is False
    assert identity["identity_probe_launch_ctas_rendered"] is True
    assert identity["identity_probe_launch_ctas_suppressed"] is False
    assert identity["identity_probe_failed_support_urls_diagnostic_only"] is False
    assert identity["identity_probe_expected_user_guard_required"] is True
    assert identity["expected_user_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=alice"
    )
    assert identity["expected_user_personal_work_url"] == "/my-work?expected_user=alice"
    assert identity["preferred_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert identity["preferred_labeler_entry_url"] == "/my-datasets?expected_user=alice"
    assert identity["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert identity["personalized_labeler_entry_url"] == "/my-datasets?expected_user=alice"
    assert identity["preferred_labeler_entry_url_matches_dataset_queue"] is True
    assert identity["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert (
        identity["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        is True
    )
    assert identity["personal_dataset_queue_link_role"] == "preferred_queue"
    assert identity["dataset_queue_link_role"] == "canonical_queue_fallback"
    assert identity["canonical_dataset_queue_link_role"] == "canonical_queue_fallback"
    assert "open the guarded dataset queue first" in identity["operator_action"]

    html = _identity_probe_html(payload).decode("utf-8")

    assert "Open your personalized dataset queue" in html
    assert "Open your datasets-waiting landing page" in html
    assert "Open your full personalized work dashboard" in html
    assert "Open canonical dataset queue fallback" in html
    assert "Open canonical work dashboard fallback" in html
    assert "Preferred entry" in html
    assert "personal_datasets_waiting_queue" in html
    assert "Preferred entry URL" in html
    assert "/my-datasets?expected_user=alice" in html
    assert "Personal queue role" in html
    assert "preferred_queue" in html
    assert "Canonical queue role" in html
    assert "canonical_queue_fallback" in html
    assert "Preferred matches personal queue" in html
    assert "Personalized matches personal queue" in html
    assert "Identity diagnostic only" in html
    assert "Identity authorizes work" in html
    assert "Unknown identity blocks work" in html
    assert "identity_probe_diagnostic_only" in html
    assert "identity_probe_does_not_authorize_work" in html
    assert "identity_probe_unknown_user_blocks_work_surfaces" in html
    assert "Open your datasets-waiting landing page" in html
    assert "Open your full personalized work dashboard" in html
    assert "Open canonical dataset queue fallback" in html
    assert "Open canonical work dashboard fallback" in html
    assert "/my-datasets?expected_user=alice" in html
    assert "/my-work?expected_user=alice" in html
    assert html.index("Open your personalized dataset queue") < html.index(
        "Open your full personalized work dashboard"
    )

    missing_expected_payload = _identity_probe_payload(
        user="alice",
        auth_source="header",
    )
    missing_expected_identity = missing_expected_payload["identity"]
    missing_expected_html = _identity_probe_html(missing_expected_payload).decode(
        "utf-8"
    )

    assert missing_expected_payload["ok"] is False
    assert missing_expected_payload["error"] == "identity_expected_user_required"
    assert missing_expected_identity["matches_expected_user"] is False
    assert missing_expected_payload["identity_probe_expected_user_guard_required"] is True
    assert missing_expected_identity["identity_probe_expected_user_guard_required"] is True
    assert missing_expected_identity["expected_user_guard_present"] is False
    assert missing_expected_payload["identity_probe_launch_ctas_rendered"] is False
    assert missing_expected_payload["identity_probe_launch_ctas_suppressed"] is True
    assert (
        missing_expected_payload[
            "identity_probe_failed_support_urls_diagnostic_only"
        ]
        is True
    )
    assert missing_expected_identity["identity_probe_launch_ctas_rendered"] is False
    assert missing_expected_identity["identity_probe_launch_ctas_suppressed"] is True
    assert (
        missing_expected_identity[
            "identity_probe_failed_support_urls_diagnostic_only"
        ]
        is True
    )
    assert "Expected user required: stop before labeling" in missing_expected_html
    assert "Do not open labeling work from this browser identity" in missing_expected_html
    assert "Open your personalized dataset queue" not in missing_expected_html
    assert "Open your datasets-waiting landing page" not in missing_expected_html
    assert "Open your full personalized work dashboard" not in missing_expected_html
    assert "Open canonical dataset queue fallback" not in missing_expected_html
    assert "Open canonical work dashboard fallback" not in missing_expected_html


def test_browser_mutation_response_metadata_reports_server_owned_zarr_and_audit_event():
    payload = _browser_mutation_response_metadata(
        workflow_kind="keypoints",
        session={
            "session_id": "session-a",
            "task_id": "task-a",
            "recording_id": "rec-a",
            "user": "alice",
        },
        mutation_event={
            "event_id": "event-1",
            "task_id": "task-a",
            "recording_id": "rec-a",
            "event_type": "save_keypoints",
            "created_at_utc": "2026-06-23T12:00:00+00:00",
        },
    )

    assert payload["schema"] == "palette.web_labeling_browser_mutation_response.v1"
    assert payload["workflow_kind"] == "keypoints"
    assert payload["assignment_authorization_checked_server_side"] is True
    assert payload["assignment_authorization_result"] == "passed"
    assert payload["active_assignment_checked_server_side"] is True
    assert payload["active_assignment_required"] is True
    assert payload["active_assignment_present"] is True
    assert payload["task_assigned_to_resolved_user_checked_server_side"] is True
    assert payload["task_assigned_to_resolved_user"] is True
    assert payload["task_state_checked_server_side"] is True
    assert payload["session_checked_server_side"] is True
    assert payload["session_ownership_checked_server_side"] is True
    assert payload["session_user_matches_resolved_user"] is True
    assert payload["current_session_checked_server_side"] is True
    assert payload["current_session_required"] is True
    assert payload["reassignment_session_safety_checked_server_side"] is True
    assert payload["reassignment_session_safety_passed"] is True
    assert payload["current_target_token_checked_server_side"] is True
    assert payload["target_token_required_for_mutation"] is True
    assert payload["mutation_authorization_contract"]["schema"] == (
        "palette.web_labeling_mutation_authorization_contract.v1"
    )
    assert payload["mutation_authorization_contract"]["ready"] is True
    assert payload["mutation_authorization_contract"]["session_lookup_result"] == "passed"
    assert payload["mutation_authorization_contract"]["session_owned_by_resolved_user"] is True
    assert payload["mutation_authorization_contract"]["task_reloaded_server_side"] is True
    assert payload["mutation_authorization_contract"]["task_assigned_to_resolved_user"] is True
    assert payload["mutation_authorization_contract"]["assignment_status_active"] is True
    assert payload["mutation_authorization_contract"]["task_open_for_mutation"] is True
    assert payload["mutation_authorization_contract"]["current_session_result"] == "passed"
    assert payload["mutation_authorization_contract"][
        "reassignment_session_safety_result"
    ] == "passed"
    assert payload["mutation_authorization_contract"]["current_target_token_result"] == "passed"
    assert payload["mutation_authorization_contract"][
        "browser_supplied_zarr_or_csv_target_allowed"
    ] is False
    assert payload["mutation_authorization_contract"][
        "browser_supplied_target_selectors_allowed"
    ] is False
    assert payload["mutation_authorization_contract"]["client_authorizes_mutation"] is False
    assert payload["mutation_authorization_contract"][
        "operator_validation_mutation_gate_checked_server_side"
    ] is True
    assert payload["mutation_authorization_contract"][
        "operator_validation_mutation_gate_required"
    ] is False
    assert payload["mutation_authorization_contract"][
        "operator_validation_mutation_gate_ready"
    ] is True
    assert payload["mutation_authorization_contract"][
        "operator_validation_mutation_gate_blocks_browser_mutation"
    ] is False
    assert payload["mutation_authorization_contract"]["server_authorizes_mutation"] is True
    assert payload["operator_validation_mutation_gate_checked_server_side"] is True
    assert payload["operator_validation_mutation_gate_required"] is False
    assert payload["operator_validation_mutation_gate_ready"] is True
    assert payload["operator_validation_mutation_gate_blocks_browser_mutation"] is False
    assert payload["operator_validation_mutation_gate"]["schema"] == (
        "palette.web_labeling_runtime_operator_validation_mutation_gate.v1"
    )
    assert payload["operator_validation_mutation_gate"]["blocks_browser_mutation"] is False
    assert "safe_share_next_action_summary" not in payload
    assert "safe_share_next_action_summary" not in payload["mutation_authorization_contract"]
    assert payload["authorization_context"]["schema"] == (
        "palette.web_labeling_authorization_context.v1"
    )
    assert payload["authorization_context"]["resolved_user"] == "alice"
    assert payload["authorization_context"]["session_id"] == "session-a"
    assert payload["authorization_context"]["session_task_id"] == "task-a"
    assert payload["authorization_context"]["session_recording_id"] == "rec-a"
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
    assert payload["authorization_context"][
        "return_personal_work_expected_user_guarded"
    ] is True
    assert payload["return_expected_user"] == "alice"
    assert payload["return_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=alice"
    )
    assert payload["return_personal_dataset_queue_expected_user_guarded"] is True
    assert payload["return_personal_work_url"] == "/my-work?expected_user=alice"
    assert payload["return_personal_work_expected_user_guarded"] is True
    assert payload["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert payload["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert payload["browser_mutation_write_checklist"]["ready"] is True
    assert payload["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert payload["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_checklist"][
        "handoff_csv_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_checklist"][
        "intermediate_csv_artifacts_are_label_write_targets"
    ] is False
    assert payload["browser_mutation_write_checklist"]["browser_writes_handoff_csv"] is False
    assert payload["browser_mutation_write_checklist"]["browser_writes_intermediate_csv"] is False
    assert payload["browser_mutation_write_checklist"]["browser_has_direct_zarr_write_authority"] is False
    assert payload["label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert payload["browser_label_write_target"] == "training_zarr"
    assert payload["server_mutates_task_scoped_zarr_targets"] is True
    assert payload["training_zarr_mutations_are_server_owned"] is True
    assert payload["promotion_training_zarr_requires_task_scope"] is True
    assert payload["handoff_artifacts_are_metadata_only"] is True
    assert payload["csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert payload["csv_handoff_artifacts_are_label_write_targets"] is False
    assert payload["handoff_csv_artifacts_are_label_write_targets"] is False
    assert payload["intermediate_csv_artifacts_are_label_write_targets"] is False
    assert payload["browser_writes_csv_or_handoff_files"] is False
    assert payload["browser_writes_handoff_csv"] is False
    assert payload["browser_writes_intermediate_csv"] is False
    assert payload["browser_receives_zarr_write_authority"] is False
    assert payload["browser_has_direct_zarr_write_authority"] is False
    assert payload["audit_event_store"] == "labeling_task_events"
    assert payload["audit_event_id"] == "event-1"
    assert payload["audit_event_type"] == "save_keypoints"
    assert payload["audit_events"] == [
        {
            "event_id": "event-1",
            "event_type": "save_keypoints",
            "task_id": "task-a",
            "recording_id": "rec-a",
            "created_at_utc": "2026-06-23T12:00:00+00:00",
        }
    ]


def test_browser_mutation_failure_metadata_reports_no_csv_or_direct_zarr_writes():
    payload = _browser_mutation_failure_metadata(
        session={
            "session_id": "session-a",
            "task_id": "task-a",
            "recording_id": "rec-a",
            "user": "alice",
        },
        error="operator_validation_mutation_blocked",
    )

    contract = payload["mutation_authorization_contract"]

    assert contract["ready"] is False
    assert contract["current_target_token_result"] == "not_checked"
    assert contract["server_authorizes_mutation"] is False
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
    assert payload["authorization_context"][
        "return_personal_work_expected_user_guarded"
    ] is True
    assert payload["return_expected_user"] == "alice"
    assert payload["return_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=alice"
    )
    assert payload["return_personal_dataset_queue_expected_user_guarded"] is True
    assert payload["return_personal_work_url"] == "/my-work?expected_user=alice"
    assert payload["return_personal_work_expected_user_guarded"] is True
    assert payload["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert payload["label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert payload["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert payload["browser_label_write_target"] == "training_zarr"
    assert payload["server_mutates_task_scoped_zarr_targets"] is True
    assert payload["training_zarr_mutations_are_server_owned"] is True
    assert payload["promotion_training_zarr_requires_task_scope"] is True
    assert payload["handoff_artifacts_are_metadata_only"] is True
    assert payload["csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert payload["csv_handoff_artifacts_are_label_write_targets"] is False
    assert payload["browser_writes_csv_or_handoff_files"] is False
    assert payload["browser_writes_handoff_csv"] is False
    assert payload["browser_writes_intermediate_csv"] is False
    assert payload["browser_receives_zarr_write_authority"] is False
    assert payload["browser_has_direct_zarr_write_authority"] is False
    assert "operator_validation_mutation_gate" in payload
    assert "safe_share_next_action_summary" not in payload
    assert "safe_share_next_action_summary" not in contract


def test_labeler_promotion_retry_response_is_operator_support_only():
    payload = _labeler_promotion_retry_operator_support_payload(
        user="alice",
        expected_user="alice",
        event={
            "event_id": "event-1",
            "task_id": "task-a",
            "recording_id": "rec-a",
        },
        event_task={
            "task_id": "task-a",
            "recording_id": "rec-a",
            "assignee_user": "alice",
            "assignment_status": "active",
            "state": "in_progress",
            "workflow_kind": "detect_analysis",
        },
        session={
            "session_id": "session-a",
            "task_id": "task-a",
            "recording_id": "rec-a",
            "user": "alice",
        },
    )

    assert payload["ok"] is False
    assert payload["error"] == "operator_support_required"
    assert payload["status"] == 403
    assert payload["failed_event_id"] == "event-1"
    assert payload["operator_recovery_route"] == "/api/admin/events/event-1/retry-promotion"
    assert payload["operator_recovery_route_template"] == "/api/admin/events/{event_id}/retry-promotion"
    assert payload["labeler_failed_promotion_retry_action"] == "operator_support_only"
    assert payload["labeler_promotion_retry_mutation_enabled"] is False
    assert payload["promotion_retry_attempted"] is False
    assert payload["promotion_retry_claimed"] is False
    assert payload["browser_mutation_write_checklist"]["ready"] is True
    assert payload["browser_label_write_target"] == "training_zarr"
    assert payload["browser_writes_csv_or_handoff_files"] is False
    assert payload["browser_writes_handoff_csv"] is False
    assert payload["browser_writes_intermediate_csv"] is False
    assert payload["browser_receives_zarr_write_authority"] is False
    assert payload["browser_has_direct_zarr_write_authority"] is False
    assert payload["operator_validation_mutation_gate_checked_server_side"] is True
    assert payload["operator_validation_mutation_gate_required"] is False
    assert payload["operator_validation_mutation_gate_ready"] is True
    assert payload["operator_validation_mutation_gate_blocks_browser_mutation"] is False
    assert payload["operator_validation_mutation_gate_not_ready_reason"] == ""
    assert "safe_share_next_action_summary" not in payload
    assert payload["authorization_context"]["resolved_user"] == "alice"
    assert payload["authorization_context"]["session_id"] == "session-a"
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
    assert payload["authorization_context"][
        "return_personal_work_expected_user_guarded"
    ] is True
    assert payload["return_expected_user"] == "alice"
    assert payload["return_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=alice"
    )
    assert payload["return_personal_dataset_queue_expected_user_guarded"] is True
    assert payload["return_personal_work_url"] == "/my-work?expected_user=alice"
    assert payload["return_personal_work_expected_user_guarded"] is True


def test_admin_promotion_retry_preflight_rejects_non_failed_events_before_claim():
    preflight = _admin_promotion_retry_preflight_error(
        {
            "event_id": "event-1",
            "event_type": "save_keypoints",
            "task_id": "task-a",
            "recording_id": "rec-a",
        }
    )

    assert preflight is not None
    payload, status = preflight
    assert status.value == 409
    assert payload["ok"] is False
    assert payload["error"] == "promotion_retry_not_supported"
    assert payload["event_id"] == "event-1"
    assert payload["event_type"] == "save_keypoints"
    assert payload["required_event_type"] == "promotion_failed"
    assert payload["promotion_retry_attempted"] is False
    assert payload["promotion_retry_claimed"] is False

    assert _admin_promotion_retry_preflight_error({"event_type": "promotion_failed"}) is None


def test_browser_mutation_status_js_surfaces_audit_event_id_and_target():
    assert "mutationStatusSuffix" in _BROWSER_MUTATION_STATUS_JS
    assert "latestMutationSupportReference" in _BROWSER_MUTATION_STATUS_JS
    assert "copyMutationSupportReference" in _BROWSER_MUTATION_STATUS_JS
    assert "copy-mutation-support-reference" in _BROWSER_MUTATION_STATUS_JS
    assert "audit_event_id" in _BROWSER_MUTATION_STATUS_JS
    assert "audit_event_type" in _BROWSER_MUTATION_STATUS_JS
    assert "task_id=" in _BROWSER_MUTATION_STATUS_JS
    assert "recording_id=" in _BROWSER_MUTATION_STATUS_JS
    assert "server_target=" in _BROWSER_MUTATION_STATUS_JS
    assert "data_plane_write_target" in _BROWSER_MUTATION_STATUS_JS
    assert "operator support" in _BROWSER_MUTATION_STATUS_JS
    assert "give audit event " in _BROWSER_MUTATION_STATUS_JS
    assert "server target " in _BROWSER_MUTATION_STATUS_JS
    assert "server-owned assigned task Zarr scope" in _BROWSER_MUTATION_STATUS_JS


def test_keypoint_editor_exposes_copy_mutation_support_reference_button():
    html = _keypoint_session_html(
        {
            "session_id": "session-a",
            "task_id": "task-a",
            "title": "Keypoint task",
            "expires_at_utc": "2026-06-23T12:30:00+00:00",
        }
    ).decode("utf-8")

    assert 'id="copy-mutation-support-reference"' in html
    assert "copyMutationSupportReference(this)" in html
    assert "Copy support reference" in html
    assert "completeTask()" in html
    assert "/api/sessions/${encodeURIComponent(sessionId)}/complete" in html
    assert "/api/tasks/${encodeURIComponent" not in html


def test_admin_html_exposes_audit_event_lookup_ui():
    html = _admin_html().decode("utf-8")

    assert "Audit event lookup" in html
    assert 'id="audit-event-lookup-form"' in html
    assert 'id="audit-event-id"' in html
    assert 'id="audit-event-lookup-result"' in html
    assert "lookupAuditEvent" in html
    assert "/api/admin/events/" in html
    assert "audit_event_lookup_route" in html
    assert "Dataset queue direct start" in html
    assert "browser writes CSV/handoff=" in html
    assert "browser receives zarr write=" in html
    assert "browser direct zarr write=" in html
    assert "Browser saves are applied server-side to your assigned task/training Zarr scope" in html
    assert "CSV, HTML, JSON, and handoff files are metadata only" in html
    assert "Each recording has one active assigned owner" in html
    assert "Labelers should not run operator evidence, repair, checksum, or validation commands" in html
    assert "dataset_queue_direct_start_policy" in html
    assert "endpoint_task_segment_must_match_row_task_id" in html
    assert "preferred_matches_personal_queue" in html
    assert "personalized_matches_personal_queue" in html
    assert "mutable data plane=" in html
    assert "training Zarr server-owned=" in html
    assert "startable task state=" in html
    assert "startable states=" in html
    assert "forwarded links recheck identity=" in html
    assert "protected routes=" in html
    assert "personalized aliases=" in html
    assert "alias headers match canonical=" in html


def test_inspect_handoff_operator_evidence_commands_reports_directory_and_zip(tmp_path):
    package_dir = tmp_path / "launch-bundle"
    package_dir.mkdir()
    command_sheet = package_dir / "operator-evidence-commands.txt"
    command_sheet.write_text(
        "\n".join(
            [
                "Palette web-labeling operator evidence command sheet",
                "",
                "Boundary: operator-only. These commands require operator authorization and are not labeler instructions.",
                "Do not send this command sheet, operator evidence templates, backup manifests, or runnable operator commands to labelers.",
                "Labelers should use only their guarded browser links and should not run Palette/Crimson commands or edit Zarrs directly.",
                "",
                "inspect-handoff --path launch-bundle",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    directory_report = _inspect_handoff_operator_evidence_commands(package_dir)
    assert directory_report["required"] is True
    assert directory_report["present"] is True
    assert directory_report["valid"] is True
    assert directory_report["matched_paths"] == [str(command_sheet)]
    assert directory_report["operator_only_boundary_present"] is True
    assert directory_report["operator_only_boundary_missing_phrases"] == []
    directory_summary = _operator_evidence_commands_public_summary(directory_report)
    assert directory_summary["schema"] == (
        "palette.web_labeling_operator_evidence_commands_summary.v1"
    )
    assert directory_summary["present"] is True
    assert directory_summary["valid"] is True
    assert directory_summary["operator_only_boundary_present"] is True
    assert directory_summary["blocking_reason_id"] == ""

    stale_package_dir = tmp_path / "stale-launch-bundle"
    stale_package_dir.mkdir()
    stale_command_sheet = stale_package_dir / "operator-evidence-commands.txt"
    stale_command_sheet.write_text("inspect-handoff --path launch-bundle\n", encoding="utf-8")
    stale_report = _inspect_handoff_operator_evidence_commands(stale_package_dir)
    assert stale_report["present"] is True
    assert stale_report["valid"] is False
    assert stale_report["operator_only_boundary_present"] is False
    assert "Boundary: operator-only" in stale_report["operator_only_boundary_missing_phrases"]
    stale_summary = _operator_evidence_commands_public_summary(stale_report)
    assert stale_summary["present"] is True
    assert stale_summary["valid"] is False
    assert stale_summary["operator_only_boundary_present"] is False
    assert stale_summary["blocking_reason_id"] == "operator_evidence_commands_boundary_missing"
    assert "Boundary: operator-only" in stale_summary[
        "operator_only_boundary_missing_phrases"
    ]
    failure_actions = _inspection_failure_actions(
        ["operator_evidence_commands_boundary_missing"],
        validation_checklist={},
    )
    assert "operator-only boundary warning" in "\n".join(failure_actions)

    missing_report = _inspect_handoff_operator_evidence_commands(tmp_path / "missing-launch-bundle")
    assert missing_report["present"] is False
    assert missing_report["valid"] is False
    assert missing_report["operator_only_boundary_present"] is False
    missing_summary = _operator_evidence_commands_public_summary(missing_report)
    assert missing_summary["blocking_reason_id"] == "operator_evidence_commands_missing"
    optional_missing_summary = _operator_evidence_commands_public_summary(
        {**missing_report, "required": False}
    )
    assert optional_missing_summary["blocking_reason_id"] == ""

    zip_path = tmp_path / "launch-bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(command_sheet, "launch-bundle/operator-evidence-commands.txt")

    zip_report = _inspect_handoff_operator_evidence_commands(zip_path)
    assert zip_report["present"] is True
    assert zip_report["valid"] is True
    assert zip_report["matched_paths"] == ["launch-bundle/operator-evidence-commands.txt"]
    assert zip_report["operator_only_boundary_present"] is True


def test_inspect_handoff_launch_evidence_execution_checklist_reports_directory_and_zip(tmp_path):
    package_dir = tmp_path / "launch-bundle"
    package_dir.mkdir()
    checklist = package_dir / "launch-evidence-execution-checklist.txt"
    checklist.write_text(
        "\n".join(
            [
                "Palette web-labeling launch evidence execution checklist",
                "",
                "Operator-only checklist. Do not send this file or these commands to labelers.",
                "record-zarr-backup-evidence",
                "record-browser-response-security-evidence",
                "record-identity-source-evidence",
                "record-browser-smoke-evidence",
                "record-disposable-zarr-mutation-smoke-evidence",
                "apply-operator-evidence-templates",
                "inspect-handoff --path PACKAGE_PATH --require-shareable",
                "labeler_links_safe_to_share=true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    directory_report = _inspect_handoff_launch_evidence_execution_checklist(package_dir)
    assert directory_report["required"] is True
    assert directory_report["present"] is True
    assert directory_report["valid"] is True
    assert directory_report["matched_paths"] == [str(checklist)]
    assert directory_report["checklist_contract_present"] is True
    assert directory_report["checklist_missing_phrases"] == []
    directory_summary = _launch_evidence_execution_checklist_public_summary(
        directory_report
    )
    assert directory_summary["schema"] == (
        "palette.web_labeling_launch_evidence_execution_checklist_summary.v1"
    )
    assert directory_summary["present"] is True
    assert directory_summary["valid"] is True
    assert directory_summary["checklist_contract_present"] is True
    assert directory_summary["blocking_reason_id"] == ""

    stale_package_dir = tmp_path / "stale-launch-bundle"
    stale_package_dir.mkdir()
    stale_checklist = stale_package_dir / "launch-evidence-execution-checklist.txt"
    stale_checklist.write_text("inspect-handoff --path PACKAGE_PATH --require-shareable\n", encoding="utf-8")
    stale_report = _inspect_handoff_launch_evidence_execution_checklist(stale_package_dir)
    assert stale_report["present"] is True
    assert stale_report["valid"] is False
    assert stale_report["checklist_contract_present"] is False
    assert "Operator-only checklist" in stale_report["checklist_missing_phrases"]
    stale_summary = _launch_evidence_execution_checklist_public_summary(stale_report)
    assert stale_summary["present"] is True
    assert stale_summary["valid"] is False
    assert stale_summary["checklist_contract_present"] is False
    assert (
        stale_summary["blocking_reason_id"]
        == "launch_evidence_execution_checklist_incomplete"
    )
    assert "Operator-only checklist" in stale_summary["checklist_missing_phrases"]
    failure_actions = _inspection_failure_actions(
        ["launch_evidence_execution_checklist_incomplete"],
        validation_checklist={},
    )
    assert "complete operator-only launch evidence runbook" in "\n".join(failure_actions)
    repair_commands = {
        row["id"]: row
        for row in _inspection_operator_repair_commands(
            failure_actions,
            launch_evidence_execution_checklist_summary=stale_summary,
        )
    }
    assert "regenerate_package_with_launch_evidence_execution_checklist" in repair_commands
    repair_command = repair_commands[
        "regenerate_package_with_launch_evidence_execution_checklist"
    ]
    assert repair_command["category"] == "handoff_regeneration"
    assert repair_command["repair_mode"] == "regenerate_package"
    assert repair_command["artifact_contract"] == "launch_evidence_execution_checklist"
    assert repair_command["safe_share_blocker"] == (
        "launch_evidence_execution_checklist_incomplete"
    )
    assert repair_command["safe_share_blockers"] == [
        "launch_evidence_execution_checklist_missing",
        "launch_evidence_execution_checklist_incomplete",
        "launch_evidence_execution_checklist_invalid",
    ]
    assert repair_command["required_file"] == "launch-evidence-execution-checklist.txt"
    assert repair_command["required_phrase_contract"] == (
        "shareability_launch_evidence_execution_checklist_required_phrases"
    )
    assert "record-browser-smoke-evidence" in repair_command["required_phrases"]
    assert repair_command["missing_phrase_count"] == len(
        stale_summary["checklist_missing_phrases"]
    )
    assert "Operator-only checklist" in repair_command["missing_phrases"]
    assert repair_command["requires_checksum_refresh_after_run"] is False

    missing_report = _inspect_handoff_launch_evidence_execution_checklist(
        tmp_path / "missing-launch-bundle"
    )
    assert missing_report["present"] is False
    assert missing_report["valid"] is False
    assert missing_report["checklist_contract_present"] is False
    missing_summary = _launch_evidence_execution_checklist_public_summary(
        missing_report
    )
    assert (
        missing_summary["blocking_reason_id"]
        == "launch_evidence_execution_checklist_missing"
    )
    optional_missing_summary = _launch_evidence_execution_checklist_public_summary(
        {**missing_report, "required": False}
    )
    assert optional_missing_summary["blocking_reason_id"] == ""

    zip_path = tmp_path / "launch-bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(checklist, "launch-bundle/launch-evidence-execution-checklist.txt")

    zip_report = _inspect_handoff_launch_evidence_execution_checklist(zip_path)
    assert zip_report["present"] is True
    assert zip_report["valid"] is True
    assert zip_report["matched_paths"] == [
        "launch-bundle/launch-evidence-execution-checklist.txt"
    ]
    assert zip_report["checklist_contract_present"] is True


def test_handoff_sendability_blocks_missing_or_unsafe_browser_mutation_write_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["browser_mutation_write_policy_missing"]
    assert "browser mutation write policy metadata" in _handoff_sendability_actions(
        ["browser_mutation_write_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "browser_mutation_write_policy": {
            **_browser_mutation_write_policy(),
            "browser_writes_csv_or_handoff_files": True,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["browser_mutation_write_policy_not_ready"]
    unsafe_action = _handoff_sendability_actions(
        ["browser_mutation_write_policy_not_ready"]
    )[0]
    assert "task-scoped training Zarrs are the mutable label data plane" in unsafe_action
    assert "label mutation target kind" in unsafe_action
    assert "metadata-only control-plane non-label-write targets" in unsafe_action

    missing_schema_evidence_handoff = {
        **ready_handoff,
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
    }
    assert _handoff_ready_to_send(missing_schema_evidence_handoff) is False
    assert _handoff_sendability_reasons(missing_schema_evidence_handoff) == [
        "assignment_ownership_contract_not_ready"
    ]

    unsafe_ownership_contract_handoff = {
        **ready_handoff,
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "single_owner_policy": {
            **_assignment_ownership_policy(),
            "recording_id_primary_key": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_ownership_contract_handoff) is False
    assert _handoff_sendability_reasons(unsafe_ownership_contract_handoff) == [
        "assignment_ownership_contract_not_ready"
    ]
    assert "recording_id primary key" in _handoff_sendability_actions(
        ["assignment_ownership_contract_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_labeler_route_authorization_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["labeler_route_authorization_policy_missing"]
    assert "labeler-route authorization policy metadata" in _handoff_sendability_actions(
        ["labeler_route_authorization_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "labeler_route_authorization_policy": {
            **_labeler_route_authorization_policy(),
            "expected_user_must_match_resolved_user": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["labeler_route_authorization_policy_not_ready"]
    route_authorization_action = _handoff_sendability_actions(
        ["labeler_route_authorization_policy_not_ready"]
    )[0]
    assert "expected user" in route_authorization_action
    assert "single-owner store proof" in route_authorization_action
    assert "zero duplicate active owners" in route_authorization_action
    assert "training-Zarr targets" in route_authorization_action
    assert "no intermediate CSV mutation" in route_authorization_action
    assert "single_owner_store_proof_ready=true" in route_authorization_action
    assert "assignment_ownership_integrity_ok=true" in route_authorization_action
    assert "browser_mutation_target_resolved_server_side=true" in route_authorization_action
    assert "labelers_mutate_assigned_training_zarrs=true" in route_authorization_action
    assert "labelers_mutate_intermediate_csvs=false" in route_authorization_action

    missing_runtime_checklist_handoff = {
        **ready_handoff,
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "labeler_route_authorization_checklist": None,
    }
    assert _handoff_ready_to_send(missing_runtime_checklist_handoff) is False
    assert _handoff_sendability_reasons(missing_runtime_checklist_handoff) == [
        "labeler_route_authorization_policy_not_ready"
    ]

    safe_handoff = {
        **ready_handoff,
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "labeler_route_authorization_checklist": {
            "ready": True,
            "single_owner_store_proof_ready": True,
            "assignment_ownership_integrity_ok": True,
            "duplicate_active_owner_count": 0,
            "browser_mutation_target_resolved_server_side": True,
            "labelers_mutate_assigned_training_zarrs": True,
            "labelers_mutate_intermediate_csvs": False,
        },
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []

    unsafe_data_plane_handoff = {
        **safe_handoff,
        "labeler_route_authorization_checklist": {
            **safe_handoff["labeler_route_authorization_checklist"],
            "labelers_mutate_assigned_training_zarrs": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_data_plane_handoff) is False
    assert _handoff_sendability_reasons(unsafe_data_plane_handoff) == [
        "labeler_route_authorization_policy_not_ready"
    ]


def test_handoff_sendability_blocks_missing_or_unsafe_signed_link_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["signed_link_policy_missing"]
    assert "signed-link policy metadata" in _handoff_sendability_actions(
        ["signed_link_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "signed_link_policy": {
            **_browser_signed_link_policy(),
            "authorization_grant": True,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == [
        "labeler_safety_policy_not_ready",
        "signed_link_policy_not_ready",
    ]
    assert "not authorization grants" in _handoff_sendability_actions(
        ["signed_link_policy_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "signed_link_policy": _browser_signed_link_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_labeler_safety_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["labeler_safety_policy_missing"]
    assert "labeler safety metadata" in _handoff_sendability_actions(
        ["labeler_safety_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "labeler_safety": {
            **_labeler_safety_policy(),
            "browser_receives_raw_zarr_paths": True,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["labeler_safety_policy_not_ready"]
    assert "raw Zarr paths" in _handoff_sendability_actions(
        ["labeler_safety_policy_not_ready"]
    )[0]

    unsafe_install_handoff = {
        **ready_handoff,
        "labeler_safety": {
            **_labeler_safety_policy(),
            "requires_local_palette_installation": True,
        },
    }
    assert _handoff_ready_to_send(unsafe_install_handoff) is False
    assert _handoff_sendability_reasons(unsafe_install_handoff) == ["labeler_safety_policy_not_ready"]
    assert "no local Palette" in _handoff_sendability_actions(
        ["labeler_safety_policy_not_ready"]
    )[0]

    for unsafe_identity_policy_field in (
        "identity_probe_expected_user_guard_required",
        "identity_probe_success_launch_ctas_rendered",
        "identity_probe_failed_launch_ctas_suppressed",
        "identity_probe_failed_support_urls_diagnostic_only",
    ):
        unsafe_identity_policy_handoff = {
            **ready_handoff,
            "labeler_safety": {
                **_labeler_safety_policy(),
                unsafe_identity_policy_field: False,
            },
        }
        assert _handoff_ready_to_send(unsafe_identity_policy_handoff) is False
        assert _handoff_sendability_reasons(unsafe_identity_policy_handoff) == [
            "labeler_safety_policy_not_ready"
        ]
        identity_policy_action = _handoff_sendability_actions(
            ["labeler_safety_policy_not_ready"]
        )[0]
        assert "failed identity probes suppress launch CTAs" in identity_policy_action
        assert "diagnostic-only support URLs" in identity_policy_action

    safe_handoff = {
        **ready_handoff,
        "labeler_safety": _labeler_safety_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_session_guard_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["session_guard_policy_missing"]
    assert "session-guard policy metadata" in _handoff_sendability_actions(
        ["session_guard_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "session_guard_policy": {
            **_session_guard_policy(),
            "stale_tab_save_rejected": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["session_guard_policy_not_ready"]
    assert "stale or superseded tabs" in _handoff_sendability_actions(
        ["session_guard_policy_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "session_guard_policy": _session_guard_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_task_state_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["task_state_policy_missing"]
    assert "task-state policy metadata" in _handoff_sendability_actions(
        ["task_state_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "task_state_policy": {
            **_browser_task_state_policy(),
            "completed_tasks_read_only": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["task_state_policy_not_ready"]
    assert "completed tasks" in _handoff_sendability_actions(
        ["task_state_policy_not_ready"]
    )[0]

    unsafe_target_handoff = {
        **ready_handoff,
        "task_state_policy": {
            **_browser_task_state_policy(),
            "browser_mutation_target_token": "",
        },
    }
    assert _handoff_ready_to_send(unsafe_target_handoff) is False
    assert _handoff_sendability_reasons(unsafe_target_handoff) == ["task_state_policy_not_ready"]
    assert "current server target token" in _handoff_sendability_actions(
        ["task_state_policy_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "task_state_policy": _browser_task_state_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_zarr_backup_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["zarr_backup_policy_missing"]
    assert "Zarr backup policy metadata" in _handoff_sendability_actions(
        ["zarr_backup_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "zarr_backup_policy": {
            **_zarr_backup_policy(),
            "copy_before_labeling": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["zarr_backup_policy_not_ready"]
    assert "copy-before-labeling" in _handoff_sendability_actions(
        ["zarr_backup_policy_not_ready"]
    )[0]

    missing_plan_handoff = {
        **ready_handoff,
        "zarr_backup_policy": _zarr_backup_policy(),
        "counts": {
            **ready_handoff["counts"],
            "zarr_backup_required_targets": 1,
        },
    }
    assert _handoff_ready_to_send(missing_plan_handoff) is False
    assert _handoff_sendability_reasons(missing_plan_handoff) == ["zarr_backup_policy_not_ready"]
    assert "backup-plan artifact" in _handoff_sendability_actions(
        ["zarr_backup_policy_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "zarr_backup_policy": _zarr_backup_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []

    safe_required_plan_handoff = {
        **safe_handoff,
        "files": {
            **ready_handoff["files"],
            "zarr_backup_plan": "/handoff/zarr-backup-plan.json",
        },
        "counts": {
            **ready_handoff["counts"],
            "zarr_backup_required_targets": 1,
        },
    }
    assert _handoff_ready_to_send(safe_required_plan_handoff) is True
    assert _handoff_sendability_reasons(safe_required_plan_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_mutation_audit_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == ["mutation_audit_policy_missing"]
    assert "mutation-audit policy metadata" in _handoff_sendability_actions(
        ["mutation_audit_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "mutation_audit_policy": {
            **_mutation_audit_policy(),
            "browser_records_events_directly": True,
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == ["mutation_audit_policy_not_ready"]
    assert "server-recorded" in _handoff_sendability_actions(
        ["mutation_audit_policy_not_ready"]
    )[0]

    unsafe_fields_handoff = {
        **ready_handoff,
        "mutation_audit_policy": {
            **_mutation_audit_policy(),
            "required_event_fields": ["event_id", "task_id"],
        },
    }
    assert _handoff_ready_to_send(unsafe_fields_handoff) is False
    assert _handoff_sendability_reasons(unsafe_fields_handoff) == ["mutation_audit_policy_not_ready"]
    assert "required task/user/event fields" in _handoff_sendability_actions(
        ["mutation_audit_policy_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "mutation_audit_policy": _mutation_audit_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_handoff_sendability_blocks_missing_or_unsafe_browser_response_security_policy():
    ready_handoff = {
        "ok": True,
        "base_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {
            "is_known_labeler": True,
            "active_assignment_count": 1,
            "assignment_count": 1,
        },
        "assignment_ownership_integrity": {
            "ok": True,
            "active_assignment_count": 1,
            "duplicate_active_owner_count": 0,
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "files": {
            "html_index": "/handoff/index.html",
            "message": "/handoff/message.txt",
            "quickstart": "/handoff/labeler-quickstart.txt",
            "dataset_queue": "/handoff/dataset-queue.json",
            "manifest": "/handoff/manifest.json",
        },
        "counts": {
            "tasks": 1,
            "signed_links": 1,
            "ready_to_share_links": 1,
        },
        **_sendability_ready_baseline_fields(),
    }

    assert _handoff_ready_to_send(ready_handoff) is False
    assert _handoff_sendability_reasons(ready_handoff) == [
        "browser_response_security_policy_missing"
    ]
    assert "browser response-security policy metadata" in _handoff_sendability_actions(
        ["browser_response_security_policy_missing"]
    )[0]

    unsafe_handoff = {
        **ready_handoff,
        "browser_response_security_policy": {
            **_browser_response_security_policy(),
            "headers": {
                **_browser_response_security_policy()["headers"],
                "X-Frame-Options": "SAMEORIGIN",
            },
        },
    }
    assert _handoff_ready_to_send(unsafe_handoff) is False
    assert _handoff_sendability_reasons(unsafe_handoff) == [
        "browser_response_security_policy_not_ready"
    ]
    assert "anti-framing" in _handoff_sendability_actions(
        ["browser_response_security_policy_not_ready"]
    )[0]

    unsafe_proxy_handoff = {
        **ready_handoff,
        "browser_response_security_policy": {
            **_browser_response_security_policy(),
            "proxy_must_preserve_headers": False,
        },
    }
    assert _handoff_ready_to_send(unsafe_proxy_handoff) is False
    assert _handoff_sendability_reasons(unsafe_proxy_handoff) == [
        "browser_response_security_policy_not_ready"
    ]
    assert "proxy header preservation" in _handoff_sendability_actions(
        ["browser_response_security_policy_not_ready"]
    )[0]

    safe_handoff = {
        **ready_handoff,
        "browser_response_security_policy": _browser_response_security_policy(),
    }
    assert _handoff_ready_to_send(safe_handoff) is True
    assert _handoff_sendability_reasons(safe_handoff) == []


def test_inspect_handoff_without_store_does_not_initialize_default_store(tmp_path, monkeypatch, capsys):
    package_dir = tmp_path / "handoff"
    package_dir.mkdir()
    default_store_path = tmp_path / "default-labeling.sqlite"
    monkeypatch.setenv("PALETTE_LABELING_STORE_PATH", str(default_store_path))
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "user": "alice",
                "ready_to_send": True,
                "sendability_reasons": [],
                "counts": {"recordings": 0, "tasks": 0, "signed_links": 0},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = labeling_work.main(["inspect-handoff", "--path", str(package_dir)])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["assignment_freshness"]["checked_against_current_store"] is False
    assert payload["assignment_freshness"]["status"] == "not_checked"
    assert not default_store_path.exists()


def test_inspect_handoff_marks_empty_snapshot_stale_when_user_gets_assignment(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    package_dir = tmp_path / "handoff"
    package_dir.mkdir()
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-new", assignee_user="alice")
    finally:
        store.close()
    (package_dir / "manifest.json").write_text(
        json.dumps(
            {
                "ok": True,
                "user": "alice",
                "ready_to_send": True,
                "sendability_reasons": [],
                "assignment_snapshot": {
                    "schema": "palette.web_labeling_assignment_snapshot.v1",
                    "user": "alice",
                    "recording_count": 0,
                    "recording_ids": [],
                    "assignments": [],
                },
                "counts": {"recordings": 0, "tasks": 0, "signed_links": 0},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = labeling_work.main(["--store", str(store_path), "inspect-handoff", "--path", str(package_dir)])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["status"] == "stale_assignment"
    assert "assignment_freshness_incomplete" in payload["failure_reasons"]
    assert payload["assignment_freshness"]["expected_recording_count"] == 0
    assert payload["assignment_freshness"]["extra_current_assignment_count"] == 1
    assert payload["assignment_freshness"]["extra_current_assignments"] == [
        {
            "recording_id": "rec-new",
            "current_user": "alice",
            "current_status": "active",
        }
    ]


def test_create_session_requires_active_recording_assignment(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")

        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        assert lease.task_id == "task-a"
        assert lease.recording_id == "rec-a"
        assert lease.user == "alice"

        with pytest.raises(PermissionError):
            store.create_session(task_id="task-a", user="bob", ttl_seconds=600)

        transition = store.assign_recording_with_session_closure(recording_id="rec-a", assignee_user="bob")
        matching_assignments = [
            row
            for row in store.list_assignments(status=None)
            if row["recording_id"] == "rec-a"
        ]
        assert len(matching_assignments) == 1
        assert matching_assignments[0]["assignee_user"] == "bob"
        assert transition["closed_session_count"] == 1

        with pytest.raises(PermissionError):
            store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        bob_lease = store.create_session(task_id="task-a", user="bob", ttl_seconds=600)
        assert bob_lease.user == "bob"
    finally:
        store.close()


def test_new_session_supersedes_existing_task_session(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_training")

        first = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        second = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        assert second.superseded_session_ids == (first.session_id,)

        first_session = store.get_session(first.session_id)
        second_session = store.get_session(second.session_id)
        current = store.get_current_task_session(task_id="task-a")

        assert first_session is not None
        assert first_session["closed_at_utc"]
        assert second_session is not None
        assert not second_session["closed_at_utc"]
        assert current is not None
        assert current["session_id"] == second.session_id
    finally:
        store.close()


def test_assignment_transition_helper_closes_sessions_on_reassignment(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="operator")
        structural_contract = store.single_owner_assignment_contract()
        assert structural_contract["schema"] == "palette.web_labeling_assignment_single_owner_contract.v1"
        assert structural_contract["assignment_table"] == "recording_assignments"
        assert structural_contract["assignment_scope"] == "recording"
        assert structural_contract["recording_assignment_key"] == "recording_id"
        assert structural_contract["current_assignment_row_present"] is False
        assert structural_contract["schema_enforced_recording_primary_key"] is True
        assert structural_contract["one_current_assignment_row_per_recording"] is True
        assert structural_contract["one_active_owner_per_recording_enforced"] is True
        assert structural_contract["one_active_owner"] is True
        assert structural_contract["multiple_labelers_per_recording_allowed"] is False
        assert structural_contract["assignment_user_match_required_for_mutation"] is True
        assert structural_contract["browser_mutation_requires_current_assignment_owner"] is True
        assert structural_contract["browser_mutation_target_resolved_server_side"] is True
        assert structural_contract["browser_mutation_target_source"] == (
            "recording_assignments.active_assignment"
        )
        assert structural_contract["labelers_mutate_assigned_training_zarrs"] is True
        assert structural_contract["labelers_mutate_intermediate_csvs"] is False
        assert structural_contract["assignment_manifests_are_control_plane"] is True
        assert structural_contract["duplicate_manifest_rows_do_not_create_multiple_owners"] is True
        assert structural_contract["operator_reassignment_helper"] == (
            "assign_recording_with_session_closure"
        )
        assert structural_contract["structural_contract_met"] is True
        assert structural_contract["recording_contract_met"] is False
        assert structural_contract["single_owner_assignment_contract_met"] is True
        assert structural_contract["ready"] is True
        alice_contract = store.single_owner_assignment_contract(recording_id="rec-a")
        assert alice_contract["recording_id"] == "rec-a"
        assert alice_contract["current_assignment_row_present"] is True
        assert alice_contract["current_assignee_user"] == "alice"
        assert alice_contract["current_status"] == "active"
        assert alice_contract["current_assignment_is_active"] is True
        assert alice_contract["recording_contract_met"] is True
        assert alice_contract["single_owner_assignment_contract_met"] is True
        assert alice_contract["ready"] is True
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        result = store.assign_recording_with_session_closure(
            recording_id="rec-a",
            assignee_user="bob",
            assigned_by="operator",
            notes="handoff",
        )
        session = store.get_session(lease.session_id)
        events = store.list_events(task_id="task-a", event_type="session_closed_by_assignment_change")

        assert result["assignment"]["assignee_user"] == "bob"
        assert result["assignment_transition"]["previous_assignee_user"] == "alice"
        assert result["assignment_transition"]["new_assignee_user"] == "bob"
        assert result["assignment_transition"]["owner_changed"] is True
        assert result["assignment_transition"]["stale_sessions_closed_before_assignment_update"] is True
        assert result["assignment_transition"]["session_closure_and_assignment_update_atomic"] is True
        assert result["assignment_transition"]["session_closure_order"] == "before_assignment_update"
        assert result["assignment_transition"]["single_owner_assignment_contract_met"] is True
        assert result["assignment_transition"]["post_assignment_current_assignee_user"] == "bob"
        assert result["assignment_transition"]["post_assignment_current_status"] == "active"
        assert result["assignment_transition"]["post_assignment_current_assignment_is_active"] is True
        single_owner_contract = result["assignment_single_owner_contract"]
        assert single_owner_contract["schema"] == "palette.web_labeling_assignment_single_owner_transition.v1"
        assert single_owner_contract["assignment_table"] == "recording_assignments"
        assert single_owner_contract["assignment_scope"] == "recording"
        assert single_owner_contract["assignment_key"] == "recording_id"
        assert single_owner_contract["recording_assignment_key"] == "recording_id"
        assert single_owner_contract["recording_id"] == "rec-a"
        assert single_owner_contract["current_assignment_row_present"] is True
        assert single_owner_contract["current_assignee_user"] == "bob"
        assert single_owner_contract["current_status"] == "active"
        assert single_owner_contract["current_assignment_is_active"] is True
        assert single_owner_contract["active_owner_status_value"] == "active"
        assert single_owner_contract["current_owner_source"] == "recording_assignments.assignee_user"
        assert single_owner_contract["recording_id_primary_key"] is True
        assert single_owner_contract["schema_enforced_recording_primary_key"] is True
        assert single_owner_contract["one_row_per_recording_enforced"] is True
        assert single_owner_contract["one_current_assignment_row_per_recording"] is True
        assert single_owner_contract["one_active_owner_per_recording_enforced"] is True
        assert single_owner_contract["one_active_owner"] is True
        assert single_owner_contract["primary_key_columns"] == ["recording_id"]
        assert single_owner_contract["multiple_labelers_per_recording_allowed"] is False
        assert single_owner_contract["assignment_user_match_required_for_mutation"] is True
        assert single_owner_contract["browser_mutation_requires_current_assignment_owner"] is True
        assert single_owner_contract["browser_mutation_target_resolved_server_side"] is True
        assert single_owner_contract["labelers_mutate_assigned_training_zarrs"] is True
        assert single_owner_contract["labelers_mutate_intermediate_csvs"] is False
        assert single_owner_contract["structural_contract_met"] is True
        assert single_owner_contract["recording_contract_met"] is True
        assert single_owner_contract["single_owner_assignment_contract_met"] is True
        assert single_owner_contract["ready"] is True
        assert result["assignment_schema_integrity"]["schema_enforced_recording_primary_key"] is True
        assert result["closed_session_count"] == 1
        assert result["closed_session_ids"] == [lease.session_id]
        assert session is not None
        assert session["closed_at_utc"]
        assert len(events) == 1
        assert events[0]["user"] == "operator"
        closure_event = store.get_session_closure_event(lease.session_id)
        assert closure_event is not None
        assert closure_event["event_type"] == "session_closed_by_assignment_change"
        assert closure_event["target"]["session_id"] == lease.session_id
    finally:
        store.close()


def test_assignment_transition_rejects_invalid_target_before_closing_sessions(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="operator")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with pytest.raises(ValueError, match="assignee_user is required"):
            store.assign_recording_with_session_closure(
                recording_id="rec-a",
                assignee_user=" ",
                assigned_by="operator",
            )

        session = store.get_session(lease.session_id)
        assignment = store.get_assignment("rec-a")
        events = store.list_events(task_id="task-a", event_type="session_closed_by_assignment_change")

        assert session is not None
        assert session["closed_at_utc"] is None
        assert assignment is not None
        assert assignment["assignee_user"] == "alice"
        assert events == []
    finally:
        store.close()


def test_raw_assignment_change_blocks_stale_open_sessions_by_default(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        with pytest.raises(RuntimeError, match="assign_recording_with_session_closure"):
            store.assign_recording(recording_id="rec-a", assignee_user="bob")

        session = store.get_session(lease.session_id)
        assignment = store.get_assignment("rec-a")
        assert session is not None
        assert not session["closed_at_utc"]
        assert assignment is not None
        assert assignment["assignee_user"] == "alice"
    finally:
        store.close()


def test_completing_task_closes_sessions_and_blocks_reopen(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="subject_mask_component")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)

        updated = store.update_task_state(task_id="task-a", state="complete", user="alice")
        closed_session = store.get_session(lease.session_id)
        completed_events = store.list_events(task_id="task-a", event_type="task_completed")

        assert updated["state"] == "complete"
        assert closed_session is not None
        assert closed_session["closed_at_utc"]
        assert len(completed_events) == 1
        assert completed_events[0]["after"]["state"] == "complete"

        with pytest.raises(PermissionError):
            store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
    finally:
        store.close()


def test_backup_to_writes_consistent_sidecar_copy(tmp_path):
    store = _store(tmp_path)
    backup_path = tmp_path / "backup.sqlite"
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="backup smoke")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_detect_analysis_bbox",
            target={"frame_idx": 12},
            after={"status": "reviewed"},
        )

        result = store.backup_to(backup_path)

        assert result["backup_path"] == str(backup_path)
        assert backup_path.is_file()
        with pytest.raises(FileExistsError):
            store.backup_to(backup_path)
    finally:
        store.close()

    backup = LabelingStore(backup_path)
    try:
        assignment = backup.get_assignment("rec-a")
        task = backup.get_task("task-a")
        events = backup.list_events(task_id="task-a", event_type="save_detect_analysis_bbox")

        assert assignment is not None
        assert assignment["assignee_user"] == "alice"
        assert assignment["notes"] == "backup smoke"
        assert task is not None
        assert task["workflow_kind"] == "detect_analysis"
        assert len(events) == 1
        assert events[0]["after"]["status"] == "reviewed"
    finally:
        backup.close()


def test_zarr_backup_plan_cli_groups_mutable_targets(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "zarr-backup-plan.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            dataset_id="dataset-a",
            zarr_use="training",
            scope={
                "zarr_path": "/data/rec-a-training.zarr",
                "registry_path": "/data/palette_registry.sqlite",
            },
        )
        store.upsert_task(
            task_id="task-b",
            recording_id="rec-b",
            workflow_kind="detect_analysis",
            dataset_id="dataset-b",
            zarr_use="analysis",
            scope={
                "zarr_path": "/data/rec-b-analysis.zarr",
                "promote_training_zarr": "/data/rec-b-training.zarr",
                "registry_path": "/data/palette_registry.sqlite",
            },
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "zarr-backup-plan",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    written = json.loads(output_path.read_text())

    assert rc == 0
    assert payload == written
    assert payload["schema"] == "palette.web_labeling_zarr_backup_plan.v1"
    assert payload["policy"]["schema"] == "palette.web_labeling_zarr_backup_policy.v1"
    assert payload["policy"]["read_only_plan"] is True
    assert payload["policy"]["operator_only"] is True
    assert payload["policy"]["mutable_zarr_backup_required_before_invite"] is True
    assert payload["policy"]["validation_gate"] == "mutable_zarr_backup_confirmation"
    assert payload["policy"]["labelers_do_not_edit_zarrs_directly"] is True
    assert payload["policy"]["labelers_do_not_receive_backup_paths"] is True
    assert payload["policy"]["sidecar_backup_command"] == "execute-zarr-backup-plan"
    assert payload["policy"]["sidecar_restore_command"] == "restore-zarr-backup"
    assert payload["policy"]["backup_execution_manifest_schema"] == (
        "palette.web_labeling_zarr_backup_execution_manifest.v1"
    )
    assert payload["counts"]["zarr_targets"] == 3
    assert payload["counts"]["backup_required_targets"] == 3
    assert payload["counts"]["zarr_targets_by_role"] == {"analysis": 1, "training": 2}
    assert payload["counts"]["backup_required_targets_by_role"] == {"analysis": 1, "training": 2}
    targets = {(row["zarr_path"], row["zarr_role"]): row for row in payload["zarr_targets"]}
    assert targets[("/data/rec-a-training.zarr", "training")]["task_ids"] == ["task-a"]
    assert targets[("/data/rec-b-training.zarr", "training")]["task_ids"] == ["task-b"]
    assert targets[("/data/rec-b-analysis.zarr", "analysis")]["backup_required"] is True
    assert targets[("/data/rec-b-training.zarr", "training")]["backup_manifest_template"]["backup_schema"] == (
        "palette.web_labeling_zarr_backup.v1"
    )


def test_inspection_failure_actions_report_missing_zarr_backup_evidence(tmp_path):
    evidence_path = tmp_path / "zarr-backup-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "mutable_zarr_backup_confirmation",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "zarr_backup_evidence_template": str(evidence_path),
            }
        )
        + "\n"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_zarr_backup_evidence_template.v1",
                "targets": [
                    {
                        "target_index": 0,
                        "role": "training_zarr",
                        "status": "pending_operator_confirmation",
                        "backup_execution_manifest_path": "",
                        "backup_manifest_path": "",
                        "backup_destination": "",
                        "backup_verified_at_utc": "",
                        "restore_test_result": "",
                        "operator_approved_at_utc": "",
                    }
                ],
            }
        )
        + "\n"
    )

    inspection = _inspect_handoff_validation_checklist(tmp_path)

    status = inspection["operator_evidence_template_statuses"]["mutable_zarr_backup_confirmation"]
    assert status["ready"] is False
    assert status["target_statuses"][0]["target_index"] == 0
    assert status["target_statuses"][0]["role"] == "training_zarr"
    assert "backup_execution_manifest_path_or_backup_location" in status["target_statuses"][0][
        "missing_fields"
    ]
    assert "restore_test_result" in status["target_statuses"][0]["missing_fields"]
    assert status["targets_missing_required_fields"][0]["target_index"] == 0
    failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist=inspection,
    )
    failure_action_text = "\n".join(failure_actions)
    assert "Mutable-Zarr backup evidence incomplete for target_index=0" in failure_action_text
    assert "role=training_zarr" in failure_action_text
    assert "record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json" in failure_action_text
    assert "--execution-manifest BACKUP_EXECUTION_MANIFEST" in failure_action_text
    assert "--target-index 0" in failure_action_text
    assert "--restore-test-result RESTORE_TEST_RESULT" in failure_action_text
    repair_commands = {row["id"]: row["command"] for row in _inspection_operator_repair_commands(failure_actions)}
    assert "record_zarr_backup_evidence" in repair_commands
    assert "record-zarr-backup-evidence --evidence zarr-backup-evidence-template.json" in repair_commands[
        "record_zarr_backup_evidence"
    ]
    assert "apply_operator_evidence_templates" in repair_commands
    assert (
        "apply-operator-evidence-templates --path validation-checklist.json"
        in repair_commands["apply_operator_evidence_templates"]
    )
    assert "refresh_handoff_checksums" in repair_commands
    repair_command_rows = {row["id"]: row for row in _inspection_operator_repair_commands(failure_actions)}
    assert repair_command_rows["record_zarr_backup_evidence"]["category"] == "operator_evidence"
    assert repair_command_rows["record_zarr_backup_evidence"]["gate_ids"] == [
        "mutable_zarr_backup_confirmation"
    ]
    assert repair_command_rows["record_zarr_backup_evidence"]["requires_checksum_refresh_after_run"] is True
    assert repair_command_rows["apply_operator_evidence_templates"]["category"] == "validation_checklist"
    assert repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == [
        "mutable_zarr_backup_confirmation"
    ]
    assert (
        repair_command_rows["apply_operator_evidence_templates"]["requires_checksum_refresh_after_run"]
        is True
    )
    assert repair_command_rows["refresh_handoff_checksums"]["category"] == "checksum_refresh"
    assert repair_command_rows["refresh_handoff_checksums"]["requires_checksum_refresh_after_run"] is False


def test_zarr_backup_execute_and_restore_cli_guard_active_assignments(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    plan_path = tmp_path / "zarr-backup-plan.json"
    backup_dir = tmp_path / "zarr-backups"
    execution_manifest_path = backup_dir / "zarr-backup-execution-manifest.json"
    restore_report_path = tmp_path / "restore-report.json"
    source_zarr = tmp_path / "rec-a-training.zarr"
    source_zarr.mkdir()
    (source_zarr / "zarr.json").write_text('{"source":"original"}\n')
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            dataset_id="dataset-a",
            zarr_use="training",
            scope={"zarr_path": str(source_zarr)},
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "zarr-backup-plan",
            "--output",
            str(plan_path),
        ]
    )
    capsys.readouterr()
    assert rc == 0

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "execute-zarr-backup-plan",
            "--plan",
            str(plan_path),
            "--backup-dir",
            str(backup_dir),
            "--operator",
            "ops",
            "--output",
            str(execution_manifest_path),
        ]
    )

    backup_payload = json.loads(capsys.readouterr().out)
    backup_destination = backup_payload["targets"][0]["backup_destination"]
    target_manifest_path = backup_payload["targets"][0]["backup_manifest_path"]
    assert rc == 0
    assert backup_payload["schema"] == "palette.web_labeling_zarr_backup_execution_manifest.v1"
    assert backup_payload["ok"] is True
    assert backup_payload["targets"][0]["status"] == "backed_up"
    assert backup_payload["targets"][0]["restore_requires_paused_assignment"] is True
    assert "restore-zarr-backup" in backup_payload["targets"][0]["restore_command"]
    assert (execution_manifest_path).is_file()
    assert (tmp_path / Path(backup_destination).relative_to(tmp_path) / "zarr.json").read_text() == (
        '{"source":"original"}\n'
    )
    assert Path(target_manifest_path).is_file()

    evidence_path = tmp_path / "zarr-backup-evidence-template.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_zarr_backup_evidence_template.v1",
                "counts": {
                    "backup_required_targets": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "targets": [
                    {
                        "target_index": 0,
                        "status": "pending_operator_confirmation",
                        "backup_execution_manifest_path": "",
                        "backup_manifest_path": "",
                        "backup_destination": "",
                        "backup_created_at_utc": "",
                        "backup_verified_at_utc": "",
                        "restore_test_result": "",
                        "operator": "",
                        "operator_approved_at_utc": "",
                    }
                ],
            }
        )
        + "\n"
    )
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "record-zarr-backup-evidence",
            "--evidence",
            str(evidence_path),
            "--execution-manifest",
            str(execution_manifest_path),
            "--target-index",
            "0",
            "--restore-test-result",
            "restore drill passed",
            "--operator",
            "ops",
        ]
    )

    evidence_update_payload = json.loads(capsys.readouterr().out)
    updated_evidence = json.loads(evidence_path.read_text())
    assert rc == 0
    assert evidence_update_payload["ok"] is True
    assert evidence_update_payload["target_indexes"] == [0]
    assert updated_evidence["counts"]["operator_approved"] == 1
    assert updated_evidence["counts"]["pending_operator_confirmation"] == 0
    assert updated_evidence["targets"][0]["status"] == "operator_approved"
    assert updated_evidence["targets"][0]["backup_execution_manifest_path"] == str(execution_manifest_path)
    assert updated_evidence["targets"][0]["backup_destination"] == backup_destination
    assert updated_evidence["targets"][0]["restore_test_result"] == "restore drill passed"

    (source_zarr / "zarr.json").write_text('{"source":"mutated"}\n')
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "restore-zarr-backup",
            "--manifest",
            str(execution_manifest_path),
            "--target-index",
            "0",
            "--operator",
            "ops",
            "--replace-current",
        ]
    )

    blocked_restore = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert blocked_restore["ok"] is False
    assert blocked_restore["targets"][0]["status"] == "blocked_active_assignment"
    assert blocked_restore["errors"][0]["error"] == "active_assignment_blocks_restore"
    assert (source_zarr / "zarr.json").read_text() == '{"source":"mutated"}\n'

    with LabelingStore(store_path) as store:
        store.assign_recording(recording_id="rec-a", assignee_user="alice", status="paused")
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "restore-zarr-backup",
            "--manifest",
            str(execution_manifest_path),
            "--target-index",
            "0",
            "--operator",
            "ops",
            "--replace-current",
            "--output",
            str(restore_report_path),
        ]
    )

    restore_payload = json.loads(capsys.readouterr().out)
    written_restore_payload = json.loads(restore_report_path.read_text())
    assert rc == 0
    assert restore_payload == written_restore_payload
    assert restore_payload["ok"] is True
    assert restore_payload["targets"][0]["status"] == "restored"
    assert restore_payload["targets"][0]["moved_existing_to"]
    assert (source_zarr / "zarr.json").read_text() == '{"source":"original"}\n'


def test_assign_cli_archives_assignment_report_and_refuses_unreported_overwrite(tmp_path):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "assign-report.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "assign",
            "--recording-id",
            "rec-a",
            "--user",
            "bob",
            "--assigned-by",
            "admin",
            "--notes",
            "Bob should finish this recording",
            "--output",
            str(report_path),
        ]
    )

    assert rc == 0
    archived = json.loads(report_path.read_text())
    assert archived["ok"] is True
    assert archived["assignment"]["recording_id"] == "rec-a"
    assert archived["assignment"]["assignee_user"] == "bob"
    assert archived["assignment"]["assigned_by"] == "admin"
    assert archived["assignment"]["notes"] == "Bob should finish this recording"
    assert archived["single_owner_policy"]["one_active_owner"] is True
    assert archived["single_owner_policy"]["stale_sessions_closed_on_reassignment"] is True
    assert archived["single_owner_policy"]["stale_sessions_closed_before_assignment_update"] is True
    assert archived["single_owner_policy"]["reassignment_target_validated_before_session_closure"] is True
    assert archived["single_owner_policy"]["session_closure_and_assignment_update_atomic"] is True
    assert archived["closed_session_count"] == 1

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "assign",
                "--recording-id",
                "rec-b",
                "--user",
                "carol",
                "--output",
                str(report_path),
            ]
        )
    with LabelingStore(store_path) as store:
        assert store.get_assignment("rec-b") is None


def test_list_cli_archives_assignment_task_snapshot_and_refuses_overwrite(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "list.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints", title="Alice task")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "list",
            "--user",
            "alice",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert [row["recording_id"] for row in payload["assignments"]] == ["rec-a"]
    assert [row["task_id"] for row in payload["tasks"]] == ["task-a"]
    archived = json.loads(report_path.read_text())
    assert archived["ok"] is True
    assert archived["assignments"][0]["assignee_user"] == "alice"
    assert archived["tasks"][0]["task_id"] == "task-a"

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "list",
                "--user",
                "alice",
                "--output",
                str(report_path),
            ]
        )


def test_list_events_filters_by_task_assignee_and_actor(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis")
        save_event = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"frame_idx": 3},
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="admin",
            event_type="promotion_failed",
            target={"target_zarr": "/safe/target-a.zarr"},
        )
        store.record_event(
            task_id="task-b",
            recording_id="rec-b",
            user="bob",
            event_type="promotion_failed",
            target={"target_zarr": "/safe/target-b.zarr"},
        )

        task_events = store.list_events(task_id="task-a", limit=10)
        actor_events = store.list_events(actor_user="admin", limit=10)
        assignee_events = store.list_events(assignee_user="alice", limit=10)
        bob_failures = store.list_events(
            event_type="promotion_failed",
            assignee_user="bob",
            actor_user="bob",
            limit=10,
        )

        assert {event["task_id"] for event in task_events} == {"task-a"}
        assert {event["user"] for event in task_events} == {"alice", "admin"}
        assert [event["task_id"] for event in actor_events] == ["task-a"]
        assert {event["task_id"] for event in assignee_events} == {"task-a"}
        assert len(bob_failures) == 1
        assert bob_failures[0]["task_id"] == "task-b"
        event_lookup = store.get_event(str(save_event["event_id"]))
        assert event_lookup is not None
        assert event_lookup["event_id"] == save_event["event_id"]
        assert event_lookup["task_id"] == "task-a"
        assert event_lookup["recording_id"] == "rec-a"
        assert event_lookup["event_type"] == "save_keypoints"
        assert event_lookup["workflow_kind"] == "keypoints"
    finally:
        store.close()


def test_task_summary_redacts_failed_promotion_path_like_strings(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="detect_analysis",
            dataset_id="dataset-a",
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="promotion_failed",
            target={
                "source_frame_index": 19,
                "analysis_zarr": "/tmp/private-analysis.zarr",
                "safe_note": "operator can use frame 19",
                "safe_support_url": "/work?expected_user=alice&task_id=task-a",
                "unsafe_support_url": "/work?target=/tmp/private-analysis.zarr",
                "unsafe_boundary_url": "/workflow-private/support",
            },
            after={
                "error": "failed opening /tmp/private-analysis.zarr",
                "details": "missing relative-output.zarr for promotion",
                "support_url": "/api/sessions/session-a/open?expected_user=alice",
                "support_sentence": "open /work?expected_user=alice and inspect /tmp/private-analysis.zarr",
                "source_frame_index": 19,
            },
        )

        summary = store.task_summary_for_user("alice")
        failed = summary["failed_promotions"][0]

        assert failed["target"]["source_frame_index"] == 19
        assert failed["target"]["safe_note"] == "operator can use frame 19"
        assert failed["target"]["safe_support_url"] == "/work?expected_user=alice&task_id=task-a"
        assert failed["target"]["unsafe_support_url"] == "[redacted_path]"
        assert failed["target"]["unsafe_boundary_url"] == "[redacted_path]"
        assert failed["target"]["redacted_fields"] == ["analysis_zarr"]
        assert "analysis_zarr" not in failed["target"]
        assert failed["after"]["error"] == "failed opening [redacted_path]"
        assert failed["after"]["details"] == "missing [redacted_zarr_path] for promotion"
        assert failed["after"]["support_url"] == "/api/sessions/session-a/open?expected_user=alice"
        assert failed["after"]["support_sentence"] == "open [redacted_path] and inspect [redacted_path]"
        assert json.dumps(summary).find("/tmp/private-analysis.zarr") == -1
        assert json.dumps(summary).find("relative-output.zarr") == -1
    finally:
        store.close()


def test_list_events_filters_by_time_window(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        early = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"frame_idx": 1},
        )
        middle = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="set_review_status",
            after={"state": "needs_fix"},
        )
        late = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="task_completed",
            after={"state": "complete"},
        )
        store.conn.execute(
            "UPDATE labeling_task_events SET created_at_utc = ? WHERE event_id = ?",
            ("2026-06-23T10:00:00+00:00", early["event_id"]),
        )
        store.conn.execute(
            "UPDATE labeling_task_events SET created_at_utc = ? WHERE event_id = ?",
            ("2026-06-23T11:00:00+00:00", middle["event_id"]),
        )
        store.conn.execute(
            "UPDATE labeling_task_events SET created_at_utc = ? WHERE event_id = ?",
            ("2026-06-23T12:00:00+00:00", late["event_id"]),
        )
        store.conn.commit()

        events = store.list_events(
            task_id="task-a",
            since_utc="2026-06-23T10:30:00+00:00",
            until_utc="2026-06-23T11:30:00+00:00",
            limit=10,
        )

        assert [event["event_type"] for event in events] == ["set_review_status"]
    finally:
        store.close()


def test_export_events_cli_emits_filtered_audit_json(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="detect_analysis")
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_detect_analysis_bbox",
            target={"frame_idx": 9},
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="admin",
            event_type="promotion_failed",
            target={"target_zarr": "/safe/target-a.zarr"},
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-events",
            "--event-type",
            "promotion_failed",
            "--actor",
            "admin",
            "--since-utc",
            "2026-01-01T00:00:00+00:00",
            "--limit",
            "50",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["filters"]["actor"] == "admin"
    assert payload["filters"]["event_type"] == "promotion_failed"
    assert payload["filters"]["since_utc"] == "2026-01-01T00:00:00+00:00"
    assert payload["count"] == 1
    assert payload["events"][0]["task_id"] == "task-a"
    assert payload["events"][0]["user"] == "admin"


def test_lookup_event_cli_resolves_labeler_reported_audit_event(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "event-lookup.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        event = store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"roi_idx": 7},
            after={"changed": True},
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "lookup-event",
            "--event-id",
            str(event["event_id"]),
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    archived = json.loads(report_path.read_text())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["schema"] == "palette.web_labeling_audit_event_lookup.v1"
    assert payload["event_id"] == event["event_id"]
    assert payload["event"]["event_type"] == "save_keypoints"
    assert payload["event"]["workflow_kind"] == "keypoints"
    assert payload["event"]["recording_id"] == "rec-a"
    assert payload["event"]["target"] == {"roi_idx": 7}
    assert "assigned task" in payload["operator_action"]
    assert archived["event_id"] == event["event_id"]

    missing_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "lookup-event",
            "--event-id",
            "missing-event",
        ]
    )
    missing = json.loads(capsys.readouterr().out)
    assert missing_rc == 2
    assert missing["ok"] is False
    assert missing["error"] == "event_not_found"


def test_export_events_cli_writes_jsonl_archive(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "audit-events.jsonl"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"frame_idx": 1},
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="set_review_status",
            after={"state": "accepted"},
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-events",
            "--task-id",
            "task-a",
            "--format",
            "jsonl",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rc == 0
    assert summary["ok"] is True
    assert summary["output_path"] == str(output_path)
    assert summary["format"] == "jsonl"
    assert summary["count"] == 2
    assert {row["event_type"] for row in rows} == {"save_keypoints", "set_review_status"}

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-events",
                "--task-id",
                "task-a",
                "--format",
                "jsonl",
                "--output",
                str(output_path),
            ]
        )


def test_set_task_state_cli_completes_task_and_records_audit(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "task-state-report.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "set-task-state",
            "--task-id",
            "task-a",
            "--state",
            "complete",
            "--user",
            "operator",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    archived_payload = json.loads(report_path.read_text())
    assert rc == 0
    assert payload["ok"] is True
    assert archived_payload["task"]["task_id"] == "task-a"
    assert payload["task"]["state"] == "complete"
    assert payload["task"]["completed_at_utc"]

    store = LabelingStore(store_path)
    try:
        closed_session = store.get_session(lease.session_id)
        completed_events = store.list_events(task_id="task-a", event_type="task_completed")
        state_events = store.list_events(task_id="task-a", event_type="task_state_changed")
        close_events = store.list_events(task_id="task-a", event_type="session_closed_by_task_completion")

        assert closed_session is not None
        assert closed_session["closed_at_utc"]
        assert len(completed_events) == 1
        assert completed_events[0]["user"] == "operator"
        assert len(state_events) == 1
        assert len(close_events) == 1
    finally:
        store.close()

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "set-task-state",
                "--task-id",
                "task-a",
                "--state",
                "complete",
                "--user",
                "operator",
                "--output",
                str(report_path),
            ]
        )


def test_set_task_state_cli_reopen_records_explicit_reopen_event(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.update_task_state(task_id="task-a", state="complete", user="operator")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "set-task-state",
            "--task-id",
            "task-a",
            "--state",
            "pending",
            "--user",
            "operator",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["task"]["state"] == "pending"
    assert payload["task"]["completed_at_utc"] is None

    store = LabelingStore(store_path)
    try:
        reopened_events = store.list_events(task_id="task-a", event_type="task_reopened")
        state_events = store.list_events(task_id="task-a", event_type="task_state_changed")

        assert len(reopened_events) == 1
        assert reopened_events[0]["before"]["state"] == "complete"
        assert reopened_events[0]["after"]["state"] == "pending"
        assert len(state_events) == 2
    finally:
        store.close()


def test_set_task_state_cli_is_idempotent_for_unchanged_state(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.update_task_state(task_id="task-a", state="complete", user="operator")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "set-task-state",
            "--task-id",
            "task-a",
            "--state",
            "complete",
            "--user",
            "operator",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["task"]["state"] == "complete"

    store = LabelingStore(store_path)
    try:
        completed_events = store.list_events(task_id="task-a", event_type="task_completed")
        state_events = store.list_events(task_id="task-a", event_type="task_state_changed")
        reopened_events = store.list_events(task_id="task-a", event_type="task_reopened")

        assert len(completed_events) == 1
        assert len(state_events) == 1
        assert reopened_events == []
    finally:
        store.close()


def test_session_cli_lists_and_cleans_up_stale_sessions(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    list_report_path = tmp_path / "list-sessions.json"
    report_path = tmp_path / "cleanup-stale-sessions.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=-60)
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "list-sessions",
            "--user",
            "alice",
            "--expired-only",
            "--output",
            str(list_report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["sessions"][0]["session_id"] == lease.session_id
    archived_list = json.loads(list_report_path.read_text())
    assert archived_list["ok"] is True
    assert archived_list["filters"]["user"] == "alice"
    assert archived_list["count"] == 1
    assert archived_list["sessions"][0]["session_id"] == lease.session_id

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "list-sessions",
                "--user",
                "alice",
                "--expired-only",
                "--output",
                str(list_report_path),
            ]
        )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "cleanup-stale-sessions",
            "--user",
            "operator",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["closed_count"] == 1
    assert payload["closed_session_count"] == 1
    assert payload["closed_session_ids"] == [lease.session_id]
    assert payload["session_closure_events"][0]["event_type"] == "stale_session_closed"
    assert payload["session_closure_events"][0]["task_id"] == "task-a"
    assert payload["session_closure_events"][0]["recording_id"] == "rec-a"
    archived = json.loads(report_path.read_text())
    assert archived["ok"] is True
    assert archived["closed_count"] == 1
    assert archived["closed_session_count"] == 1
    assert archived["closed_session_ids"] == [lease.session_id]
    assert archived["sessions"][0]["session_id"] == lease.session_id

    store = LabelingStore(store_path)
    try:
        session = store.get_session(lease.session_id)
        events = store.list_events(task_id="task-a", event_type="stale_session_closed")

        assert session is not None
        assert session["closed_at_utc"]
        assert len(events) == 1
        assert events[0]["user"] == "operator"
        assert events[0]["target"]["session_id"] == lease.session_id
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
        second_lease = store.create_session(task_id="task-b", user="bob", ttl_seconds=-60)
    finally:
        store.close()

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "cleanup-stale-sessions",
                "--user",
                "operator",
                "--output",
                str(report_path),
            ]
        )
    store = LabelingStore(store_path)
    try:
        session = store.get_session(second_lease.session_id)
        events = store.list_events(task_id="task-b", event_type="stale_session_closed")

        assert session is not None
        assert not session["closed_at_utc"]
        assert events == []
    finally:
        store.close()


def test_repair_reassignment_sessions_cli_closes_only_mismatched_sessions(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "repair-reassignment-sessions.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a-old", recording_id="rec-a", workflow_kind="keypoints")
        old_lease = store.create_session(task_id="task-a-old", user="alice", ttl_seconds=600)
        store.assign_recording(
            recording_id="rec-a",
            assignee_user="bob",
            allow_stale_open_sessions=True,
        )
        store.upsert_task(task_id="task-a-new", recording_id="rec-a", workflow_kind="keypoints")
        store.assign_recording(recording_id="rec-b", assignee_user="charlie")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
        charlie_lease = store.create_session(task_id="task-b", user="charlie", ttl_seconds=600)
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "repair-reassignment-sessions",
            "--user",
            "operator",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["recording_ids"] == ["rec-a"]
    assert payload["closed_count"] == 1
    assert payload["closed_session_count"] == 1
    assert payload["closed_session_ids"] == [old_lease.session_id]
    assert payload["reassignment_session_safety_before"]["ok"] is False
    assert payload["reassignment_session_safety_after"]["ok"] is True
    assert payload["repairs"][0]["recording_id"] == "rec-a"
    assert payload["repairs"][0]["closed_session_ids"] == [old_lease.session_id]
    archived = json.loads(report_path.read_text())
    assert archived["closed_session_ids"] == [old_lease.session_id]

    store = LabelingStore(store_path)
    try:
        old_session = store.get_session(old_lease.session_id)
        current_session = store.get_session(charlie_lease.session_id)
        repair_events = store.list_events(
            task_id="task-a-old",
            event_type="session_closed_by_reassignment_safety_repair",
        )

        assert old_session is not None
        assert old_session["closed_at_utc"]
        assert current_session is not None
        assert not current_session["closed_at_utc"]
        assert len(repair_events) == 1
        assert repair_events[0]["user"] == "operator"
        assert repair_events[0]["target"]["session_id"] == old_lease.session_id
        assert repair_events[0]["target"]["repair_reason"] == "assignment_mismatched_session"
        assert repair_events[0]["before"]["session_user"] == "alice"
        assert repair_events[0]["before"]["assignment_user"] == "bob"
    finally:
        store.close()


def test_check_store_cli_archives_consistency_report_and_refuses_overwrite(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "check-store.json"

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "check-store",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["store_path"] == str(store_path)
    assert payload["single_owner_policy"]["one_active_owner"] is True
    assert payload["assignment_ownership_integrity"]["ok"] is True
    assert payload["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    archived = json.loads(report_path.read_text())
    assert archived["ok"] is True
    assert archived["store_path"] == str(store_path)
    assert archived["assignment_ownership_integrity"] == payload["assignment_ownership_integrity"]

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "check-store",
                "--output",
                str(report_path),
            ]
        )


def test_server_cli_accepts_browser_work_operator_validation_alias():
    args = labeling_web_module.build_parser().parse_args(
        [
            "--store",
            "labeling_work.sqlite",
            "preflight",
            "--require-operator-validation-for-browser-work",
        ]
    )

    assert args.require_operator_validation_for_browser_work is True
    assert args.require_operator_validation_for_start is False


def test_preflight_cli_archives_server_safety_report_and_refuses_overwrite(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "preflight.json"

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "preflight",
            "--production",
            "--trust-auth-header",
            "--admin-user",
            "admin@example.org",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["error_count"] == 0
    assert payload["preflight"]["production_enabled"] is True
    assert payload["preflight"]["trusted_auth_header_enabled"] is True
    assert payload["preflight"]["admin_user_count"] == 1
    assert payload["preflight"]["operator_authorization_policy"]["admin_routes_require_operator"] is True
    assert payload["preflight"]["operator_authorization_policy"]["admin_users_configured"] is True
    assert payload["preflight"]["operator_authorization_policy"]["operator_boundary_ready"] is True
    assert payload["preflight"]["operator_authorization_policy"]["operator_boundary_known"] is True
    assert payload["preflight"]["operator_authorization_policy"]["runtime_preflight_required"] is False
    assert payload["preflight"]["operator_authorization_policy"]["admin_users"] == ["admin@example.org"]
    assert payload["preflight"]["operator_recovery_contract"]["reassignment_closes_previous_owner_sessions"] is True
    assert payload["preflight"]["operator_recovery_contract"][
        "reassignment_closes_previous_owner_sessions_before_assignment_update"
    ] is True
    assert payload["preflight"]["operator_recovery_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert payload["preflight"]["operator_recovery_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert payload["preflight"]["dataset_queue_direct_start_policy"]["enabled"] is True
    assert payload["preflight"]["dataset_queue_direct_start_policy"]["endpoint_route_template"] == (
        "/api/tasks/{task_id}/open"
    )
    assert payload["preflight"]["dataset_queue_direct_start_policy"][
        "endpoint_task_segment_must_match_row_task_id"
    ] is True
    assert payload["preflight"]["operator_validation_start_gate"]["schema"] == (
        "palette.web_labeling_runtime_operator_validation_start_gate.v1"
    )
    assert payload["preflight"]["operator_validation_start_gate"][
        "required_for_browser_start"
    ] is False
    assert payload["preflight"]["operator_validation_start_gate"]["ready"] is True
    assert payload["preflight"]["operator_validation_start_gate"][
        "blocks_task_open"
    ] is False
    assert payload["preflight"]["operator_validation_mutation_gate"]["schema"] == (
        "palette.web_labeling_runtime_operator_validation_mutation_gate.v1"
    )
    assert payload["preflight"]["operator_validation_mutation_gate"][
        "required_for_browser_mutation"
    ] is False
    assert payload["preflight"]["operator_validation_mutation_gate"]["ready"] is True
    assert payload["preflight"]["operator_validation_mutation_gate"][
        "blocks_browser_mutation"
    ] is False
    gate_cli_policy = payload["preflight"]["runtime_operator_validation_gate_cli_policy"]
    assert gate_cli_policy["schema"] == (
        "palette.web_labeling_runtime_operator_validation_gate_cli_policy.v1"
    )
    assert gate_cli_policy["validation_checklist_flag"] == "--validation-checklist"
    assert gate_cli_policy["preferred_require_flag"] == (
        "--require-operator-validation-for-browser-work"
    )
    assert gate_cli_policy["legacy_require_flag"] == (
        "--require-operator-validation-for-start"
    )
    assert gate_cli_policy["legacy_require_flag_retained_for_compatibility"] is True
    assert gate_cli_policy["protects_browser_start_open"] is True
    assert gate_cli_policy["protects_browser_mutations"] is True
    assert gate_cli_policy["blocks_before_session_creation"] is True
    assert gate_cli_policy["blocks_before_target_token_check"] is True
    assert gate_cli_policy["blocks_before_zarr_write"] is True
    assert gate_cli_policy["blocks_before_audit_event_creation"] is True
    assert "production_operator_validation_start_gate_disabled" in payload["preflight"][
        "warnings"
    ]
    assert "production_operator_validation_mutation_gate_disabled" in payload["preflight"][
        "warnings"
    ]
    assert payload["preflight"]["browser_mutation_write_policy"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["preflight"]["browser_mutation_write_checklist"]["ready"] is True
    assert payload["preflight"]["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert payload["preflight"]["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert payload["preflight"]["browser_response_security_policy"]["no_store_cache"] is True
    assert payload["preflight"]["browser_response_security_policy"]["headers"]["X-Frame-Options"] == "DENY"
    assert payload["preflight"]["browser_response_security_policy"]["headers"]["Content-Security-Policy"] == (
        "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'"
    )
    assert payload["preflight"]["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke=missing_evidence" in payload["preflight"][
        "safe_share_next_action_summary"
    ]
    archived = json.loads(report_path.read_text())
    assert archived["ok"] is True
    assert archived["preflight"]["admin_users"] == ["admin@example.org"]
    assert archived["preflight"]["operator_authorization_policy"]["admin_users"] == ["admin@example.org"]
    assert archived["preflight"]["operator_recovery_contract"] == payload["preflight"][
        "operator_recovery_contract"
    ]
    assert archived["preflight"]["dataset_queue_direct_start_policy"] == payload["preflight"][
        "dataset_queue_direct_start_policy"
    ]
    assert archived["preflight"]["browser_mutation_write_checklist"] == payload["preflight"][
        "browser_mutation_write_checklist"
    ]
    assert archived["preflight"]["safe_share_next_action_summary"] == payload["preflight"][
        "safe_share_next_action_summary"
    ]

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "preflight",
                "--production",
                "--trust-auth-header",
                "--admin-user",
                "admin@example.org",
                "--output",
                str(report_path),
            ]
        )


def test_import_assignments_cli_dry_run_and_apply(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    manifest_path = tmp_path / "assignments.json"
    report_path = tmp_path / "assignments-report.json"
    manifest_path.write_text(
        json.dumps(
            {
                "assignments": [
                    {"recording_id": "rec-a", "assignee_user": "alice", "notes": "keypoints first"},
                    {"recording_id": "rec-b", "user": "bob", "status": "paused"},
                ]
            }
        ),
        encoding="utf-8",
    )

    dry_run_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
            "--output",
            str(report_path),
        ]
    )

    dry_run_payload = json.loads(capsys.readouterr().out)
    archived_payload = json.loads(report_path.read_text())
    assert dry_run_rc == 0
    assert dry_run_payload["ok"] is True
    assert dry_run_payload["dry_run"] is True
    assert archived_payload["count"] == dry_run_payload["count"]
    assert dry_run_payload["count"] == 2
    assert dry_run_payload["applied_count"] == 0
    assert {row["recording_id"] for row in dry_run_payload["assignments"]} == {"rec-a", "rec-b"}

    store = LabelingStore(store_path)
    try:
        assert store.get_assignment("rec-a") is None
        assert store.get_assignment("rec-b") is None
    finally:
        store.close()

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "import-assignments",
                "--input",
                str(manifest_path),
                "--output",
                str(report_path),
            ]
        )

    apply_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
            "--apply",
        ]
    )

    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_rc == 0
    assert apply_payload["ok"] is True
    assert apply_payload["dry_run"] is False
    assert apply_payload["applied_count"] == 2

    store = LabelingStore(store_path)
    try:
        rec_a = store.get_assignment("rec-a")
        rec_b = store.get_assignment("rec-b")

        assert rec_a is not None
        assert rec_a["assignee_user"] == "alice"
        assert rec_a["assigned_by"] == "operator"
        assert rec_a["notes"] == "keypoints first"
        assert rec_b is not None
        assert rec_b["assignee_user"] == "bob"
        assert rec_b["status"] == "paused"
    finally:
        store.close()


def test_import_assignments_cli_accepts_csv_manifest(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    manifest_path = tmp_path / "assignments.csv"
    manifest_path.write_text(
        "recording_id,assignee_user,status,notes\n"
        "rec-a,alice,active,keypoints first\n"
        "rec-b,bob,paused,waiting on data\n"
        "\n",
        encoding="utf-8",
    )

    dry_run_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
        ]
    )

    dry_run_payload = json.loads(capsys.readouterr().out)
    assert dry_run_rc == 0
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["count"] == 2
    assert {row["recording_id"] for row in dry_run_payload["assignments"]} == {"rec-a", "rec-b"}
    assert {row["source_line"] for row in dry_run_payload["assignments"]} == {2, 3}

    apply_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
            "--apply",
        ]
    )

    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_rc == 0
    assert apply_payload["applied_count"] == 2
    assert {row["source_line"] for row in apply_payload["assignments"]} == {2, 3}

    store = LabelingStore(store_path)
    try:
        rec_a = store.get_assignment("rec-a")
        rec_b = store.get_assignment("rec-b")
        assert rec_a is not None
        assert rec_a["assignee_user"] == "alice"
        assert rec_a["notes"] == "keypoints first"
        assert rec_b is not None
        assert rec_b["assignee_user"] == "bob"
        assert rec_b["status"] == "paused"
    finally:
        store.close()


def test_import_assignments_cli_rejects_csv_missing_required_headers(tmp_path):
    manifest_path = tmp_path / "assignments.csv"
    manifest_path.write_text(
        "recording_id,status,notes\n"
        "rec-a,active,missing user\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assignee_user or user"):
        labeling_work.main(
            [
                "--store",
                str(tmp_path / "labeling_work.sqlite"),
                "import-assignments",
                "--input",
                str(manifest_path),
            ]
        )


def test_sign_links_cli_exports_active_incomplete_task_links(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "links.jsonl"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="alice", status="paused")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints", title="Open task")
        store.upsert_task(task_id="task-complete", recording_id="rec-a", workflow_kind="keypoints", state="complete")
        store.upsert_task(task_id="task-paused", recording_id="rec-b", workflow_kind="detect_analysis")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "sign-links",
            "--user",
            "alice",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--format",
            "jsonl",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rc == 0
    assert summary["ok"] is True
    assert summary["count"] == 1
    assert summary["ready_to_share"] is True
    assert summary["ready_to_share_count"] == 1
    assert summary["not_ready_to_share_count"] == 0
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-a"
    assert rows[0]["assignee_user"] == "alice"
    assert rows[0]["url"].startswith("https://labeling.example.org/t/")
    assert rows[0]["url_is_absolute"] is True
    assert rows[0]["ready_to_share"] is True
    assert rows[0]["task_launchable"] is True
    assert rows[0]["shareability_warnings"] == []
    assert rows[0]["issued_at_utc"]
    assert rows[0]["expires_at_utc"]
    assert rows[0]["expires_in_seconds"] >= 60

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "sign-links",
            "--user",
            "alice",
            "--include-completed",
            "--link-secret",
            "test-secret",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["ready_to_share"] is False
    assert payload["ready_to_share_count"] == 0
    assert payload["not_ready_to_share_count"] == 2
    assert payload["shareability_warnings"][0]["code"] == "missing_base_url"
    assert {row["task_id"] for row in payload["links"]} == {"task-a", "task-complete"}
    assert {row["ready_to_share"] for row in payload["links"]} == {False}
    assert {row["url_is_absolute"] for row in payload["links"]} == {False}
    assert {row["task_launchable"] for row in payload["links"]} == {False, True}
    warnings_by_task = {
        row["task_id"]: [warning["code"] for warning in row["shareability_warnings"]]
        for row in payload["links"]
    }
    assert warnings_by_task["task-a"] == ["missing_base_url"]
    assert warnings_by_task["task-complete"] == ["missing_base_url", "task_completed"]

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "sign-links",
                "--user",
                "alice",
                "--link-secret",
                "test-secret",
                "--output",
                str(output_path),
            ]
        )


def test_sign_link_cli_reports_direct_link_launchability(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "task-a-link.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-paused", assignee_user="bob", status="paused")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-blocked", recording_id="rec-a", workflow_kind="keypoints", state="blocked")
        store.upsert_task(task_id="task-paused", recording_id="rec-paused", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "sign-link",
            "--task-id",
            "task-a",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--output",
            str(output_path),
        ]
    )

    active_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert active_payload["ready_to_share"] is True
    assert active_payload["task_launchable"] is True
    assert active_payload["url_is_absolute"] is True
    assert active_payload["assignment_status"] == "active"
    assert active_payload["shareability_warnings"] == []
    archived = json.loads(output_path.read_text())
    assert archived["task_id"] == "task-a"
    assert archived["url"] == active_payload["url"]
    assert archived["ready_to_share"] is True

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "sign-link",
                "--task-id",
                "task-a",
                "--link-secret",
                "test-secret",
                "--output",
                str(output_path),
            ]
        )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "sign-link",
            "--task-id",
            "task-paused",
            "--link-secret",
            "test-secret",
        ]
    )

    paused_payload = json.loads(capsys.readouterr().out)
    warning_codes = [warning["code"] for warning in paused_payload["shareability_warnings"]]
    assert rc == 0
    assert paused_payload["ready_to_share"] is False
    assert paused_payload["task_launchable"] is False
    assert paused_payload["url_is_absolute"] is False
    assert paused_payload["assignment_status"] == "paused"
    assert warning_codes == ["missing_base_url", "task_assignment_not_active"]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "sign-link",
            "--task-id",
            "task-blocked",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
        ]
    )

    blocked_payload = json.loads(capsys.readouterr().out)
    warning_codes = [warning["code"] for warning in blocked_payload["shareability_warnings"]]
    assert rc == 0
    assert blocked_payload["ready_to_share"] is False
    assert blocked_payload["task_launchable"] is False
    assert blocked_payload["url_is_absolute"] is True
    assert blocked_payload["assignment_status"] == "active"
    assert blocked_payload["state"] == "blocked"
    assert blocked_payload["startable_task_states"] == ["pending", "in_progress"]
    assert warning_codes == ["task_not_startable"]
    assert blocked_payload["shareability_warnings"][0]["startable_task_states"] == ["pending", "in_progress"]


def test_sign_links_cli_marks_non_startable_tasks_not_ready(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-blocked", recording_id="rec-a", workflow_kind="keypoints", state="blocked")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "sign-links",
            "--user",
            "alice",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    links_by_task = {row["task_id"]: row for row in payload["links"]}
    assert rc == 0
    assert payload["ok"] is True
    assert payload["startable_task_states"] == ["pending", "in_progress"]
    assert payload["ready_to_share"] is False
    assert payload["ready_to_share_count"] == 1
    assert payload["not_ready_to_share_count"] == 1
    assert links_by_task["task-a"]["ready_to_share"] is True
    assert links_by_task["task-a"]["task_launchable"] is True
    assert links_by_task["task-blocked"]["ready_to_share"] is False
    assert links_by_task["task-blocked"]["task_launchable"] is False
    assert links_by_task["task-blocked"]["state"] == "blocked"
    assert links_by_task["task-blocked"]["startable_task_states"] == ["pending", "in_progress"]
    assert [warning["code"] for warning in links_by_task["task-blocked"]["shareability_warnings"]] == [
        "task_not_startable"
    ]


def test_import_assignments_reapply_is_idempotent_and_reassignment_closes_sessions(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    manifest_path = tmp_path / "assignments.json"
    duplicate_manifest_path = tmp_path / "duplicate-assignments.csv"
    blocked_reassignment_path = tmp_path / "blocked-reassignment.json"
    manifest_path.write_text(
        json.dumps({"assignments": [{"recording_id": "rec-a", "assignee_user": "alice", "notes": "batch"}]}),
        encoding="utf-8",
    )
    duplicate_manifest_path.write_text(
        "recording_id,assignee_user,status\n"
        "rec-a,alice,active\n"
        "rec-a,bob,active\n",
        encoding="utf-8",
    )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
            "--apply",
        ]
    )
    assert rc == 0
    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_payload["single_owner_policy"]["one_active_owner"] is True
    assert apply_payload["single_owner_policy"]["reassignment_replaces_owner"] is True
    assert apply_payload["assignment_ownership_integrity"]["primary_key_columns"] == ["recording_id"]
    assert apply_payload["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    assert apply_payload["assignment_ownership_contract"]["one_active_owner"] is True
    assert apply_payload["assignment_ownership_contract"][
        "duplicate_manifest_rows_do_not_create_multiple_owners"
    ] is True
    assert apply_payload["assignment_manifest_artifact_role"] == "metadata_only_control_plane"
    assert apply_payload["assignment_manifest_artifacts_are_label_write_targets"] is False
    assert apply_payload["assignment_manifest_browser_writes_label_data"] is False
    assert apply_payload["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert apply_payload["browser_mutation_write_policy"]["csv_handoff_artifacts_are_label_write_targets"] is False
    assert apply_payload["browser_mutation_write_checklist"]["browser_writes_csv_or_handoff_files"] is False
    assert apply_payload["browser_mutation_write_checklist"]["browser_has_direct_zarr_write_authority"] is False

    store = LabelingStore(store_path)
    try:
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        lease = store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        assigned_at = store.get_assignment("rec-a")["assigned_at_utc"]
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
            "--apply",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["generated_at_utc"]
    assert payload["blocked_by_warnings"] is False
    assert payload["warning_count"] == 0
    assert payload["warning_codes"] == []
    assert payload["assignments"][0]["would_change"] is False
    assert payload["assignments"][0]["closed_session_count"] == 0

    store = LabelingStore(store_path)
    try:
        session = store.get_session(lease.session_id)
        assignment = store.get_assignment("rec-a")

        assert session is not None
        assert session["closed_at_utc"] is None
        assert assignment["assigned_at_utc"] == assigned_at
    finally:
        store.close()

    manifest_path.write_text(
        json.dumps({"assignments": [{"recording_id": "rec-a", "assignee_user": "bob", "notes": "handoff"}]}),
        encoding="utf-8",
    )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(manifest_path),
            "--assigned-by",
            "operator",
            "--apply",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["warning_count"] == 1
    assert payload["warning_codes"] == ["assignment_reassigns_existing_recording"]
    assert payload["warnings"][0]["previous_assignee_user"] == "alice"
    assert payload["warnings"][0]["new_assignee_user"] == "bob"
    assert payload["assignments"][0]["warnings"][0]["code"] == "assignment_reassigns_existing_recording"
    assert payload["assignments"][0]["would_change"] is True
    assert payload["assignments"][0]["closed_session_count"] == 1

    store = LabelingStore(store_path)
    try:
        session = store.get_session(lease.session_id)
        events = store.list_events(task_id="task-a", event_type="session_closed_by_assignment_change")
        assignment = store.get_assignment("rec-a")

        assert session is not None
        assert session["closed_at_utc"]
        assert assignment["assignee_user"] == "bob"
        assert len(events) == 1
        assert events[0]["user"] == "operator"
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(duplicate_manifest_path),
        ]
    )

    duplicate_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert duplicate_payload["warning_count"] == 2
    assert duplicate_payload["warning_codes"] == [
        "assignment_reassigns_existing_recording",
        "duplicate_recording_assignment_rows",
    ]
    assert duplicate_payload["warnings"][0]["code"] == "duplicate_recording_assignment_rows"
    assert duplicate_payload["warnings"][0]["source_lines"] == [2, 3]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(duplicate_manifest_path),
            "--apply",
        ]
    )

    duplicate_apply_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert duplicate_apply_payload["input_row_count"] == 2
    assert duplicate_apply_payload["count"] == 2
    assert duplicate_apply_payload["applied_count"] == 1
    assert duplicate_apply_payload["deduplicated_apply_count"] == 1
    assert duplicate_apply_payload["skipped_duplicate_apply_count"] == 1
    assert duplicate_apply_payload["assignments"][0]["skipped_by_duplicate_apply"] is True
    assert duplicate_apply_payload["assignments"][0]["applied"] is False
    assert duplicate_apply_payload["assignments"][0]["warnings"][0]["code"] == "duplicate_assignment_row_skipped_for_apply"
    assert duplicate_apply_payload["assignments"][1]["skipped_by_duplicate_apply"] is False
    assert duplicate_apply_payload["assignments"][1]["applied"] is True
    assert duplicate_apply_payload["assignments"][1]["assignment"]["assignee_user"] == "bob"
    assert duplicate_apply_payload["assignment_ownership_integrity"]["active_assignment_count"] == 1
    assert duplicate_apply_payload["assignment_ownership_integrity"]["unique_active_recording_count"] == 1
    assert duplicate_apply_payload["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    assert duplicate_apply_payload["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert duplicate_apply_payload["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    store = LabelingStore(store_path)
    try:
        assert store.get_assignment("rec-a")["assignee_user"] == "bob"
    finally:
        store.close()

    blocked_reassignment_path.write_text(
        json.dumps({"assignments": [{"recording_id": "rec-a", "assignee_user": "alice", "notes": "blocked"}]}),
        encoding="utf-8",
    )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-assignments",
            "--input",
            str(blocked_reassignment_path),
            "--assigned-by",
            "operator",
            "--warnings-as-errors",
            "--apply",
        ]
    )

    blocked_payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert blocked_payload["ok"] is False
    assert blocked_payload["generated_at_utc"]
    assert blocked_payload["blocked_by_warnings"] is True
    assert blocked_payload["warnings_as_errors"] is True
    assert blocked_payload["warning_codes"] == ["assignment_reassigns_existing_recording"]
    assert blocked_payload["blocking_warning_count"] == 1
    assert blocked_payload["applied_count"] == 0
    store = LabelingStore(store_path)
    try:
        assert store.get_assignment("rec-a")["assignee_user"] == "bob"
    finally:
        store.close()


def test_assignment_events_record_create_change_and_skip_unchanged(tmp_path):
    store = _store(tmp_path)
    try:
        store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="operator", notes="first")
        store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="operator", notes="first")
        store.assign_recording(recording_id="rec-a", assignee_user="bob", assigned_by="operator", notes="handoff")

        events = store.list_assignment_events(recording_id="rec-a", actor_user="operator", limit=10)

        assert [event["event_type"] for event in events] == ["assignment_changed", "assignment_created"]
        assert events[0]["before"]["assignee_user"] == "alice"
        assert events[0]["after"]["assignee_user"] == "bob"
        assert events[1]["before"] is None
        assert events[1]["after"]["assignee_user"] == "alice"
    finally:
        store.close()


def test_export_assignment_events_cli_writes_jsonl_archive(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "assignment-events.jsonl"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="operator")
        store.assign_recording(recording_id="rec-a", assignee_user="bob", assigned_by="operator")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-assignment-events",
            "--recording-id",
            "rec-a",
            "--actor",
            "operator",
            "--format",
            "jsonl",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rc == 0
    assert summary["ok"] is True
    assert summary["count"] == 2
    assert summary["output_path"] == str(output_path)
    assert [row["event_type"] for row in rows] == ["assignment_changed", "assignment_created"]

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-assignment-events",
                "--recording-id",
                "rec-a",
                "--output",
                str(output_path),
            ]
        )


def test_add_task_cli_reports_assignment_visibility_warnings(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    report_path = tmp_path / "add-task-report.json"

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "add-task",
            "--task-id",
            "task-missing-assignment",
            "--recording-id",
            "rec-missing",
            "--workflow-kind",
            "keypoints",
            "--output",
            str(report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    archived_payload = json.loads(report_path.read_text())
    assert rc == 0
    assert payload["ok"] is True
    assert payload["applied"] is True
    assert archived_payload["task"]["task_id"] == "task-missing-assignment"
    assert payload["warning_codes"] == ["task_recording_missing_assignment"]
    assert payload["warnings"][0]["recording_id"] == "rec-missing"
    assert payload["task"]["task_id"] == "task-missing-assignment"

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "add-task",
            "--task-id",
            "task-blocked",
            "--recording-id",
            "rec-blocked",
            "--workflow-kind",
            "keypoints",
            "--warnings-as-errors",
        ]
    )

    blocked_payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert blocked_payload["ok"] is False
    assert blocked_payload["applied"] is False
    assert blocked_payload["blocked_by_warnings"] is True
    assert blocked_payload["blocking_warning_codes"] == ["task_recording_missing_assignment"]
    assert blocked_payload["task"] is None

    store = LabelingStore(store_path)
    try:
        assert store.get_task("task-missing-assignment") is not None
        assert store.get_task("task-blocked") is None
    finally:
        store.close()

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "add-task",
                "--task-id",
                "task-existing-output",
                "--recording-id",
                "rec-missing",
                "--workflow-kind",
                "keypoints",
                "--output",
                str(report_path),
            ]
        )


def test_task_generation_cli_payload_summarizes_skipped_rows():
    payload = labeling_work._task_generation_cli_payload(
        {
            "registry_path": "/tmp/registry.sqlite",
            "assignment_count": 1,
            "dataset_count": 2,
            "generated_count": 1,
            "skipped_count": 1,
            "generated": [],
            "skipped": [
                {
                    "dataset_id": "dataset-a",
                    "recording_id": "rec-a",
                    "reason": "not_keypoint_reviewable",
                }
            ],
        }
    )

    assert payload["ok"] is True
    assert payload["generated_at_utc"]
    assert payload["warning_count"] == 1
    assert payload["warning_codes"] == ["generation_skipped_not_keypoint_reviewable"]
    assert payload["warnings_as_errors"] is False
    assert payload["failed_by_warnings"] is False
    assert payload["blocking_warning_count"] == 0
    assert payload["warnings"][0]["dataset_id"] == "dataset-a"
    assert payload["warnings"][0]["recording_id"] == "rec-a"
    assert payload["warnings"][0]["reason"] == "not_keypoint_reviewable"

    strict_payload = labeling_work._task_generation_cli_payload(
        {
            "registry_path": "/tmp/registry.sqlite",
            "assignment_count": 1,
            "dataset_count": 1,
            "generated_count": 0,
            "skipped_count": 1,
            "generated": [],
            "skipped": [{"dataset_id": "dataset-a", "reason": "already_approved"}],
        },
        warnings_as_errors=True,
    )

    assert strict_payload["ok"] is False
    assert strict_payload["warnings_as_errors"] is True
    assert strict_payload["failed_by_warnings"] is True
    assert strict_payload["blocking_warning_count"] == 1
    assert strict_payload["blocking_warning_codes"] == ["generation_skipped_already_approved"]


def test_write_optional_json_report_refuses_overwrite(tmp_path):
    output_path = tmp_path / "generation-report.json"
    payload = {"ok": True, "generated_count": 0}

    labeling_work._write_optional_json_report(
        payload,
        str(output_path),
        overwrite=False,
        description="task-generation report",
    )

    assert json.loads(output_path.read_text()) == payload
    with pytest.raises(FileExistsError):
        labeling_work._write_optional_json_report(
            payload,
            str(output_path),
            overwrite=False,
            description="task-generation report",
        )
    labeling_work._write_optional_json_report(
        {"ok": True, "generated_count": 1},
        str(output_path),
        overwrite=True,
        description="task-generation report",
    )
    assert json.loads(output_path.read_text())["generated_count"] == 1


def test_import_tasks_cli_dry_run_apply_and_reapply(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    manifest_path = tmp_path / "tasks.json"
    report_path = tmp_path / "tasks-report.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task-a",
                        "recording_id": "rec-a",
                        "workflow_kind": "keypoints",
                        "title": "Review keypoints",
                        "scope": {"frames": [1, 2, 3]},
                        "priority": 5,
                    },
                    {
                        "task_id": "task-b",
                        "recording_id": "rec-b",
                        "workflow_kind": "detect_analysis",
                        "state": "pending",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    dry_run_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-tasks",
            "--input",
            str(manifest_path),
            "--output",
            str(report_path),
        ]
    )

    dry_run_payload = json.loads(capsys.readouterr().out)
    archived_payload = json.loads(report_path.read_text())
    assert dry_run_rc == 0
    assert dry_run_payload["ok"] is True
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["generated_at_utc"]
    assert archived_payload["count"] == dry_run_payload["count"]
    assert dry_run_payload["blocked_by_warnings"] is False
    assert dry_run_payload["applied_count"] == 0
    assert dry_run_payload["warning_count"] == 2
    assert dry_run_payload["warning_codes"] == ["task_recording_missing_assignment"]
    assert {row["task_id"] for row in dry_run_payload["tasks"]} == {"task-a", "task-b"}
    assert {row["warnings"][0]["code"] for row in dry_run_payload["tasks"]} == {"task_recording_missing_assignment"}

    store = LabelingStore(store_path)
    try:
        assert store.get_task("task-a") is None
    finally:
        store.close()

    strict_rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-tasks",
            "--input",
            str(manifest_path),
            "--warnings-as-errors",
            "--apply",
        ]
    )

    strict_payload = json.loads(capsys.readouterr().out)
    assert strict_rc == 2
    assert strict_payload["ok"] is False
    assert strict_payload["generated_at_utc"]
    assert strict_payload["blocked_by_warnings"] is True
    assert strict_payload["applied_count"] == 0
    assert strict_payload["blocking_warning_count"] == 2
    assert strict_payload["blocking_warning_codes"] == ["task_recording_missing_assignment"]
    store = LabelingStore(store_path)
    try:
        assert store.get_task("task-a") is None
    finally:
        store.close()

    apply_rc = labeling_work.main(["--store", str(store_path), "import-tasks", "--input", str(manifest_path), "--apply"])

    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_rc == 0
    assert apply_payload["ok"] is True
    assert apply_payload["dry_run"] is False
    assert apply_payload["applied_count"] == 2
    assert apply_payload["warning_count"] == 2

    store = LabelingStore(store_path)
    try:
        task = store.get_task("task-a")
        assert task is not None
        assert task["recording_id"] == "rec-a"
        assert task["workflow_kind"] == "keypoints"
        assert task["title"] == "Review keypoints"
        assert task["scope"] == {"frames": [1, 2, 3]}
        assert task["priority"] == 5
    finally:
        store.close()

    reapply_rc = labeling_work.main(["--store", str(store_path), "import-tasks", "--input", str(manifest_path)])

    reapply_payload = json.loads(capsys.readouterr().out)
    assert reapply_rc == 0
    assert reapply_payload["ok"] is True
    assert {row["would_change"] for row in reapply_payload["tasks"]} == {False}

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "import-tasks",
                "--input",
                str(manifest_path),
                "--output",
                str(report_path),
            ]
        )


def test_import_tasks_cli_accepts_csv_manifest_with_scope_json(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    manifest_path = tmp_path / "tasks.csv"
    manifest_path.write_text(
        "task_id,recording_id,workflow_kind,title,scope_json,priority,notes\n"
        'task-a,rec-a,keypoints,Review keypoints,"{""frames"":[1,2,3]}",7,spreadsheet task\n'
        'task-b,rec-b,detect_analysis,Review boxes,"{""frames"":[4]}",2,\n'
        "\n",
        encoding="utf-8",
    )

    dry_run_rc = labeling_work.main(["--store", str(store_path), "import-tasks", "--input", str(manifest_path)])

    dry_run_payload = json.loads(capsys.readouterr().out)
    assert dry_run_rc == 0
    assert dry_run_payload["ok"] is True
    assert dry_run_payload["dry_run"] is True
    assert dry_run_payload["warning_count"] == 2
    assert {row["task_id"] for row in dry_run_payload["tasks"]} == {"task-a", "task-b"}
    assert {row["source_line"] for row in dry_run_payload["tasks"]} == {2, 3}

    apply_rc = labeling_work.main(["--store", str(store_path), "import-tasks", "--input", str(manifest_path), "--apply"])

    apply_payload = json.loads(capsys.readouterr().out)
    assert apply_rc == 0
    assert apply_payload["applied_count"] == 2
    assert {row["source_line"] for row in apply_payload["tasks"]} == {2, 3}

    store = LabelingStore(store_path)
    try:
        task = store.get_task("task-a")
        assert task is not None
        assert task["title"] == "Review keypoints"
        assert task["scope"] == {"frames": [1, 2, 3]}
        assert task["priority"] == 7
        assert task["notes"] == "spreadsheet task"
    finally:
        store.close()


def test_import_tasks_cli_rejects_csv_missing_required_headers(tmp_path):
    manifest_path = tmp_path / "tasks.csv"
    manifest_path.write_text(
        "task_id,recording_id,title\n"
        "task-a,rec-a,missing workflow\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workflow_kind"):
        labeling_work.main(
            [
                "--store",
                str(tmp_path / "labeling_work.sqlite"),
                "import-tasks",
                "--input",
                str(manifest_path),
            ]
        )


def test_write_manifest_templates_cli_writes_assignment_and_task_csvs(tmp_path, capsys):
    output_dir = tmp_path / "templates"

    rc = labeling_work.main(
        [
            "--store",
            str(tmp_path / "labeling_work.sqlite"),
            "write-manifest-templates",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assignment_rows = list(csv.DictReader((output_dir / "assignments-template.csv").open()))
    task_rows = list(csv.DictReader((output_dir / "tasks-template.csv").open()))
    readme = (output_dir / "manifest-templates-readme.txt").read_text()

    assert rc == 0
    assert payload["ok"] is True
    assert payload["files"]["assignments_template"] == str(output_dir / "assignments-template.csv")
    assert payload["files"]["tasks_template"] == str(output_dir / "tasks-template.csv")
    assert payload["files"]["readme"] == str(output_dir / "manifest-templates-readme.txt")
    assert assignment_rows[0]["recording_id"] == "recording-a"
    assert assignment_rows[0]["assignee_user"] == "alice"
    assert task_rows[0]["task_id"] == "recording-a-keypoints-review"
    assert task_rows[0]["workflow_kind"] == "keypoints"
    assert json.loads(task_rows[0]["scope_json"]) == {"frames": [1, 2, 3]}
    assert "Dry-run assignments" in readme
    assert "source_line" in readme

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(tmp_path / "labeling_work.sqlite"),
                "write-manifest-templates",
                "--output-dir",
                str(output_dir),
            ]
        )


def test_export_user_handoffs_without_base_url_is_preview_not_ready(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "handoffs-no-base-url"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Alice task",
            priority=7,
            notes="Task-specific instructions",
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-user-handoffs",
            "--link-secret",
            "test-secret",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    index = json.loads((output_dir / "index.json").read_text())
    html_index = (output_dir / "index.html").read_text()
    readme = (output_dir / "handoff-readme.txt").read_text()
    validation_log = (output_dir / "validation-log-template.md").read_text()
    roster_rows = list(csv.DictReader((output_dir / "labeler-roster.csv").open()))
    inspection = _inspect_handoff_package(output_dir)
    alice_html = (output_dir / "alice" / "index.html").read_text()
    alice_message = (output_dir / "alice" / "message.txt").read_text()
    alice_quickstart = (output_dir / "alice" / "labeler-quickstart.txt").read_text()

    assert rc == 2
    assert payload["ok"] is False
    assert payload["store_checks_ok"] is True
    assert payload["dashboard_path"] == "/work"
    assert payload["dashboard_url"] == ""
    assert payload["labeler_safety"]["dashboard_identity_check_required"] is True
    assert payload["labeler_safety"]["labeler_runtime_surface"] == "browser"
    assert payload["labeler_safety"]["requires_local_palette_installation"] is False
    assert payload["labeler_safety"]["requires_local_crimson_installation"] is False
    assert payload["labeler_safety"]["requires_local_conda_environment"] is False
    assert payload["labeler_safety"]["requires_local_project_dependencies"] is False
    assert payload["labeler_safety"]["labeler_landing_page_path"] == "/"
    assert payload["labeler_safety"]["labeler_landing_page_kind"] == "datasets_waiting_queue"
    assert payload["labeler_safety"]["landing_serves_datasets_waiting_queue"] is True
    assert payload["labeler_safety"]["datasets_waiting_alias_paths"] == [
        "/",
        "/me",
        "/labeling",
        "/datasets",
        "/my-datasets",
    ]
    assert payload["labeler_safety"]["dashboard_is_fallback"] is True
    assert payload["labeler_safety"]["queue_first_landing_paths"] == [
        "/",
        "/me",
        "/labeling",
        "/datasets",
        "/my-datasets",
    ]
    assert payload["labeler_safety"]["dataset_queue_page_path"] == "/datasets"
    assert payload["labeler_safety"]["personal_dataset_queue_page_path"] == "/my-datasets"
    assert payload["labeler_safety"]["personal_work_page_path"] == "/my-work"
    assert payload["labeler_safety"]["work_filter_query_keys"] == [
        "expected_user",
        "dataset_id",
        "recording_id",
        "task_id",
        "workflow",
    ]
    assert payload["labeler_safety"]["expected_user_guards"]["dataset_queue_page"] == "dashboard_user_mismatch"
    assert payload["labeler_safety"]["expected_user_guards"]["personal_work_page"] == "dashboard_user_mismatch"
    assert payload["labeler_safety"]["expected_user_guards"]["personal_dataset_queue_page"] == (
        "dashboard_user_mismatch"
    )
    assert payload["labeler_safety"]["expected_user_guards"]["labeler_landing_page"] == "dashboard_user_mismatch"
    assert payload["labeler_safety"]["expected_user_guards"]["labeler_me_page"] == "dashboard_user_mismatch"
    assert payload["labeler_safety"]["expected_user_guards"]["task_open_api"] == "task_open_user_mismatch"
    assert payload["labeler_safety"]["expected_user_guards"]["promotion_retry_api"] == "promotion_retry_user_mismatch"
    assert payload["labeler_safety"]["promotion_retry_labeler_mutation_enabled"] is False
    assert payload["labeler_safety"]["promotion_retry_labeler_rejection_error"] == (
        "operator_support_required"
    )
    assert payload["task_state_policy"]["labeler_promotion_retry_mutation_enabled"] is False
    assert payload["task_state_policy"]["labeler_promotion_retry_rejection_error"] == (
        "operator_support_required"
    )
    assert payload["labeler_safety"]["expected_user_guards"]["signed_task_link"] == "signed_link_user_mismatch"
    assert payload["labeler_safety"]["personal_work_expected_user_guard_supported"] is True
    assert payload["labeler_safety"]["personal_dataset_queue_expected_user_guard_supported"] is True
    assert payload["labeler_safety"]["identity_probe_expected_user_guard_required"] is True
    assert payload["labeler_safety"]["identity_probe_diagnostic_only"] is True
    assert payload["labeler_safety"]["identity_probe_does_not_authorize_work"] is True
    assert payload["labeler_safety"]["identity_probe_unknown_user_blocks_work_surfaces"] is True
    assert payload["labeler_safety"]["identity_probe_success_launch_ctas_rendered"] is True
    assert payload["labeler_safety"]["identity_probe_failed_launch_ctas_suppressed"] is True
    assert (
        payload["labeler_safety"]["identity_probe_failed_support_urls_diagnostic_only"]
        is True
    )
    assert payload["labeler_safety"]["browser_response_security_policy"]["proxy_must_preserve_headers"] is True
    assert payload["labeler_safety"]["browser_receives_task_scope"] is False
    assert payload["labeler_safety"]["browser_receives_raw_zarr_paths"] is False
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_runtime_state_paths"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_mutation_response_paths"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_error_detail_paths"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_path_like_string_values"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["admin_diagnostics_unredacted"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_direct_storage_paths"] is True
    assert payload["task_state_policy"]["completed_tasks_read_only"] is True
    assert payload["signed_link_policy"]["authorization_grant"] is False
    assert payload["signed_link_policy"]["binds_expected_user_in_new_links"] is True
    assert {row["workflow_kind"] for row in payload["browser_workflows"]} == {
        "keypoints",
        "detect_training",
        "detect_analysis",
        "subject_mask_component",
    }
    workflow_write_contracts = {
        str(row["workflow_kind"]): row["write_contract"]
        for row in payload["browser_workflows"]
    }
    assert {
        contract["csv_handoff_artifact_role"]
        for contract in workflow_write_contracts.values()
    } == {"metadata_only_control_plane"}
    assert {
        contract["csv_handoff_artifacts_are_label_write_targets"]
        for contract in workflow_write_contracts.values()
    } == {False}
    assert {
        contract["browser_label_write_target"]
        for contract in workflow_write_contracts.values()
    } == {"training_zarr"}
    assert {
        contract["browser_writes_csv_or_handoff_files"]
        for contract in workflow_write_contracts.values()
    } == {False}
    assert {
        contract["browser_writes_handoff_csv"]
        for contract in workflow_write_contracts.values()
    } == {False}
    assert {
        contract["browser_writes_intermediate_csv"]
        for contract in workflow_write_contracts.values()
    } == {False}
    assert {
        contract["training_zarr_mutation_target_kind"]
        for contract in workflow_write_contracts.values()
    } == {"task_scoped_training_zarr"}
    assert workflow_write_contracts["detect_training"]["training_zarr_write_mode"] == "direct"
    assert workflow_write_contracts["detect_analysis"]["training_zarr_write_mode"] == (
        "promotion_when_configured"
    )
    assert workflow_write_contracts["detect_analysis"]["primary_mutation_target_kind"] == (
        "task_scoped_analysis_zarr"
    )
    assert workflow_write_contracts["detect_analysis"]["promotion_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["counts"]["ready_to_send"] == 0
    assert payload["counts"]["not_ready_to_send"] == 1
    assert payload["counts"]["sendability_reasons"] == {
        "missing_base_url": 1,
        "operator_validation_needs_review": 1,
    }
    assert "--base-url" in payload["sendability_actions"][0]
    assert payload["sendability_warnings"][0]["user"] == "alice"
    assert payload["sendability_warnings"][0]["reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert "--base-url" in payload["sendability_warnings"][0]["actions"][0]
    assert index["counts"]["ready_to_send"] == 0
    assert index["counts"]["not_ready_to_send"] == 1
    assert index["counts"]["sendability_reasons"] == {
        "missing_base_url": 1,
        "operator_validation_needs_review": 1,
    }
    assert index["ok"] is False
    assert index["store_checks_ok"] is True
    assert index["dashboard_path"] == "/work"
    assert index["dashboard_url"] == ""
    assert index["labeler_safety"]["dashboard_identity_check_required"] is True
    assert index["labeler_safety"]["expected_user_guards"] == payload["labeler_safety"]["expected_user_guards"]
    assert index["task_state_policy"] == payload["task_state_policy"]
    assert index["signed_link_policy"] == payload["signed_link_policy"]
    assert index["browser_workflows"] == payload["browser_workflows"]
    assert "--base-url" in index["sendability_actions"][0]
    assert index["handoffs"][0]["ready_to_send"] is False
    assert index["handoffs"][0]["sendability_reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert "--base-url" in index["handoffs"][0]["sendability_actions"][0]
    assert index["handoffs"][0]["dashboard_url"] == ""
    assert index["handoffs"][0]["dataset_queue_state_code"] == "has_open_dataset_work"
    assert index["handoffs"][0]["labeler_work_completion_status"] == "waiting"
    assert index["handoffs"][0]["labeler_work_completion_has_waiting_work"] is True
    assert index["handoffs"][0]["labeler_work_completion_completed"] is False
    assert index["handoffs"][0]["labeler_work_completion_ready_for_more_labeling"] is True
    assert index["handoffs"][0]["dataset_queue_blocks_labeler_start"] is False
    assert index["handoffs"][0]["known_labeler"] is True
    assert index["handoffs"][0]["known_user_active_assignment_count"] == 1
    assert index["handoffs"][0]["known_user_readiness"] == "passed"
    assert index["handoffs"][0]["assignment_ownership_ok"] is True
    assert index["handoffs"][0]["assignment_duplicate_active_owner_count"] == 0
    assert index["handoffs"][0]["assignment_ownership_readiness"] == "passed"
    assert index["handoffs"][0]["guarded_links_ready"] is False
    assert "absolute_base_url" in index["handoffs"][0]["missing_guarded_links"]
    assert index["handoffs"][0]["handoff_artifacts_ready"] is True
    assert index["handoffs"][0]["missing_handoff_artifacts"] == []
    assert index["handoffs"][0]["handoff_entry_readiness"] == "missing_base_url"
    assert index["handoffs"][0]["dataset_queue_start_ready"] is True
    assert index["handoffs"][0]["dataset_queue_start_status"] == "passed"
    assert index["handoffs"][0]["labeler_safety"]["dashboard_identity_check_required"] is True
    assert index["handoffs"][0]["task_state_policy"] == index["task_state_policy"]
    assert index["handoffs"][0]["signed_link_policy"] == index["signed_link_policy"]
    assert index["handoffs"][0]["browser_workflows"] == index["browser_workflows"]
    assert roster_rows[0]["ready_to_send"] == "False"
    assert roster_rows[0]["known_labeler"] == "True"
    assert roster_rows[0]["known_user_active_assignment_count"] == "1"
    assert roster_rows[0]["known_user_readiness"] == "passed"
    assert roster_rows[0]["assignment_ownership_ok"] == "True"
    assert roster_rows[0]["assignment_duplicate_active_owner_count"] == "0"
    assert roster_rows[0]["assignment_ownership_readiness"] == "passed"
    assert roster_rows[0]["assignment_ownership_contract_ready"] == "True"
    assert roster_rows[0]["assignment_ownership_contract_assignment_scope"] == "recording"
    assert roster_rows[0]["assignment_ownership_contract_recording_assignment_key"] == (
        "recording_id"
    )
    assert roster_rows[0]["assignment_ownership_contract_primary_key_columns"] == (
        '["recording_id"]'
    )
    assert roster_rows[0]["assignment_ownership_contract_one_active_owner"] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_multiple_labelers_per_recording_allowed"
    ] == "False"
    assert roster_rows[0][
        "assignment_ownership_contract_browser_mutation_requires_current_assignment_owner"
    ] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_browser_mutation_target_resolved_server_side"
    ] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_browser_mutation_target_source"
    ] == "recording_assignments.active_assignment"
    assert roster_rows[0][
        "assignment_ownership_contract_labelers_mutate_assigned_training_zarrs"
    ] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_labelers_mutate_intermediate_csvs"
    ] == "False"
    assert roster_rows[0][
        "assignment_ownership_contract_store_single_owner_assignment_contract_present"
    ] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_store_single_owner_assignment_contract_ready"
    ] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_store_single_owner_assignment_contract_met"
    ] == "True"
    assert roster_rows[0][
        "assignment_ownership_contract_store_single_owner_assignment_contract_schema"
    ] == "palette.web_labeling_assignment_single_owner_contract.v1"
    assert roster_rows[0]["assignment_ownership_contract_duplicate_active_owner_count"] == "0"
    assert roster_rows[0]["guarded_links_ready"] == "False"
    assert "absolute_base_url" in roster_rows[0]["missing_guarded_links"]
    assert roster_rows[0]["handoff_artifacts_ready"] == "True"
    assert roster_rows[0]["missing_handoff_artifacts"] == "[]"
    assert roster_rows[0]["handoff_entry_readiness"] == "missing_base_url"
    assert (
        roster_rows[0]["sendability_reasons"]
        == "missing_base_url, operator_validation_needs_review"
    )
    assert "--base-url" in roster_rows[0]["sendability_actions"]
    assert roster_rows[0]["dashboard_url"] == ""
    assert roster_rows[0]["dataset_queue_state_code"] == "has_open_dataset_work"
    assert roster_rows[0]["labeler_work_completion_status"] == "waiting"
    assert roster_rows[0]["labeler_work_completion_has_waiting_work"] == "True"
    assert roster_rows[0]["labeler_work_completion_completed"] == "False"
    assert roster_rows[0]["labeler_work_completion_ready_for_more_labeling"] == "True"
    assert roster_rows[0]["dataset_queue_blocks_labeler_start"] == "False"
    assert roster_rows[0]["dataset_queue_start_ready"] == "True"
    assert roster_rows[0]["dataset_queue_start_status"] == "passed"
    assert "Reasons" in html_index
    assert "Guarded links" in html_index
    assert "Entry readiness" in html_index
    assert "Queue state" in html_index
    assert "Start status" in html_index
    assert "not ready" in html_index
    assert "missing_base_url" in html_index
    assert "Store checks ok: True" in html_index
    assert "Store checks ok: True" in readme
    assert "Dashboard URL: (missing --base-url)" in html_index
    assert "Dashboard URL: (missing --base-url)" in readme
    assert (
        'Not-ready reasons: {"missing_base_url": 1, '
        '"operator_validation_needs_review": 1}'
    ) in readme
    assert "--base-url" in readme
    assert "Wait for operator review" in alice_html
    assert "Review reasons: missing_base_url" in alice_html
    assert "This handoff was generated without a service URL" in alice_html
    assert "Needs service URL: /t/" in alice_html
    assert "Your Palette labeling handoff needs operator review before starting." in alice_message
    assert "Review reasons: missing_base_url" in alice_message
    assert "Your Palette labeling work is ready." not in alice_message
    assert "This handoff was generated without a service base URL." in alice_message
    assert "record-browser-smoke-evidence --evidence" not in alice_message
    assert "apply-operator-evidence-templates --path" not in alice_message
    assert "record-zarr-backup-evidence --evidence" not in alice_message
    assert "Wait for operator review before starting." in alice_quickstart
    assert "Review reasons: missing_base_url" in alice_quickstart
    assert "How to preview while waiting:" in alice_quickstart
    assert "the handoff was generated without a service URL" in alice_quickstart
    assert "record-browser-smoke-evidence --evidence" not in alice_quickstart
    assert "apply-operator-evidence-templates --path" not in alice_quickstart
    assert "record-zarr-backup-evidence --evidence" not in alice_quickstart

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )

    inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert inspection["status"] == "needs_review"
    assert "handoff_not_ready" in inspection["failure_reasons"]
    assert "validation_checklist_needs_review" in inspection["failure_reasons"]
    assert "validation_evidence_pending" in inspection["failure_reasons"]
    assert inspection["validation_log"]["present"] is True
    assert inspection["validation_log"]["matched_paths"] == [str(output_dir / "validation-log-template.md")]
    assert inspection["validation_checklist"]["present"] is True
    assert inspection["validation_checklist"]["ready_for_operator_validation"] is False
    assert inspection["validation_checklist"]["dataset_queue_page_path"] == "/datasets"
    assert inspection["validation_checklist"]["expected_user_dataset_queue_url"] == ""
    assert "identity_probe_verification" in inspection["validation_checklist"]["operator_evidence_gate_ids"]
    assert "static_readiness" in inspection["validation_checklist"]["generated_contract_gate_ids"]
    assert "identity_probe_verification" in inspection["validation_checklist"][
        "operator_evidence_needs_review_gate_ids"
    ]
    assert "browser_response_security_headers" in inspection["validation_checklist"][
        "operator_evidence_needs_review_gate_ids"
    ]
    assert "static_readiness" not in inspection["validation_checklist"]["generated_contract_failed_gate_ids"]
    assert "identity_probe_verification" in inspection["validation_checklist"]["needs_review_gate_ids"]
    assert "browser_response_security_headers" in inspection["validation_checklist"]["needs_review_gate_ids"]
    assert inspection["validation_checklist"]["evidence_recorded_gate_ids"] == []
    assert "identity_probe_verification" in inspection["validation_checklist"]["required_missing_evidence_gate_ids"]
    assert "browser_response_security_headers" in inspection["validation_checklist"]["required_missing_evidence_gate_ids"]
    assert "static_readiness" not in inspection["validation_checklist"]["required_missing_evidence_gate_ids"]
    assert inspection["validation_checklist"]["evidence_recorded_gate_count"] == 0
    assert inspection["validation_checklist"]["required_missing_evidence_gate_count"] >= 1
    assert inspection["counts"]["sendability_reasons"] == {
        "missing_base_url": 1,
        "operator_validation_needs_review": 1,
    }
    assert inspection["counts"]["dataset_queue_states"] == {"has_open_dataset_work": 1}
    assert inspection["counts"]["dataset_queue_blocked_start_users"] == []
    assert inspection["handoffs"][0]["ready_to_send"] is False
    assert inspection["handoffs"][0]["sendability_reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert inspection["handoffs"][0]["dataset_queue_state_code"] == "has_open_dataset_work"
    assert inspection["handoffs"][0]["dataset_queue_blocks_labeler_start"] is False
    assert inspection["handoffs"][0]["dataset_queue_start_ready"] is True
    assert inspection["handoffs"][0]["dataset_queue_start_status"] == "passed"


def test_export_user_handoffs_blocks_reassignment_session_safety_issue(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "handoffs-reassignment-session-safety"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(
            recording_id="rec-a",
            assignee_user="bob",
            allow_stale_open_sessions=True,
        )
        store.assign_recording(recording_id="rec-b", assignee_user="charlie")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-user-handoffs",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    index = json.loads((output_dir / "index.json").read_text())
    manifest = json.loads((output_dir / "bob" / "manifest.json").read_text())
    work_summary = json.loads((output_dir / "bob" / "work-summary.json").read_text())
    dataset_queue_payload = json.loads((output_dir / "bob" / "dataset-queue.json").read_text())
    charlie_manifest = json.loads((output_dir / "charlie" / "manifest.json").read_text())
    charlie_work_summary = json.loads(
        (output_dir / "charlie" / "work-summary.json").read_text()
    )
    charlie_dataset_queue_payload = json.loads(
        (output_dir / "charlie" / "dataset-queue.json").read_text()
    )
    roster_rows = list(csv.DictReader((output_dir / "labeler-roster.csv").open()))

    assert rc == 2
    assert payload["ok"] is False
    assert payload["store_checks_ok"] is False
    assert index["store_checks_ok"] is False
    assert manifest["ok"] is False
    assert manifest["reassignment_session_safety"]["ok"] is False
    assert manifest["reassignment_session_safety"][
        "active_session_assignment_mismatch_count"
    ] == 1
    assert manifest["reassignment_session_safety"]["blocks_labeler_mutation"] is True
    assert manifest["counts"]["reassignment_session_safety_ok"] is False
    assert manifest["counts"]["reassignment_session_safety_mismatch_count"] == 1
    assert manifest["counts"]["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert work_summary["reassignment_session_safety"]["ok"] is False
    assert work_summary["reassignment_session_safety_ok"] is False
    assert work_summary["reassignment_session_safety"][
        "active_session_assignment_mismatch_count"
    ] == 1
    assert (
        work_summary["reassignment_session_safety_active_session_assignment_mismatch_count"]
        == 1
    )
    assert work_summary["work"]["reassignment_session_safety"]["blocks_labeler_mutation"] is True
    assert work_summary["work"]["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert dataset_queue_payload["reassignment_session_safety"]["ok"] is False
    assert dataset_queue_payload["reassignment_session_safety_ok"] is False
    assert dataset_queue_payload["reassignment_session_safety"][
        "active_session_assignment_mismatch_recording_ids"
    ] == ["rec-a"]
    assert dataset_queue_payload[
        "reassignment_session_safety_active_session_assignment_mismatch_recording_ids"
    ] == ["rec-a"]
    assert manifest["labeler_start_ready"] is False
    assert manifest["labeler_start_status"] == "reassignment_session_safety_failed"
    assert manifest["dataset_queue_state"]["blocks_labeler_start"] is True
    assert manifest["dataset_queue_state"]["code"] == "reassignment_session_safety_failed"
    assert "reassignment_session_safety_failed" in manifest["sendability_reasons"]
    assert "store_check_failed" in manifest["sendability_reasons"]
    assert "stale previous-owner sessions" in " ".join(manifest["sendability_actions"])
    assert payload["counts"]["sendability_reasons"]["reassignment_session_safety_failed"] == 1
    assert index["counts"]["sendability_reasons"]["reassignment_session_safety_failed"] == 1
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )
    inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert inspection["counts"]["reassignment_session_safety_blocked_users"] == ["bob"]
    assert inspection["counts"]["reassignment_session_safety_blocked_user_count"] == 1
    assert inspection["counts"]["reassignment_session_safety_mismatch_count"] == 1
    assert inspection["counts"]["reassignment_session_safety_blocked_recording_ids"] == ["rec-a"]
    assert "reassignment_session_safety_failed" in inspection["failure_reasons"]
    assert any(
        "assign_recording_with_session_closure" in action
        for action in inspection["failure_actions"]
    )
    repair_commands = {row["id"]: row for row in inspection["operator_repair_commands"]}
    assert repair_commands["repair_reassignment_sessions"]["category"] == "session_repair"
    assert "repair-reassignment-sessions --user OPERATOR" in repair_commands[
        "repair_reassignment_sessions"
    ]["command"]
    assert repair_commands["repair_reassignment_sessions"][
        "requires_checksum_refresh_after_run"
    ] is False
    inspection_by_user = {row["user"]: row for row in inspection["handoffs"]}
    assert inspection_by_user["bob"]["reassignment_session_safety"]["ok"] is False
    assert inspection_by_user["bob"]["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert (
        inspection_by_user["bob"][
            "reassignment_session_safety_active_session_assignment_mismatch_count"
        ]
        == 1
    )
    assert inspection_by_user["charlie"]["reassignment_session_safety"]["ok"] is True
    assert inspection_by_user["charlie"]["reassignment_session_safety_blocks_labeler_mutation"] is False
    assert charlie_manifest["ok"] is True
    assert charlie_manifest["reassignment_session_safety"]["ok"] is True
    assert charlie_manifest["counts"]["reassignment_session_safety_mismatch_count"] == 0
    assert charlie_work_summary["reassignment_session_safety"]["ok"] is True
    assert charlie_work_summary["reassignment_session_safety_ok"] is True
    assert (
        charlie_work_summary["reassignment_session_safety"][
            "active_session_assignment_mismatch_count"
        ]
        == 0
    )
    assert (
        charlie_work_summary[
            "reassignment_session_safety_active_session_assignment_mismatch_count"
        ]
        == 0
    )
    assert charlie_dataset_queue_payload["reassignment_session_safety"]["ok"] is True
    assert charlie_dataset_queue_payload["reassignment_session_safety_ok"] is True
    assert (
        charlie_dataset_queue_payload["reassignment_session_safety"][
            "active_session_assignment_mismatch_recording_ids"
        ]
        == []
    )
    assert (
        charlie_dataset_queue_payload[
            "reassignment_session_safety_active_session_assignment_mismatch_recording_ids"
        ]
        == []
    )
    assert "reassignment_session_safety_failed" not in charlie_manifest["sendability_reasons"]
    assert "store_check_failed" not in charlie_manifest["sendability_reasons"]
    roster_by_user = {row["user"]: row for row in roster_rows}
    assert roster_by_user["bob"]["ready_to_send"] == "False"
    assert "reassignment_session_safety_failed" in roster_by_user["bob"]["sendability_reasons"]
    assert "stale previous-owner sessions" in roster_by_user["bob"]["sendability_actions"]
    assert roster_by_user["bob"]["reassignment_session_safety_ok"] == "False"
    assert roster_by_user["bob"]["reassignment_session_safety_blocks_labeler_mutation"] == "True"
    assert (
        roster_by_user["bob"][
            "reassignment_session_safety_active_session_assignment_mismatch_count"
        ]
        == "1"
    )
    assert "rec-a" in roster_by_user["bob"][
        "reassignment_session_safety_active_session_assignment_mismatch_recording_ids"
    ]
    assert "assign_recording_with_session_closure" in roster_by_user["bob"][
        "reassignment_session_safety_operator_action"
    ]
    assert "reassignment_session_safety_failed" not in roster_by_user["charlie"]["sendability_reasons"]
    assert "store_check_failed" not in roster_by_user["charlie"]["sendability_reasons"]
    assert roster_by_user["charlie"]["reassignment_session_safety_ok"] == "True"
    assert roster_by_user["charlie"]["reassignment_session_safety_blocks_labeler_mutation"] == "False"
    assert (
        roster_by_user["charlie"][
            "reassignment_session_safety_active_session_assignment_mismatch_count"
        ]
        == "0"
    )


def test_update_validation_checklist_cli_records_gate_evidence(tmp_path, capsys):
    checklist_path = tmp_path / "validation-checklist.json"
    updated_path = tmp_path / "updated-validation-checklist.json"
    log_path = tmp_path / "validation-log-template.md"
    log_path.write_text("# Web Labeling Validation Log\n", encoding="utf-8")
    checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "bundle_label": "launch bundle",
                "all_validation_complete": False,
                "ready_for_operator_validation": True,
                "counts": {
                    "assignment_ownership_duplicate_active_owners": 0,
                    "gates": 2,
                    "passed": 1,
                    "pending_operator_evidence": 1,
                },
                "gates": [
                    {
                        "id": "static_readiness",
                        "title": "Static readiness and store checks",
                        "status": "passed",
                        "required": True,
                        "blocks_invitation": True,
                    },
                    {
                        "id": "browser_smoke",
                        "title": "Browser smoke",
                        "status": "pending_operator_evidence",
                        "required": True,
                        "blocks_invitation": True,
                        "evidence_files": [str(log_path)],
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = labeling_work.main(
        [
            "--store",
            str(tmp_path / "labeling_work.sqlite"),
            "update-validation-checklist",
            "--path",
            str(checklist_path),
            "--gate",
            "browser_smoke",
            "--status",
            "passed",
            "--evidence",
            "Opened /work, /admin, task session, failure state, completion, and reopen.",
            "--evidence-file",
            str(log_path),
            "--append-log",
            str(log_path),
            "--operator",
            "operator@example.org",
            "--output",
            str(updated_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(updated_path.read_text())
    log_text = log_path.read_text()
    browser_gate = next(gate for gate in updated["gates"] if gate["id"] == "browser_smoke")
    checklist_summary = _validation_checklist_gate_summary(updated)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["path"] == str(checklist_path)
    assert payload["output"] == str(updated_path)
    assert payload["gate_id"] == "browser_smoke"
    assert payload["previous_status"] == "pending_operator_evidence"
    assert payload["status"] == "passed"
    assert payload["all_validation_complete"] is True
    assert payload["ready_for_operator_validation"] is True
    assert payload["operator_evidence_gate_ids"] == ["browser_smoke"]
    assert payload["generated_contract_gate_ids"] == ["static_readiness"]
    assert payload["operator_evidence_pending_gate_ids"] == []
    assert payload["operator_evidence_needs_review_gate_ids"] == []
    assert payload["generated_contract_failed_gate_ids"] == []
    assert payload["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
    assert payload["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert payload["safe_share_ready_to_send_is_sufficient"] is False
    assert payload["safe_share_required_inspection_field"] == "labeler_links_safe_to_share"
    assert payload["safe_share_checklist_gate_evidence_complete"] is False
    assert "browser_smoke" in payload["safe_share_launch_blocking_satisfied_gate_ids"]
    assert "identity_probe_verification" in payload["safe_share_launch_blocking_missing_gate_ids"]
    assert payload["counts"] == {
        "assignment_ownership_duplicate_active_owners": 0,
        "generated_contract_failed_gates": 0,
        "generated_contract_gates": 1,
        "gates": 2,
        "operator_evidence_gates": 1,
        "operator_evidence_needs_review_gates": 0,
        "operator_evidence_pending_gates": 0,
        "passed": 2,
    }
    assert {gate["id"]: gate["status"] for gate in payload["available_gates"]} == {
        "static_readiness": "passed",
        "browser_smoke": "passed",
    }
    assert {
        gate["blocks_invitation_legacy_semantics"]
        for gate in payload["available_gates"]
    } == {"blocks_ready_row_draft_or_launch_readiness_not_safe_share_approval"}
    assert {gate["blocks_invitation_is_safe_share_approval"] for gate in payload["available_gates"]} == {
        False
    }
    assert {gate["blocks_invitation_safe_share_field"] for gate in payload["available_gates"]} == {
        "labeler_links_safe_to_share"
    }
    assert payload["validation_log_appended"] is True
    assert payload["validation_log"] == str(log_path)
    assert updated["all_validation_complete"] is True
    assert updated["validation_gate_classification"]["operator_evidence_gate_ids"] == ["browser_smoke"]
    assert updated["operator_evidence_gate_ids"] == ["browser_smoke"]
    assert updated["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
    assert updated["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert updated["safe_share_ready_to_send_is_sufficient"] is False
    assert updated["safe_share_checklist_gate_evidence_complete"] is False
    assert checklist_summary["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert checklist_summary["safe_share_ready_to_send_is_sufficient"] is False
    assert checklist_summary["safe_share_checklist_gate_evidence_complete"] is False
    assert "browser_smoke" in checklist_summary["safe_share_launch_blocking_satisfied_gate_ids"]
    assert updated["generated_contract_gate_ids"] == ["static_readiness"]
    assert updated["operator_evidence_pending_gate_ids"] == []
    assert updated["operator_evidence_needs_review_gate_ids"] == []
    assert updated["generated_contract_failed_gate_ids"] == []
    assert updated["counts"] == payload["counts"]
    assert browser_gate["status"] == "passed"
    assert browser_gate["blocks_invitation_legacy_semantics"] == (
        "blocks_ready_row_draft_or_launch_readiness_not_safe_share_approval"
    )
    assert browser_gate["blocks_invitation_is_safe_share_approval"] is False
    assert browser_gate["blocks_invitation_safe_share_field"] == "labeler_links_safe_to_share"
    assert browser_gate["evidence_recorded_by"] == "operator@example.org"
    assert browser_gate["evidence_notes"] == [
        "Opened /work, /admin, task session, failure state, completion, and reopen."
    ]
    assert browser_gate["evidence_files"] == [str(log_path)]
    assert browser_gate["evidence"][0]["previous_status"] == "pending_operator_evidence"
    assert browser_gate["evidence"][0]["status"] == "passed"
    assert "browser_smoke" in checklist_summary["evidence_recorded_gate_ids"]
    assert checklist_summary["operator_evidence_gate_ids"] == ["browser_smoke"]
    assert checklist_summary["generated_contract_gate_ids"] == ["static_readiness"]
    assert checklist_summary["operator_evidence_pending_gate_ids"] == []
    assert checklist_summary["generated_contract_failed_gate_ids"] == []
    assert checklist_summary["gates"][1]["evidence_recorded"] is True
    assert checklist_summary["gates"][1]["gate_kind"] == "operator_evidence"
    assert checklist_summary["gates"][1]["operator_evidence_gate"] is True
    assert checklist_summary["gates"][1]["evidence_file_count"] == 1
    assert "## Validation Evidence: Browser smoke" in log_text
    assert "- Gate ID: browser_smoke" in log_text
    assert "- Status: passed" in log_text
    assert "- Operator: operator@example.org" in log_text
    assert str(updated_path) in log_text
    assert "Opened /work, /admin, task session, failure state, completion, and reopen." in log_text

    with pytest.raises(ValueError, match=r"missing_gate.*static_readiness\(passed\).*browser_smoke\(pending_operator_evidence\)"):
        labeling_work.main(
            [
                "--store",
                str(tmp_path / "labeling_work.sqlite"),
                "update-validation-checklist",
                "--path",
                str(checklist_path),
                "--gate",
                "missing_gate",
                "--status",
                "passed",
            ]
        )

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(tmp_path / "labeling_work.sqlite"),
                "update-validation-checklist",
                "--path",
                str(checklist_path),
                "--gate",
                "browser_smoke",
                "--status",
                "passed",
                "--output",
                str(updated_path),
            ]
        )


def test_apply_operator_evidence_templates_refreshes_handoff_sendability(tmp_path, capsys):
    package_dir = tmp_path / "handoff"
    package_dir.mkdir()
    checklist_path = package_dir / "validation-checklist.json"
    evidence_path = package_dir / "browser-smoke-evidence-template.json"
    manifest_path = package_dir / "manifest.json"
    work_path = package_dir / "work-summary.json"
    links_path = package_dir / "signed-links.jsonl"
    html_path = package_dir / "index.html"
    message_path = package_dir / "message.txt"
    quickstart_path = package_dir / "labeler-quickstart.txt"
    dataset_queue_path = package_dir / "dataset-queue.json"
    for path in (html_path, message_path, quickstart_path):
        path.write_text("Wait for operator review before starting.\n", encoding="utf-8")
    dataset_queue_path.write_text("{}\n", encoding="utf-8")
    checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "all_validation_complete": False,
                "ready_for_operator_validation": True,
                "browser_smoke_evidence_template": str(evidence_path),
                "gates": [
                    {"id": "static_readiness", "status": "passed", "required": True},
                    {"id": "browser_smoke", "status": "pending_operator_evidence", "required": True},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_browser_smoke_evidence_template.v1",
                "personalized_route_smoke_contract": _browser_smoke_personalized_route_contract(),
                "users": [
                    {
                        "expected_user": "alice",
                        "resolved_user": "alice",
                        "run_status": "operator_approved",
                        "identity_matches_expected_user": True,
                        "browser_only_runtime_verified": True,
                        "no_local_palette_install_verified": True,
                        "no_local_crimson_install_verified": True,
                        "no_local_conda_or_project_dependencies_verified": True,
                        "personalized_dataset_queue_verified": True,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": True,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": True,
                        "personalized_work_dashboard_verified": True,
                        "labeler_sees_only_assigned_work": True,
                        "support_text_redacted": True,
                        "expected_user_mismatch_rejected": True,
                        "task_opened": True,
                        "induced_failure_support_detail_redacted": True,
                        "completion_verified": True,
                        "completed_task_read_only_verified": True,
                        "stale_tab_save_rejected": True,
                        "operator_reopen_verified": True,
                        "operator": "ops",
                        "operator_approved_at_utc": "2026-06-23T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    work_path.write_text(
        json.dumps(
            {
                "progress_summary": {"waiting_recording_count": 1},
                "recordings": [
                    {
                        "recording_id": "rec-a",
                        "assignment_notes": "Review this recording.",
                        "tasks": [
                            {
                                "task_id": "task-a",
                                "workflow_kind": "keypoints",
                                "title": "Keypoints",
                                "state": "pending",
                                "priority": 1,
                            }
                        ],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    links_path.write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "url": "https://labeling.example.org/t/task-a",
                "ready_to_share": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    single_owner_policy = _assignment_ownership_policy()
    single_owner_assignment_contract = {
        "schema": "palette.web_labeling_single_owner_assignment_contract.v1",
        "ready": True,
        "single_owner_assignment_contract_met": True,
        "recording_id_primary_key": True,
        "schema_enforced_recording_primary_key": True,
        "primary_key_columns": ["recording_id"],
        "browser_mutation_target_resolved_server_side": True,
        "browser_mutation_target_source": "recording_assignments.active_assignment",
        "labelers_mutate_assigned_training_zarrs": True,
        "labelers_mutate_intermediate_csvs": False,
    }
    assignment_ownership_integrity = {
        "ok": True,
        "active_assignment_count": 1,
        "unique_active_recording_count": 1,
        "duplicate_active_owner_count": 0,
        "duplicate_active_owners": [],
        "recording_id_primary_key": True,
        "schema_enforced_recording_primary_key": True,
        "primary_key_columns": ["recording_id"],
        "schema_integrity_source": "fixture",
    }
    assignment_ownership_contract = (
        labeling_web_module._assignment_ownership_contract_policy(
            single_owner_policy,
            assignment_ownership_integrity,
            store_single_owner_contract=single_owner_assignment_contract,
        )
    )
    manifest = {
        "ok": True,
        "store_path": str(tmp_path / "labeling_work.sqlite"),
        "user": "alice",
        "include_completed": False,
        "base_url": "https://labeling.example.org",
        "labeler_landing_page_path": "/",
        "labeler_landing_url": "https://labeling.example.org",
        "expected_user_labeler_landing_url": "https://labeling.example.org?expected_user=alice",
        "dashboard_path": "/work",
        "dashboard_url": "https://labeling.example.org/work",
        "expected_user_dashboard_url": "https://labeling.example.org/work?expected_user=alice",
        "dataset_queue_page_path": "/datasets",
        "dataset_queue_url": "https://labeling.example.org/datasets",
        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
        "expected_user_identity_probe_url": "https://labeling.example.org/identity?expected_user=alice",
        "known_user_status": {"is_known_labeler": True, "active_assignment_count": 1, "assignment_count": 1},
        "assignment_snapshot": {"users": ["alice"]},
        "single_owner_policy": single_owner_policy,
        "single_owner_assignment_contract": single_owner_assignment_contract,
        "assignment_ownership_integrity": assignment_ownership_integrity,
        "assignment_ownership_contract": assignment_ownership_contract,
        **labeling_web_module._assignment_ownership_contract_fields(
            assignment_ownership_contract
        ),
        "single_owner_policy_contract_met": True,
        "files": {
            "html_index": str(html_path),
            "message": str(message_path),
            "quickstart": str(quickstart_path),
            "dataset_queue": str(dataset_queue_path),
            "manifest": str(manifest_path),
            "work_summary": str(work_path),
            "signed_links": str(links_path),
        },
        "labeler_safety": _labeler_safety_policy(),
        "labeler_route_authorization_policy": _labeler_route_authorization_policy(),
        "signed_link_policy": _browser_signed_link_policy(),
        "session_guard_policy": _session_guard_policy(),
        "task_state_policy": _browser_task_state_policy(),
        "zarr_backup_policy": _zarr_backup_policy(),
        "mutation_audit_policy": _mutation_audit_policy(),
        "browser_response_security_policy": _browser_response_security_policy(),
        "browser_mutation_write_policy": _browser_mutation_write_policy(),
        "operator_validation_required_before_invite": True,
        "operator_validation_all_complete": False,
        "operator_validation_status": "pending_operator_evidence",
        "operator_validation_pending_gate_ids": ["browser_smoke"],
        "operator_validation_needs_review_gate_ids": [],
        "operator_validation_operator_action": "Complete required operator validation evidence.",
        "counts": {"recordings": 1, "tasks": 1, "signed_links": 1, "ready_to_share_links": 1},
    }
    manifest["ready_to_send"] = _handoff_ready_to_send(manifest)
    manifest["sendability_reasons"] = _handoff_sendability_reasons(manifest)
    manifest["sendability_actions"] = _handoff_sendability_actions(manifest["sendability_reasons"])
    manifest["sendability_warnings"] = [{"user": "alice", "reasons": manifest["sendability_reasons"]}]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert manifest["ready_to_send"] is False

    rc = labeling_work.main(
        [
            "--store",
            str(tmp_path / "labeling_work.sqlite"),
            "apply-operator-evidence-templates",
            "--path",
            str(checklist_path),
            "--operator",
            "ops",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    refreshed_manifest = json.loads(manifest_path.read_text())
    refreshed_work = json.loads(work_path.read_text())
    refreshed_work_detail = refreshed_work.get("work") if isinstance(refreshed_work.get("work"), dict) else refreshed_work
    refreshed_dataset_queue = json.loads(dataset_queue_path.read_text())
    assert rc == 0
    assert payload["applied_count"] == 1
    assert payload["all_validation_complete"] is True
    assert payload["handoff_refresh_enabled"] is True
    assert payload["handoff_refresh_refreshed_manifest_count"] == 1
    assert payload["handoff_refresh_refreshed_file_count"] >= 5
    assert str(manifest_path) in payload["handoff_refresh_refreshed_files"]
    assert str(work_path) in payload["handoff_refresh_refreshed_files"]
    assert str(dataset_queue_path) in payload["handoff_refresh_refreshed_files"]
    assert payload["handoff_refresh_refreshed_visible_json_file_count"] == 3
    assert set(payload["handoff_refresh_refreshed_visible_json_files"]) == {
        str(manifest_path),
        str(work_path),
        str(dataset_queue_path),
    }
    assert payload["handoff_refresh"]["refreshed_visible_json_file_count"] == 3
    assert set(payload["handoff_refresh"]["refreshed_visible_json_files"]) == {
        str(manifest_path),
        str(work_path),
        str(dataset_queue_path),
    }
    assert payload["handoff_refresh_skipped_count"] == 0
    assert payload["handoff_refresh_skipped"] == []
    assert payload["handoff_refresh"]["skipped_count"] == 0
    assert payload["handoff_refresh"]["skipped"] == []
    assert payload["checksum_refresh_required"] is True
    assert "refresh-handoff-checksums" in payload["checksum_refresh_command"]
    assert refreshed_manifest["operator_validation_all_complete"] is True
    assert refreshed_manifest["operator_validation_status"] == "passed"
    for artifact in (
        refreshed_manifest,
        refreshed_work,
        refreshed_work_detail,
        refreshed_dataset_queue,
    ):
        for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
            assert artifact[f"operator_validation_gate_{gate_id}_status"] == "passed"
            assert artifact[f"operator_validation_gate_{gate_id}_pending"] is False
            assert artifact[f"operator_validation_gate_{gate_id}_missing_evidence"] is False
            assert artifact[f"operator_validation_gate_{gate_id}_needs_review"] is False
            assert artifact[f"operator_validation_gate_{gate_id}_passed"] is True
    assert refreshed_manifest["ready_to_send"] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"]["ready"] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "expected_user_matches_resolved_user"
    ] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "known_assignment_store_user"
    ] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "task_open_requires_active_assignment"
    ] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "mutation_requires_current_target_token"
    ] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "browser_mutation_target_resolved_server_side"
    ] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "labelers_mutate_assigned_training_zarrs"
    ] is True
    assert refreshed_manifest["labeler_route_authorization_checklist"][
        "labelers_mutate_intermediate_csvs"
    ] is False
    assert refreshed_work["labeler_route_authorization_checklist"] == refreshed_manifest[
        "labeler_route_authorization_checklist"
    ]
    assert refreshed_manifest["operator_validation_visibility_policy"][
        "operator_only_fields"
    ] == ["operator_validation_checklist_path"]
    assert refreshed_work["operator_validation_visibility_policy"] == refreshed_manifest[
        "operator_validation_visibility_policy"
    ]
    assert refreshed_work_detail["operator_validation_visibility_policy"] == refreshed_manifest[
        "operator_validation_visibility_policy"
    ]
    assert refreshed_manifest["operator_validation_command_templates"]["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert refreshed_manifest["operator_validation_command_templates"]["gate_ids"] == []
    assert refreshed_manifest["operator_validation_command_templates"]["command_count"] == 0
    assert refreshed_manifest["operator_validation_command_templates"]["command_ids"] == []
    assert refreshed_work["operator_validation_command_templates"] == refreshed_manifest[
        "operator_validation_command_templates"
    ]
    assert refreshed_work_detail["operator_validation_command_templates"] == refreshed_manifest[
        "operator_validation_command_templates"
    ]
    assert refreshed_work["browser_mutation_write_checklist"]["ready"] is True
    assert refreshed_work["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert refreshed_work["browser_mutation_write_checklist"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert refreshed_work["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert refreshed_work["dataset_queue_direct_start_policy"]["enabled"] is True
    assert refreshed_work["dataset_queue_direct_start_policy"][
        "browser_writes_csv_or_handoff_files"
    ] is False
    assert refreshed_work["dataset_queue_direct_start_policy"]["browser_writes_handoff_csv"] is False
    assert refreshed_work["dataset_queue_direct_start_policy"][
        "browser_writes_intermediate_csv"
    ] is False
    assert refreshed_work_detail["browser_mutation_write_checklist"] == refreshed_work[
        "browser_mutation_write_checklist"
    ]
    assert refreshed_work_detail["dataset_queue_direct_start_policy"] == refreshed_work[
        "dataset_queue_direct_start_policy"
    ]
    assert refreshed_work["runtime_operator_validation_gate_cli_policy"] == refreshed_manifest[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert refreshed_work_detail["runtime_operator_validation_gate_cli_policy"] == refreshed_work[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert refreshed_work["runtime_operator_validation_gate_cli_policy"][
        "preferred_require_flag"
    ] == "--require-operator-validation-for-browser-work"
    assert refreshed_work["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_mutations"
    ] is True
    assert refreshed_work["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_zarr_write"
    ] is True
    assert refreshed_work["ready_to_send"] is True
    assert refreshed_work["sendability_reasons"] == []
    assert refreshed_work_detail["dataset_queue_summary"]["open_task_count"] == 1
    assert refreshed_dataset_queue["schema"] == "palette.web_labeling_dataset_queue.v1"
    assert refreshed_dataset_queue["store_path"] == str(tmp_path / "labeling_work.sqlite")
    assert refreshed_dataset_queue["include_completed"] is False
    assert refreshed_dataset_queue["base_url"] == "https://labeling.example.org"
    assert refreshed_dataset_queue["labeler_landing_page_path"] == "/"
    assert refreshed_dataset_queue["dashboard_path"] == "/work"
    assert refreshed_dataset_queue["dataset_queue_page_path"] == "/datasets"
    assert refreshed_dataset_queue["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert refreshed_dataset_queue["expected_user"] == "alice"
    assert refreshed_dataset_queue["assignment_snapshot"] == {"users": ["alice"]}
    assert refreshed_dataset_queue["known_user_status"]["is_known_labeler"] is True
    assert refreshed_dataset_queue["operator_authorization_policy"]["admin_routes_require_operator"] is True
    assert refreshed_dataset_queue["operator_validation_visibility_policy"] == refreshed_manifest[
        "operator_validation_visibility_policy"
    ]
    assert refreshed_dataset_queue["operator_validation_command_templates"] == refreshed_manifest[
        "operator_validation_command_templates"
    ]
    assert refreshed_dataset_queue["ready_to_send"] is True
    assert refreshed_dataset_queue["labeler_route_authorization_checklist"] == refreshed_manifest[
        "labeler_route_authorization_checklist"
    ]
    assert refreshed_dataset_queue["dataset_queue_summary"]["open_task_count"] == 1
    assert refreshed_dataset_queue["dataset_queue"][0]["recordings"][0]["tasks"][0]["task_id"] == "task-a"
    assert refreshed_manifest["sendability_reasons"] == []
    assert "Your Palette labeling work is ready." in message_path.read_text()
    assert "This handoff is ready to use." in quickstart_path.read_text()
    assert "record-browser-smoke-evidence --evidence" not in message_path.read_text()
    assert "apply-operator-evidence-templates --path" not in message_path.read_text()
    assert "record-zarr-backup-evidence --evidence" not in message_path.read_text()
    assert "record-browser-smoke-evidence --evidence" not in quickstart_path.read_text()
    assert "apply-operator-evidence-templates --path" not in quickstart_path.read_text()
    assert "record-zarr-backup-evidence --evidence" not in quickstart_path.read_text()
    assert "Open your personalized dataset queue" in html_path.read_text()


def test_import_batch_plan_cli_dry_run_apply_and_missing_assignment_check(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    assignments_path = tmp_path / "assignments.csv"
    tasks_path = tmp_path / "tasks.csv"
    inactive_assignments_path = tmp_path / "inactive-assignments.csv"
    inactive_tasks_path = tmp_path / "inactive-tasks.csv"
    duplicate_assignments_path = tmp_path / "duplicate-assignments.csv"
    reassignment_assignments_path = tmp_path / "reassignment-assignments.csv"
    multi_workflow_tasks_path = tmp_path / "multi-workflow-tasks.csv"
    duplicate_scope_tasks_path = tmp_path / "duplicate-scope-tasks.csv"
    missing_tasks_path = tmp_path / "missing-tasks.csv"
    report_path = tmp_path / "batch-plan-report.json"
    html_report_path = tmp_path / "batch-plan-report.html"
    duplicate_apply_html_report_path = tmp_path / "batch-plan-duplicate-apply.html"
    assignments_path.write_text(
        "recording_id,assignee_user,status,notes\n"
        "rec-a,alice,active,keypoints first\n",
        encoding="utf-8",
    )
    tasks_path.write_text(
        "task_id,recording_id,workflow_kind,title,scope_json,priority\n"
        'task-a,rec-a,keypoints,Review keypoints,"{""frames"":[1,2]}",4\n',
        encoding="utf-8",
    )
    missing_tasks_path.write_text(
        "task_id,recording_id,workflow_kind,title\n"
        "task-missing,rec-missing,keypoints,Missing assignment\n",
        encoding="utf-8",
    )
    inactive_assignments_path.write_text(
        "recording_id,assignee_user,status,notes\n"
        "rec-inactive,alice,paused,not ready\n",
        encoding="utf-8",
    )
    inactive_tasks_path.write_text(
        "task_id,recording_id,workflow_kind,title\n"
        "task-inactive,rec-inactive,keypoints,Inactive assignment\n",
        encoding="utf-8",
    )
    duplicate_assignments_path.write_text(
        "recording_id,assignee_user,status,notes\n"
        "rec-a,alice,active,first owner\n"
        "rec-a,bob,active,second owner\n",
        encoding="utf-8",
    )
    reassignment_assignments_path.write_text(
        "recording_id,assignee_user,status,notes\n"
        "rec-a,bob,active,new owner\n",
        encoding="utf-8",
    )
    multi_workflow_tasks_path.write_text(
        "task_id,recording_id,workflow_kind,title\n"
        "task-kp,rec-a,keypoints,Keypoints\n"
        "task-box,rec-a,detect_analysis,Boxes\n",
        encoding="utf-8",
    )
    duplicate_scope_tasks_path.write_text(
        "task_id,recording_id,workflow_kind,title,scope_json\n"
        'task-a1,rec-a,keypoints,First,"{""frames"":[1,2]}"\n'
        'task-a2,rec-a,keypoints,Second,"{""frames"":[1,2]}"\n',
        encoding="utf-8",
    )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(assignments_path),
            "--tasks",
            str(tasks_path),
            "--assigned-by",
            "operator",
            "--actor",
            "operator",
            "--output",
            str(report_path),
            "--html-output",
            str(html_report_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    archived_payload = json.loads(report_path.read_text())
    html_report = html_report_path.read_text()
    assert rc == 0
    assert payload["dry_run"] is True
    assert archived_payload["assignment_count"] == payload["assignment_count"]
    assert "Palette batch plan report" in html_report
    assert "Assignment changes" in html_report
    assert "Task changes" in html_report
    assert "Issue codes" in html_report
    assert "Warning codes" in html_report
    assert "Blocking warning codes" in html_report
    assert "Closed sessions" in html_report
    assert "task-a" in html_report
    assert "rec-a" in html_report
    assert payload["assignment_count"] == 1
    assert payload["task_count"] == 1
    assert payload["assignments"][0]["source_line"] == 2
    assert payload["tasks"][0]["source_line"] == 2
    assert payload["assignment_ownership_integrity"]["primary_key_columns"] == ["recording_id"]
    assert payload["assignment_ownership_contract"]["one_active_owner"] is True
    assert payload["assignment_manifest_artifact_role"] == "metadata_only_control_plane"
    assert payload["assignment_manifest_artifacts_are_label_write_targets"] is False
    assert payload["assignment_manifest_browser_writes_label_data"] is False
    assert payload["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert payload["browser_mutation_write_policy"]["csv_handoff_artifacts_are_label_write_targets"] is False
    assert payload["browser_mutation_write_checklist"]["browser_writes_csv_or_handoff_files"] is False
    assert payload["browser_mutation_write_checklist"]["browser_has_direct_zarr_write_authority"] is False

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(inactive_assignments_path),
            "--tasks",
            str(inactive_tasks_path),
        ]
    )

    warning_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert warning_payload["ok"] is True
    assert warning_payload["warning_count"] == 1
    assert warning_payload["warning_codes"] == ["task_recording_assignment_not_active_after_plan"]
    assert warning_payload["blocking_warning_count"] == 0
    assert warning_payload["blocking_warning_codes"] == []
    assert warning_payload["warnings"][0]["code"] == "task_recording_assignment_not_active_after_plan"
    assert warning_payload["warnings"][0]["source_line"] == 2

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(inactive_assignments_path),
            "--tasks",
            str(inactive_tasks_path),
            "--warnings-as-errors",
            "--apply",
        ]
    )

    warning_error_payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert warning_error_payload["ok"] is False
    assert warning_error_payload["warnings_as_errors"] is True
    assert warning_error_payload["warning_count"] == 1
    assert warning_error_payload["warning_codes"] == ["task_recording_assignment_not_active_after_plan"]
    assert warning_error_payload["blocking_warning_count"] == 1
    assert warning_error_payload["blocking_warning_codes"] == ["task_recording_assignment_not_active_after_plan"]
    store = LabelingStore(store_path)
    try:
        assert store.get_assignment("rec-inactive") is None
        assert store.get_task("task-inactive") is None
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(assignments_path),
            "--tasks",
            str(multi_workflow_tasks_path),
        ]
    )

    workflow_warning_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert workflow_warning_payload["warning_count"] == 1
    assert workflow_warning_payload["warnings"][0]["code"] == "recording_has_multiple_workflow_kinds"
    assert workflow_warning_payload["warnings"][0]["workflow_kinds"] == ["detect_analysis", "keypoints"]
    assert workflow_warning_payload["warnings"][0]["source_lines"] == [2, 3]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(assignments_path),
            "--tasks",
            str(duplicate_scope_tasks_path),
        ]
    )

    duplicate_warning_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert duplicate_warning_payload["warning_count"] == 1
    assert duplicate_warning_payload["warnings"][0]["code"] == "duplicate_logical_task_scope"
    assert duplicate_warning_payload["warnings"][0]["task_ids"] == ["task-a1", "task-a2"]
    assert duplicate_warning_payload["warnings"][0]["source_lines"] == [2, 3]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(duplicate_assignments_path),
            "--tasks",
            str(tasks_path),
        ]
    )

    duplicate_assignment_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert duplicate_assignment_payload["single_owner_policy"]["one_active_owner"] is True
    assert duplicate_assignment_payload["warning_count"] == 1
    assert duplicate_assignment_payload["warning_codes"] == ["duplicate_recording_assignment_rows"]
    assert duplicate_assignment_payload["warnings"][0]["code"] == "duplicate_recording_assignment_rows"
    assert duplicate_assignment_payload["warnings"][0]["recording_id"] == "rec-a"
    assert duplicate_assignment_payload["warnings"][0]["assignee_users"] == ["alice", "bob"]
    assert duplicate_assignment_payload["warnings"][0]["source_lines"] == [2, 3]

    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(reassignment_assignments_path),
            "--tasks",
            str(tasks_path),
        ]
    )

    reassignment_payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert reassignment_payload["single_owner_policy"]["reassignment_replaces_owner"] is True
    assert reassignment_payload["warning_count"] == 1
    assert reassignment_payload["warning_codes"] == ["assignment_reassigns_existing_recording"]
    assert reassignment_payload["warnings"][0]["code"] == "assignment_reassigns_existing_recording"
    assert reassignment_payload["warnings"][0]["recording_id"] == "rec-a"
    assert reassignment_payload["warnings"][0]["previous_assignee_user"] == "alice"
    assert reassignment_payload["warnings"][0]["new_assignee_user"] == "bob"
    assert reassignment_payload["warnings"][0]["source_line"] == 2

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "import-batch-plan",
                "--assignments",
                str(assignments_path),
                "--tasks",
                str(tasks_path),
                "--output",
                str(report_path),
                "--html-output",
                str(html_report_path),
            ]
        )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(assignments_path),
            "--tasks",
            str(tasks_path),
            "--assigned-by",
            "operator",
            "--actor",
            "operator",
            "--apply",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["dry_run"] is False
    assert payload["applied_assignment_count"] == 1
    assert payload["applied_task_count"] == 1

    store = LabelingStore(store_path)
    try:
        assignment = store.get_assignment("rec-a")
        task = store.get_task("task-a")
        assert assignment is not None
        assert assignment["assignee_user"] == "alice"
        assert task is not None
        assert task["scope"] == {"frames": [1, 2]}
        assert task["priority"] == 4
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(assignments_path),
            "--tasks",
            str(missing_tasks_path),
            "--apply",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["ok"] is False
    assert payload["issue_codes"] == ["task_recording_missing_assignment_after_plan"]
    assert payload["issues"][0]["code"] == "task_recording_missing_assignment_after_plan"
    assert payload["issues"][0]["source_line"] == 2
    store = LabelingStore(store_path)
    try:
        assert store.get_task("task-missing") is None
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "import-batch-plan",
            "--assignments",
            str(duplicate_assignments_path),
            "--tasks",
            str(tasks_path),
            "--apply",
            "--html-output",
            str(duplicate_apply_html_report_path),
        ]
    )

    duplicate_apply_payload = json.loads(capsys.readouterr().out)
    duplicate_apply_html = duplicate_apply_html_report_path.read_text()
    assert rc == 0
    assert duplicate_apply_payload["assignment_input_row_count"] == 2
    assert duplicate_apply_payload["assignment_count"] == 2
    assert duplicate_apply_payload["applied_assignment_count"] == 1
    assert duplicate_apply_payload["deduplicated_assignment_apply_count"] == 1
    assert duplicate_apply_payload["skipped_duplicate_assignment_apply_count"] == 1
    assert duplicate_apply_payload["assignments"][0]["skipped_by_duplicate_apply"] is True
    assert duplicate_apply_payload["assignments"][0]["applied"] is False
    assert duplicate_apply_payload["assignments"][0]["warnings"][0]["code"] == "duplicate_assignment_row_skipped_for_apply"
    assert duplicate_apply_payload["assignments"][1]["skipped_by_duplicate_apply"] is False
    assert duplicate_apply_payload["assignments"][1]["applied"] is True
    assert duplicate_apply_payload["assignments"][1]["assignment"]["assignee_user"] == "bob"
    assert duplicate_apply_payload["assignment_ownership_integrity"]["active_assignment_count"] == 1
    assert duplicate_apply_payload["assignment_ownership_integrity"]["unique_active_recording_count"] == 1
    assert duplicate_apply_payload["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    assert duplicate_apply_payload["assignment_ownership_contract"][
        "duplicate_manifest_rows_do_not_create_multiple_owners"
    ] is True
    assert duplicate_apply_payload["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert duplicate_apply_payload["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert "Skipped duplicate assignment rows" in duplicate_apply_html
    assert "Skipped duplicate" in duplicate_apply_html
    assert "duplicate_assignment_row_skipped_for_apply" in duplicate_apply_html
    store = LabelingStore(store_path)
    try:
        assert store.get_assignment("rec-a")["assignee_user"] == "bob"
    finally:
        store.close()


def test_import_tasks_cli_rejects_duplicate_task_ids(tmp_path):
    manifest_path = tmp_path / "tasks.jsonl"
    manifest_path.write_text(
        '\n'.join(
            [
                json.dumps({"task_id": "task-a", "recording_id": "rec-a", "workflow_kind": "keypoints"}),
                json.dumps({"task_id": "task-a", "recording_id": "rec-b", "workflow_kind": "detect_analysis"}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate task_id"):
        labeling_work.main(["--store", str(tmp_path / "labeling_work.sqlite"), "import-tasks", "--input", str(manifest_path)])


def test_task_definition_events_record_create_change_and_skip_unchanged(tmp_path):
    store = _store(tmp_path)
    try:
        first = store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Initial title",
            scope={"frames": [1, 2]},
            actor_user="operator",
        )
        unchanged = store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Initial title",
            scope={"frames": [1, 2]},
            actor_user="operator",
        )
        changed = store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Updated title",
            scope={"frames": [1, 2, 3]},
            priority=7,
            actor_user="operator",
        )
        events = store.list_task_definition_events(task_id="task-a", actor_user="operator", limit=10)

        assert unchanged["updated_at_utc"] == first["updated_at_utc"]
        assert changed["title"] == "Updated title"
        assert changed["priority"] == 7
        assert [event["event_type"] for event in events] == ["task_definition_changed", "task_definition_created"]
        assert events[0]["before"]["title"] == "Initial title"
        assert events[0]["after"]["title"] == "Updated title"
        assert events[0]["after"]["scope"] == {"frames": [1, 2, 3]}
        assert events[1]["before"] is None
    finally:
        store.close()


def test_export_task_definition_events_cli_writes_jsonl_archive(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "task-definition-events.jsonl"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Initial",
            actor_user="operator",
        )
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Updated",
            actor_user="operator",
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-task-definition-events",
            "--task-id",
            "task-a",
            "--actor",
            "operator",
            "--format",
            "jsonl",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rc == 0
    assert summary["ok"] is True
    assert summary["count"] == 2
    assert [row["event_type"] for row in rows] == ["task_definition_changed", "task_definition_created"]

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-task-definition-events",
                "--task-id",
                "task-a",
                "--output",
                str(output_path),
            ]
        )


def test_check_store_cli_reports_clean_store(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(["--store", str(store_path), "check-store"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["issue_count"] == 0
    assert payload["counts"]["assignments"] == 1
    assert payload["counts"]["tasks"] == 1
    assert payload["single_owner_policy"]["recording_id_primary_key"] is True
    assert payload["assignment_ownership_integrity"]["schema_integrity_source"] == "store_pragma"
    assert payload["assignment_ownership_integrity"]["recording_id_primary_key"] is True
    assert payload["assignment_ownership_integrity"]["schema_enforced_recording_primary_key"] is True
    assert payload["assignment_ownership_integrity"]["primary_key_columns"] == ["recording_id"]
    assert payload["assignment_ownership_integrity"]["one_row_per_recording_enforced"] is True
    assert payload["assignment_ownership_integrity"]["active_assignment_count"] == 1
    assert payload["assignment_ownership_integrity"]["unique_active_recording_count"] == 1
    assert payload["assignment_ownership_integrity"]["duplicate_active_owners"] == []
    assert payload["reassignment_session_safety"]["ok"] is True
    assert payload["reassignment_session_safety"]["active_session_assignment_mismatch_count"] == 0
    assert payload["reassignment_session_safety"]["blocks_labeler_mutation"] is False


def test_check_store_cli_reports_assignment_safety_issues(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-unassigned", recording_id="rec-missing", workflow_kind="detect_analysis")
        store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(recording_id="rec-a", assignee_user="bob", allow_stale_open_sessions=True)
    finally:
        store.close()

    rc = labeling_work.main(["--store", str(store_path), "check-store"])

    payload = json.loads(capsys.readouterr().out)
    codes = {issue["code"] for issue in payload["issues"]}
    issues_by_code = {issue["code"]: issue for issue in payload["issues"]}
    assert rc == 2
    assert payload["ok"] is False
    assert "task_missing_assignment" in codes
    assert "active_session_assignment_mismatch" in codes
    assert payload["issue_counts"]["task_missing_assignment"] == 1
    assert payload["issue_counts"]["active_session_assignment_mismatch"] == 1
    assert payload["reassignment_session_safety"]["ok"] is False
    assert payload["reassignment_session_safety"]["active_session_assignment_mismatch_count"] == 1
    assert payload["reassignment_session_safety"]["active_session_assignment_mismatch_recording_ids"] == ["rec-a"]
    assert payload["reassignment_session_safety"]["blocks_labeler_mutation"] is True
    assert payload["reassignment_session_safety"]["requires_operator_recovery"] is True
    assert "assign_recording_with_session_closure" in payload["reassignment_session_safety"]["operator_action"]
    assert payload["warning_counts"] == {}
    assert all(issue["blocks_labeler_mutation"] is True for issue in payload["issues"])
    assert "assign --recording-id rec-missing --user USER --assigned-by OPERATOR" in issues_by_code[
        "task_missing_assignment"
    ]["operator_action"]
    assert "assign --recording-id rec-a --user bob --assigned-by OPERATOR" in issues_by_code[
        "active_session_assignment_mismatch"
    ]["operator_action"]
    assert payload["operator_actions"] == [
        issues_by_code["task_missing_assignment"]["operator_action"],
        issues_by_code["active_session_assignment_mismatch"]["operator_action"],
    ]


def test_export_audit_bundle_cli_writes_all_event_families(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "audit-bundle"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", assigned_by="operator")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Bundle task",
            actor_user="operator",
        )
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"frame_idx": 1},
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-audit-bundle",
            "--user",
            "alice",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    task_events = [json.loads(line) for line in (output_dir / "task-events.jsonl").read_text().splitlines()]
    assignment_events = [json.loads(line) for line in (output_dir / "assignment-events.jsonl").read_text().splitlines()]
    task_definition_events = [json.loads(line) for line in (output_dir / "task-definition-events.jsonl").read_text().splitlines()]

    assert rc == 0
    assert payload["ok"] is True
    assert manifest["counts"] == {
        "task_events": 1,
        "assignment_events": 1,
        "task_definition_events": 1,
    }
    assert task_events[0]["event_type"] == "save_keypoints"
    assert assignment_events[0]["event_type"] == "assignment_created"
    assert task_definition_events[0]["event_type"] == "task_definition_created"

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-audit-bundle",
                "--user",
                "alice",
                "--output-dir",
                str(output_dir),
            ]
        )


def test_work_summary_cli_exports_personalized_dashboard_payload(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "alice-work-summary.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.assign_recording(recording_id="rec-done", assignee_user="alice", notes="Finished recording")
        store.assign_recording(recording_id="rec-empty", assignee_user="alice", notes="Waiting for task generation")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints", title="Alice open")
        store.upsert_task(task_id="task-a-complete", recording_id="rec-a", workflow_kind="keypoints", state="complete")
        store.upsert_task(task_id="task-done", recording_id="rec-done", workflow_kind="keypoints", state="complete")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis", title="Bob open")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "work-summary",
            "--user",
            "alice",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(output_path.read_text())
    work = payload["work"]
    assert rc == 0
    assert summary["ok"] is True
    assert summary["recording_count"] == 3
    assert summary["task_count"] == 1
    assert summary["known_user"] is True
    assert summary["labeler_landing_page_path"] == "/"
    assert summary["dashboard_path"] == "/work"
    assert summary["dataset_queue_page_path"] == "/datasets"
    assert summary["personal_work_page_path"] == "/my-work"
    assert summary["personal_dataset_queue_page_path"] == "/my-datasets"
    assert summary["expected_user_labeler_landing_url"] == "/?expected_user=alice"
    assert summary["expected_user_dashboard_url"] == "/work?expected_user=alice"
    assert summary["expected_user_dataset_queue_url"] == "/datasets?expected_user=alice"
    assert summary["expected_user_personal_work_url"] == "/my-work?expected_user=alice"
    assert summary["expected_user_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=alice"
    )
    assert summary["expected_user_identity_probe_url"] == "/identity?expected_user=alice"
    assert summary["preferred_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert summary["preferred_labeler_entry_url"] == "/my-datasets?expected_user=alice"
    assert summary["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert summary["personalized_labeler_entry_url"] == "/my-datasets?expected_user=alice"
    assert payload["personalized_labeler_entry_url"] == summary["personalized_labeler_entry_url"]
    assert work["personalized_labeler_entry_url"] == summary["personalized_labeler_entry_url"]
    assert payload["personalized_launch_readiness"]["schema"] == (
        "palette.web_labeling_personalized_launch_readiness.v1"
    )
    assert payload["personalized_launch_readiness"][
        "personalized_labeler_entry_url"
    ] == "/my-datasets?expected_user=alice"
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
        "external_launch_evidence_gap_count"
    ] == payload["safe_share_external_launch_evidence_gap_count"]
    assert work["personalized_launch_readiness"][
        "personalized_labeler_entry_url"
    ] == payload["personalized_launch_readiness"]["personalized_labeler_entry_url"]
    assert work["personalized_launch_readiness"][
        "external_launch_evidence_gap_count"
    ] == payload["personalized_launch_readiness"]["external_launch_evidence_gap_count"]
    assert payload["queue_first_entry_contract"]["ready"] is True
    assert payload["queue_first_entry_contract"]["personalized_labeler_entry_url"] == (
        "/my-datasets?expected_user=alice"
    )
    assert work["queue_first_entry_contract"] == payload["queue_first_entry_contract"]
    assert summary["active_assignment_count"] == 3
    assert summary["assignment_ownership_ok"] is True
    assert summary["assignment_schema_enforced_recording_primary_key"] is True
    assert summary["assignment_duplicate_active_owner_count"] == 0
    assert payload["single_owner_policy"]["assignment_scope"] == "recording"
    assert payload["single_owner_policy"]["recording_assignment_key"] == "recording_id"
    assert payload["single_owner_policy"]["one_current_assignment_row_per_recording"] is True
    assert payload["single_owner_policy"]["multiple_labelers_per_recording_allowed"] is False
    assert payload["single_owner_policy"]["assignment_manifests_are_control_plane"] is True
    assert payload["single_owner_policy"][
        "duplicate_manifest_rows_do_not_create_multiple_owners"
    ] is True
    assert payload["single_owner_policy"][
        "browser_mutation_requires_current_assignment_owner"
    ] is True
    assert payload["single_owner_policy"]["browser_mutation_target_resolved_server_side"] is True
    assert payload["single_owner_policy"]["browser_mutation_target_source"] == (
        "recording_assignments.active_assignment"
    )
    assert payload["single_owner_policy"]["labelers_mutate_assigned_training_zarrs"] is True
    assert payload["single_owner_policy"]["labelers_mutate_intermediate_csvs"] is False
    single_owner_assignment_contract = payload["single_owner_assignment_contract"]
    assert single_owner_assignment_contract["schema"] == (
        "palette.web_labeling_assignment_single_owner_contract.v1"
    )
    assert single_owner_assignment_contract["assignment_scope"] == "recording"
    assert single_owner_assignment_contract["recording_assignment_key"] == "recording_id"
    assert single_owner_assignment_contract["one_active_owner"] is True
    assert single_owner_assignment_contract["multiple_labelers_per_recording_allowed"] is False
    assert single_owner_assignment_contract["browser_mutation_requires_current_assignment_owner"] is True
    assert single_owner_assignment_contract["browser_mutation_target_resolved_server_side"] is True
    assert single_owner_assignment_contract["browser_mutation_target_source"] == (
        "recording_assignments.active_assignment"
    )
    assert single_owner_assignment_contract["labelers_mutate_assigned_training_zarrs"] is True
    assert single_owner_assignment_contract["labelers_mutate_intermediate_csvs"] is False
    assert single_owner_assignment_contract["ready"] is True
    assignment_ownership_contract = payload["assignment_ownership_contract"]
    assert assignment_ownership_contract[
        "store_single_owner_assignment_contract_present"
    ] is True
    assert assignment_ownership_contract["store_single_owner_assignment_contract_ready"] is True
    assert assignment_ownership_contract["store_single_owner_assignment_contract_met"] is True
    assert assignment_ownership_contract["store_single_owner_assignment_contract_schema"] == (
        "palette.web_labeling_assignment_single_owner_contract.v1"
    )
    assert assignment_ownership_contract["store_single_owner_assignment_contract"] == (
        single_owner_assignment_contract
    )
    assert assignment_ownership_contract["browser_mutation_target_resolved_server_side"] is True
    assert assignment_ownership_contract["browser_mutation_target_source"] == (
        "recording_assignments.active_assignment"
    )
    assert assignment_ownership_contract["labelers_mutate_assigned_training_zarrs"] is True
    assert assignment_ownership_contract["labelers_mutate_intermediate_csvs"] is False
    assert work["single_owner_policy_assignment_scope"] == "recording"
    assert work["single_owner_policy_recording_assignment_key"] == "recording_id"
    assert work["single_owner_policy_multiple_labelers_per_recording_allowed"] is False
    assert work[
        "single_owner_policy_browser_mutation_requires_current_assignment_owner"
    ] is True
    assert work["single_owner_policy_browser_mutation_target_resolved_server_side"] is True
    assert work["single_owner_policy_browser_mutation_target_source"] == (
        "recording_assignments.active_assignment"
    )
    assert work["single_owner_policy_labelers_mutate_assigned_training_zarrs"] is True
    assert work["single_owner_policy_labelers_mutate_intermediate_csvs"] is False
    assert summary["single_owner_policy_assignment_scope"] == "recording"
    assert summary["single_owner_policy_recording_assignment_key"] == "recording_id"
    assert summary["single_owner_policy_one_current_assignment_row_per_recording"] is True
    assert summary["single_owner_policy_one_active_owner"] is True
    assert summary["single_owner_policy_multiple_labelers_per_recording_allowed"] is False
    assert summary["single_owner_policy_assignment_manifests_are_control_plane"] is True
    assert summary[
        "single_owner_policy_duplicate_manifest_rows_do_not_create_multiple_owners"
    ] is True
    assert summary[
        "single_owner_policy_browser_mutation_requires_current_assignment_owner"
    ] is True
    assert summary["single_owner_policy_browser_mutation_target_resolved_server_side"] is True
    assert summary["single_owner_policy_browser_mutation_target_source"] == (
        "recording_assignments.active_assignment"
    )
    assert summary["single_owner_policy_labelers_mutate_assigned_training_zarrs"] is True
    assert summary["single_owner_policy_labelers_mutate_intermediate_csvs"] is False
    assert summary[
        "assignment_ownership_contract_store_single_owner_assignment_contract_present"
    ] is True
    assert summary[
        "assignment_ownership_contract_store_single_owner_assignment_contract_ready"
    ] is True
    assert summary[
        "assignment_ownership_contract_store_single_owner_assignment_contract_met"
    ] is True
    assert summary[
        "assignment_ownership_contract_store_single_owner_assignment_contract_schema"
    ] == "palette.web_labeling_assignment_single_owner_contract.v1"
    assert summary["store_consistency_ok"] is True
    assert summary["reassignment_session_safety_ok"] is True
    assert summary["reassignment_session_safety_mismatch_count"] == 0
    assert summary["reassignment_session_safety_blocks_labeler_mutation"] is False
    assert summary["store_consistency_issue_count"] == 0
    assert summary["store_consistency_issue_codes"] == []
    assert summary["store_consistency_warning_count"] == 0
    assert summary["store_consistency_warning_codes"] == []
    assert summary["store_consistency_blocking_warning_count"] == 0
    assert summary["store_consistency_blocking_warning_codes"] == []
    assert summary["labeler_start_ready"] is True
    assert summary["labeler_start_status"] == "has_open_dataset_work"
    assert summary["labeler_action"] == "open_dataset_queue"
    assert summary["labeler_start_message"] == "Open a dataset, recording, or task from this queue."
    assert summary["labeler_start_operator_action"] == ""
    assert summary["waiting_dataset_count"] == 1
    assert summary["dataset_open_task_count"] == 1
    assert summary["recordings_without_open_tasks"] == 2
    assert summary["recordings_without_open_tasks_by_reason"] == {
        "all_tasks_complete": 1,
        "tasks_not_generated": 1,
    }
    assert "Reopen a completed task only if more labeling work is required" in " ".join(
        summary["recordings_without_open_tasks_actions"]
    )
    assert "Generate or import browser-labeling tasks" in " ".join(
        summary["recordings_without_open_tasks_actions"]
    )
    assert summary["labeler_route_authorization_ready"] is True
    assert summary["labeler_route_authorization_active_assignment_required"] is True
    assert summary["labeler_route_authorization_active_assignment_count"] == 3
    assert summary["labeler_route_authorization_has_active_assignment"] is True
    assert summary["single_owner_policy_contract_met"] is True
    assert summary["operator_admin_routes_require_operator"] is True
    assert summary["operator_boundary_ready"] == payload["operator_authorization_policy"][
        "operator_boundary_ready"
    ]
    assert summary["operator_recovery_task_reopen_operator_only"] is True
    assert summary[
        "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update"
    ] is True
    assert summary[
        "operator_recovery_reassignment_target_validated_before_session_closure"
    ] is True
    assert summary["operator_recovery_session_closure_and_assignment_update_atomic"] is True
    assert summary["operator_recovery_failed_promotion_retry_operator_only"] is True
    assert summary["operator_validation_all_complete"] is False
    assert summary["operator_validation_declared_all_complete"] is False
    assert summary["operator_validation_status"] == "pending_operator_evidence"
    assert summary["operator_validation_gate_count"] == 6
    assert summary["operator_validation_pending_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert summary["operator_validation_required_missing_evidence_gate_ids"] == summary[
        "operator_validation_pending_gate_ids"
    ]
    assert summary["operator_validation_required_pending_gate_count"] == 6
    assert summary["operator_validation_needs_review_gate_count"] == 0
    assert summary["operator_validation_required_missing_evidence_gate_count"] == 6
    assert summary["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert summary["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert summary["operator_validation_gate_flat_field_suffixes"] == [
        "status",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]
    for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
        assert summary[f"operator_validation_gate_{gate_id}_status"] == "missing_evidence"
        assert summary[f"operator_validation_gate_{gate_id}_pending"] is True
        assert summary[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
        assert summary[f"operator_validation_gate_{gate_id}_needs_review"] is False
        assert summary[f"operator_validation_gate_{gate_id}_passed"] is False
    assert summary["operator_validation_source"] == "none"
    assert "mutable_zarr_backup_confirmation" in summary["operator_validation_operator_action"]
    assert "disposable_zarr_mutation_smoke" in summary["operator_validation_operator_action"]
    assert summary["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert summary["safe_share_checklist_gate_evidence_complete"] is False
    assert summary["safe_share_launch_blocking_next_action_count"] == 6
    assert "operator_validation_record_command_ids" in summary[
        "safe_share_launch_blocking_next_action_detail_fields"
    ]
    assert "operator_validation_evidence_template_path" in summary[
        "safe_share_launch_blocking_next_action_command_fields"
    ]
    assert summary["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke=missing_evidence" in summary["safe_share_next_action_summary"]
    assert any(
        action["gate_id"] == "browser_smoke" and action["blocks_share"] is True
        for action in summary["safe_share_launch_blocking_next_actions"]
    )
    browser_smoke_action = next(
        action
        for action in summary["safe_share_launch_blocking_next_actions"]
        if action["gate_id"] == "browser_smoke"
    )
    assert browser_smoke_action["operator_validation_record_command_ids"] == [
        "record_browser_smoke_evidence"
    ]
    assert browser_smoke_action["operator_validation_apply_command_id"] == (
        "apply_operator_evidence_templates"
    )
    assert browser_smoke_action["operator_validation_evidence_template_field"] == (
        "browser_smoke_evidence_template"
    )
    assert "browser_smoke" in summary[
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert payload["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
    assert payload["safe_share_checklist_gate_evidence_complete"] is False
    assert payload["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke=missing_evidence" in payload["safe_share_next_action_summary"]
    assert "disposable_zarr_mutation_smoke" in payload[
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert work["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert work["safe_share_checklist_gate_evidence_complete"] is False
    assert "identity_probe_verification" in work[
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert summary["operator_validation_command_template_command_count"] == 7
    assert "record_browser_smoke_evidence" in summary[
        "operator_validation_command_template_command_ids"
    ]
    assert summary[
        "operator_validation_command_template_launch_evidence_collection_plan_schema"
    ] == "palette.web_labeling_launch_evidence_collection_plan.v1"
    assert (
        summary[
            "operator_validation_command_template_launch_evidence_collection_step_count"
        ]
        == 6
    )
    assert "browser_smoke" in summary[
        "operator_validation_command_template_launch_evidence_collection_gate_ids"
    ]
    assert "record_browser_smoke_evidence" in summary[
        "operator_validation_command_template_launch_evidence_collection_record_command_ids"
    ]
    assert (
        summary[
            "operator_validation_command_template_launch_evidence_collection_required_final_field"
        ]
        == "labeler_links_safe_to_share"
    )
    assert summary["operator_validation_command_template_commands_are_operator_only"] is True
    assert (
        summary["operator_validation_command_template_commands_are_labeler_instructions"]
        is False
    )
    assert (
        summary["operator_validation_command_template_labelers_must_not_run_commands"]
        is True
    )
    assert summary["operator_validation_operator_only_fields"] == [
        "operator_validation_checklist_path"
    ]
    assert payload["operator_validation_command_templates"] == work[
        "operator_validation_command_templates"
    ]
    assert payload["operator_validation_command_templates"]["commands_are_operator_only"] is True
    assert payload["operator_validation_command_templates"][
        "commands_are_labeler_instructions"
    ] is False
    assert "record_browser_smoke_evidence" in payload[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert payload["operator_validation_visibility_policy"]["operator_action_fields"] == [
        "operator_validation_command_templates"
    ]
    assert (
        summary["operator_validation_labeler_visible_payloads_include_operator_only_fields"]
        is False
    )
    assert summary["operator_validation_per_user_payloads_use_public_fields_only"] is True
    assert (
        summary[
            "operator_validation_top_level_operator_reports_may_include_operator_only_fields"
        ]
        is True
    )
    assert payload["runtime_operator_validation_gate_cli_policy"] == work[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "preferred_require_flag"
    ] == "--require-operator-validation-for-browser-work"
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_start_open"
    ] is True
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_mutations"
    ] is True
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_target_token_check"
    ] is True
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_zarr_write"
    ] is True
    assert summary["runtime_operator_validation_gate_cli_policy"] == payload[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert summary[
        "runtime_operator_validation_gate_cli_policy_preferred_require_flag"
    ] == "--require-operator-validation-for-browser-work"
    assert summary[
        "runtime_operator_validation_gate_cli_policy_protects_browser_start_open"
    ] is True
    assert summary[
        "runtime_operator_validation_gate_cli_policy_protects_browser_mutations"
    ] is True
    assert summary[
        "runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check"
    ] is True
    assert summary[
        "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write"
    ] is True
    assert summary[
        "runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation"
    ] is True
    assert summary["browser_mutation_write_ready"] is True
    assert summary["browser_mutation_target_contract_met"] is True
    assert summary["browser_mutation_target_mismatch_count"] == 0
    assert summary["browser_mutation_target_mismatch_users"] == []
    assert summary["browser_mutation_label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert summary["browser_mutation_browser_label_write_target"] == "training_zarr"
    assert summary["browser_mutation_csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert summary["browser_mutation_csv_handoff_artifacts_are_label_write_targets"] is False
    assert summary["browser_mutation_browser_writes_csv_or_handoff_files"] is False
    assert summary["browser_mutation_browser_writes_handoff_csv"] is False
    assert summary["browser_mutation_browser_writes_intermediate_csv"] is False
    assert summary["browser_mutation_browser_has_direct_zarr_write_authority"] is False
    assert summary["dataset_queue_direct_start_enabled"] is True
    assert summary["direct_browser_start_contract_met"] is True
    assert summary["direct_browser_start_mismatch_count"] == 0
    assert summary["direct_browser_start_mismatch_users"] == []
    assert summary["dataset_queue_direct_start_label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert summary["dataset_queue_direct_start_browser_label_write_target"] == "training_zarr"
    assert summary["dataset_queue_direct_start_csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert summary["dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets"] is False
    assert summary["dataset_queue_direct_start_post_body_expected_user_required"] is True
    assert summary["dataset_queue_direct_start_post_body_expected_user_field"] == "expected_user"
    assert summary[
        "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract"
    ] is True
    assert summary[
        "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract"
    ] is True
    assert summary[
        "dataset_queue_direct_start_denied_start_support_includes_authorization_context"
    ] is True
    assert summary[
        "dataset_queue_direct_start_denied_start_contract_reports_no_session_created"
    ] is True
    assert summary[
        "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false"
    ] is True
    assert summary["dataset_queue_direct_start_browser_writes_csv_or_handoff_files"] is False
    assert summary["dataset_queue_direct_start_browser_writes_handoff_csv"] is False
    assert summary["dataset_queue_direct_start_browser_writes_intermediate_csv"] is False
    assert summary["dataset_queue_direct_start_browser_has_direct_zarr_write_authority"] is False
    assert summary["labeler_browser_only"] is True
    assert summary["labeler_requires_local_palette_installation"] is False
    assert summary["labeler_requires_local_crimson_installation"] is False
    assert summary["zarr_backup_copy_before_labeling"] is True
    assert summary["zarr_backup_labelers_do_not_receive_backup_paths"] is True
    assert summary["mutation_audit_server_records_events"] is True
    assert summary["browser_response_security_clickjacking_protection"] is True
    assert summary["session_guard_stale_tab_save_rejected"] is True
    assert summary["task_state_completed_tasks_read_only"] is True
    assert summary["task_state_requires_current_target_token"] is True
    assert summary["signed_link_authorization_grant"] is False
    assert summary["signed_link_binds_expected_user"] is True
    assert summary["signed_link_expected_user_required_on_open"] is True
    assert summary["signed_link_runtime_operator_validation_start_gate_enforced"] is True
    assert (
        summary["signed_link_operator_validation_start_gate_checked_before_session_create"]
        is True
    )
    assert payload["labeler_safety"]["browser_only"] is True
    assert payload["labeler_safety"]["requires_local_palette_installation"] is False
    assert payload["labeler_safety"]["requires_local_crimson_installation"] is False
    assert payload["zarr_backup_policy"]["copy_before_labeling"] is True
    assert payload["zarr_backup_policy"]["labelers_do_not_receive_backup_paths"] is True
    assert payload["mutation_audit_policy"]["server_records_events"] is True
    assert payload["browser_response_security_policy"]["clickjacking_protection"] is True
    assert payload["session_guard_policy"]["stale_tab_save_rejected"] is True
    assert payload["operator_authorization_policy"]["admin_routes_require_operator"] is True
    assert isinstance(payload["operator_authorization_policy"]["operator_boundary_ready"], bool)
    assert payload["operator_recovery_policy"]["task_reopen_operator_only"] is True
    assert payload["operator_recovery_policy"]["failed_promotion_retry_operator_only"] is True
    assert payload["operator_validation_required_before_invite"] is True
    assert payload["operator_validation_all_complete"] is False
    assert payload["operator_validation_status"] == "pending_operator_evidence"
    assert payload["operator_validation_source"] == "none"
    assert payload["operator_validation_gate_count"] == 6
    assert payload["operator_validation_pending_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert payload["operator_validation_required_missing_evidence_gate_ids"] == payload[
        "operator_validation_pending_gate_ids"
    ]
    for artifact in (payload, payload["work"]):
        for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
            assert artifact[f"operator_validation_gate_{gate_id}_status"] == (
                "missing_evidence"
            )
            assert artifact[f"operator_validation_gate_{gate_id}_pending"] is True
            assert artifact[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
            assert artifact[f"operator_validation_gate_{gate_id}_needs_review"] is False
            assert artifact[f"operator_validation_gate_{gate_id}_passed"] is False
    assert payload["operator_validation_visibility_policy"]["operator_only_fields"] == [
        "operator_validation_checklist_path"
    ]
    assert payload["work"]["operator_validation_visibility_policy"] == payload[
        "operator_validation_visibility_policy"
    ]
    assert work["operator_validation_status"] == payload["operator_validation_status"]
    assert work["operator_validation_source"] == payload["operator_validation_source"]
    assert payload["dataset_queue_direct_start_policy"]["enabled"] is True
    assert payload["dataset_queue_direct_start_policy"]["endpoint_route_template"] == (
        "/api/tasks/{task_id}/open"
    )
    assert payload["browser_mutation_write_checklist"]["ready"] is True
    assert payload["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert payload["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert payload["known_user_status"]["is_known_labeler"] is True
    assert payload["known_user_status"]["active_assignment_count"] == 3
    assert payload["single_owner_policy"]["one_active_owner"] is True
    assert payload["assignment_ownership_integrity"]["ok"] is True
    assert payload["assignment_ownership_integrity"]["active_assignment_count"] == 3
    assert payload["assignment_ownership_integrity"]["unique_active_recording_count"] == 3
    assert payload["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    assert payload["assignment_ownership_integrity"][
        "schema_enforced_recording_primary_key"
    ] is True
    assert payload["store_consistency"]["assignment_ownership_integrity"] == payload[
        "assignment_ownership_integrity"
    ]
    assert payload["store_consistency"]["reassignment_session_safety"]["ok"] is True
    assert payload["store_consistency"]["reassignment_session_safety"][
        "active_session_assignment_mismatch_count"
    ] == 0
    assert work["single_owner_policy"] == payload["single_owner_policy"]
    assert work["assignment_ownership_integrity"] == payload["assignment_ownership_integrity"]
    assert payload["labeler_route_authorization_checklist"]["ready"] is True
    assert payload["labeler_route_authorization_checklist"][
        "expected_user_matches_resolved_user"
    ] is True
    assert payload["labeler_route_authorization_checklist"][
        "known_assignment_store_user"
    ] is True
    assert payload["labeler_route_authorization_checklist"][
        "task_open_requires_active_assignment"
    ] is True
    assert payload["labeler_route_authorization_checklist"][
        "mutation_requires_current_target_token"
    ] is True
    assert payload["labeler_route_authorization_checklist"][
        "browser_mutation_target_resolved_server_side"
    ] is True
    assert payload["labeler_route_authorization_checklist"][
        "labelers_mutate_assigned_training_zarrs"
    ] is True
    assert payload["labeler_route_authorization_checklist"][
        "labelers_mutate_intermediate_csvs"
    ] is False
    assert payload["labeler_route_authorization_checklist"] == work[
        "labeler_route_authorization_checklist"
    ]
    assert work["labeler_route_authorization_policy"] == payload[
        "labeler_route_authorization_policy"
    ]
    assert payload["browser_mutation_write_checklist"] == work["browser_mutation_write_checklist"]
    assert work["browser_mutation_write_policy"]["browser_has_direct_zarr_write_authority"] is False
    assert work["dataset_queue_direct_start_policy"] == payload["dataset_queue_direct_start_policy"]
    assert work["user"] == "alice"
    assert [recording["recording_id"] for recording in work["recordings"]] == ["rec-a", "rec-done", "rec-empty"]
    assert work["recordings"][0]["recording_id"] == "rec-a"
    assert work["recordings"][0]["assignment_notes"] == "Alice instructions"
    assert {task["task_id"] for task in work["recordings"][0]["tasks"]} == {"task-a"}
    assert work["recordings"][1]["assignment_notes"] == "Finished recording"
    assert work["recordings"][1]["tasks"] == []
    assert work["recordings"][1]["total_task_count"] == 1
    assert work["recordings"][1]["complete_task_count"] == 1
    assert work["recordings"][1]["no_open_task_reason"] == "all_tasks_complete"
    assert "All tasks for this recording are complete" in work["recordings"][1]["no_open_task_message"]
    assert work["recordings"][2]["assignment_notes"] == "Waiting for task generation"
    assert work["recordings"][2]["tasks"] == []
    assert work["recordings"][2]["total_task_count"] == 0
    assert work["recordings"][2]["no_open_task_reason"] == "tasks_not_generated"
    assert "no browser-labeling tasks have been generated yet" in work["recordings"][2]["no_open_task_message"]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "work-summary",
            "--user",
            "alice",
            "--include-completed",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    task_ids = {task["task_id"] for recording in payload["work"]["recordings"] for task in recording["tasks"]}
    assert rc == 0
    assert task_ids == {"task-a", "task-a-complete", "task-done"}

    approved_checklist_path = tmp_path / "validation-checklist.json"
    approved_output_path = tmp_path / "alice-approved-work-summary.json"
    approved_checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "all_validation_complete": True,
                "ready_for_operator_validation": True,
                "gates": [
                    {
                        "id": gate_id,
                        "status": "passed",
                        "required": True,
                        "evidence": [{"kind": "operator_approval"}],
                        "evidence_recorded_at_utc": "2026-01-01T00:00:00+00:00",
                        "evidence_recorded_by": "operator",
                    }
                    for gate_id in [
                        "mutable_zarr_backup_confirmation",
                        "browser_response_security_headers",
                        "identity_probe_verification",
                        "browser_smoke",
                        "disposable_zarr_mutation_smoke",
                        "operator_recovery_contract",
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "work-summary",
            "--user",
            "alice",
            "--operator-validation-checklist",
            str(approved_checklist_path),
            "--output",
            str(approved_output_path),
        ]
    )

    approved_summary = json.loads(capsys.readouterr().out)
    approved_payload = json.loads(approved_output_path.read_text())
    assert rc == 0
    assert approved_summary["operator_validation_all_complete"] is True
    assert approved_summary["operator_validation_declared_all_complete"] is True
    assert approved_summary["operator_validation_status"] == "passed"
    assert approved_summary["operator_validation_pending_gate_ids"] == []
    assert approved_summary["operator_validation_required_missing_evidence_gate_ids"] == []
    assert approved_summary["operator_validation_operator_only_fields"] == [
        "operator_validation_checklist_path"
    ]
    assert (
        approved_summary[
            "operator_validation_labeler_visible_payloads_include_operator_only_fields"
        ]
        is False
    )
    assert approved_summary["operator_validation_required_pending_gate_count"] == 0
    assert approved_summary["operator_validation_needs_review_gate_count"] == 0
    assert approved_summary["operator_validation_required_missing_evidence_gate_count"] == 0
    assert approved_summary["operator_validation_source"] == "validation_checklist"
    assert approved_summary["operator_validation_operator_action"] == ""
    assert approved_summary["safe_share_checklist_gate_evidence_complete"] is True
    assert approved_summary["safe_share_launch_blocking_unsatisfied_gate_ids"] == []
    assert approved_payload["operator_validation_gate_count"] == 6
    assert approved_payload["operator_validation_checklist_path"] == str(approved_checklist_path)
    assert "operator_validation_checklist_path" not in approved_payload["work"]
    assert approved_payload["work"]["operator_validation_status"] == "passed"
    assert approved_payload["work"]["operator_validation_source"] == "validation_checklist"
    assert approved_payload["safe_share_checklist_gate_evidence_complete"] is True
    assert approved_payload["work"]["safe_share_checklist_gate_evidence_complete"] is True

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "work-summary",
                "--user",
                "alice",
                "--output",
                str(output_path),
            ]
        )


def test_dashboard_roster_cli_exports_dashboard_only_invitations(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "dashboard-roster.json"
    html_path = tmp_path / "dashboard-roster.html"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-empty", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.assign_recording(recording_id="rec-c", assignee_user="carol", status="inactive")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis")
        store.upsert_task(task_id="task-c", recording_id="rec-c", workflow_kind="subject_mask_component")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    payload = json.loads(output_path.read_text())
    alice = next(row for row in payload["users"] if row["user"] == "alice")
    bob = next(row for row in payload["users"] if row["user"] == "bob")

    assert rc == 2
    assert summary["ok"] is False
    assert payload["ok"] is False
    assert payload["dashboard_path"] == "/work"
    assert payload["dashboard_url"] == "https://labeling.example.org/work"
    assert payload["labeler_landing_page_path"] == "/"
    assert payload["labeler_landing_url"] == "https://labeling.example.org"
    assert payload["labeling_home_page_path"] == "/labeling"
    assert payload["labeling_home_url"] == "https://labeling.example.org/labeling"
    assert payload["dataset_queue_page_path"] == "/datasets"
    assert payload["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert payload["personal_work_page_path"] == "/my-work"
    assert payload["personal_work_url"] == "https://labeling.example.org/my-work"
    assert payload["personal_dataset_queue_page_path"] == "/my-datasets"
    assert payload["personal_dataset_queue_url"] == "https://labeling.example.org/my-datasets"
    assert payload["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
    assert payload["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert payload["safe_share_requires_require_shareable_inspection"] is True
    assert payload["safe_share_ready_to_send_is_sufficient"] is False
    assert payload["safe_share_required_inspection_field"] == "labeler_links_safe_to_share"
    assert payload["safe_share_required_inspection_value"] is True
    assert "disposable_zarr_mutation_smoke" in payload[
        "safe_share_launch_blocking_evidence_gate_ids"
    ]
    assert payload["safe_share_checklist_gate_evidence_complete"] is False
    assert "disposable_zarr_mutation_smoke" in payload[
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert "operator_recovery_contract" in payload[
        "safe_share_launch_blocking_unsatisfied_gate_ids"
    ]
    assert summary["safe_share_checklist_gate_evidence_complete"] is False
    assert summary["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke=missing_evidence" in summary["safe_share_next_action_summary"]
    assert "browser_smoke" in summary[
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert payload["labeler_safety"]["dashboard_identity_check_required"] is True
    assert payload["operator_recovery_contract"]["reassignment_closes_previous_owner_sessions"] is True
    assert payload["operator_recovery_contract"][
        "reassignment_closes_previous_owner_sessions_before_assignment_update"
    ] is True
    assert payload["operator_recovery_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert payload["operator_recovery_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert payload["operator_validation_visibility_policy"]["operator_only_fields"] == [
        "operator_validation_checklist_path"
    ]
    assert payload["operator_validation_visibility_policy"][
        "per_user_payloads_use_public_fields_only"
    ] is True
    assert payload["operator_validation_required_before_invite"] is True
    assert payload["operator_validation_all_complete"] is False
    assert payload["operator_validation_status"] == "pending_operator_evidence"
    assert payload["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert payload["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert payload["operator_validation_gate_flat_field_suffixes"] == [
        "status",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]
    assert summary["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert summary["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert "Approve required operator validation evidence" in payload["operator_validation_operator_action"]
    assert payload["operator_validation_command_templates"]["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert "record_browser_smoke_evidence" in payload[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert "apply_operator_evidence_templates" in payload[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert payload["operator_validation_command_templates"]["commands_by_gate_id"][
        "browser_smoke"
    ] == [
        "record_browser_smoke_evidence",
        "apply_operator_evidence_templates",
    ]
    assert alice["operator_validation_command_template_schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert alice["operator_validation_command_template_commands_are_operator_only"] is True
    assert alice["operator_validation_command_template_commands_are_labeler_instructions"] is False
    assert alice["operator_validation_command_template_labelers_must_not_run_commands"] is True
    assert alice["operator_validation_command_template_command_count"] == 7
    assert "record_browser_smoke_evidence" in alice[
        "operator_validation_command_template_command_ids"
    ]
    assert alice[
        "operator_validation_command_template_launch_evidence_collection_plan_schema"
    ] == "palette.web_labeling_launch_evidence_collection_plan.v1"
    assert (
        alice[
            "operator_validation_command_template_launch_evidence_collection_step_count"
        ]
        == 6
    )
    assert "browser_smoke" in alice[
        "operator_validation_command_template_launch_evidence_collection_gate_ids"
    ]
    assert "record_browser_smoke_evidence" in alice[
        "operator_validation_command_template_launch_evidence_collection_record_command_ids"
    ]
    assert alice[
        "operator_validation_command_template_launch_evidence_collection_required_final_field"
    ] == "labeler_links_safe_to_share"
    for row in (alice, bob):
        for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
            assert row[f"operator_validation_gate_{gate_id}_status"] == "missing_evidence"
            assert row[f"operator_validation_gate_{gate_id}_pending"] is True
            assert row[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
            assert row[f"operator_validation_gate_{gate_id}_needs_review"] is False
            assert row[f"operator_validation_gate_{gate_id}_passed"] is False
    for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
        assert payload[f"operator_validation_gate_{gate_id}_status"] == "missing_evidence"
        assert payload[f"operator_validation_gate_{gate_id}_pending"] is True
        assert payload[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
        assert payload[f"operator_validation_gate_{gate_id}_needs_review"] is False
        assert payload[f"operator_validation_gate_{gate_id}_passed"] is False
        assert summary[f"operator_validation_gate_{gate_id}_status"] == "missing_evidence"
        assert summary[f"operator_validation_gate_{gate_id}_pending"] is True
        assert summary[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
        assert summary[f"operator_validation_gate_{gate_id}_needs_review"] is False
        assert summary[f"operator_validation_gate_{gate_id}_passed"] is False
    for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
        assert payload["status_report"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_status"
        ] == "missing_evidence"
        assert payload["status_report"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_pending"
        ] is True
        assert payload["status_report"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_missing_evidence"
        ] is True
    for status_row in payload["status_report"]["user_statuses"]:
        for gate_id in labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS:
            assert status_row[f"operator_validation_gate_{gate_id}_status"] == (
                "missing_evidence"
            )
            assert status_row[f"operator_validation_gate_{gate_id}_pending"] is True
            assert status_row[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
            assert status_row[f"operator_validation_gate_{gate_id}_needs_review"] is False
            assert status_row[f"operator_validation_gate_{gate_id}_passed"] is False
    assert payload["counts"]["users"] == 2
    assert payload["counts"]["copy_intents"] == {"diagnostic_note": 2}
    assert payload["counts"]["ready_states"] == {"not_ready_to_invite": 2}
    assert payload["counts"]["invite_reasons"] == {"operator_validation_pending": 2}
    assert payload["counts"]["total_tasks"] == 2
    assert payload["counts"]["complete_tasks"] == 0
    assert payload["counts"]["completion_percent"] == 0.0
    assert payload["counts"]["completion_states"] == {"not_started": 2}
    assert payload["counts"]["identity_probe_required"] == 2
    assert payload["counts"]["identity_probe_available"] == 2
    assert payload["counts"]["identity_probe_missing"] == 0
    assert payload["counts"]["identity_probe_missing_users"] == []
    assert payload["counts"]["waiting_datasets"] == 2
    assert payload["counts"]["dataset_open_tasks"] == 2
    assert payload["counts"]["users_with_waiting_datasets"] == 2
    assert payload["counts"]["dataset_queue_states"] == {"has_open_dataset_work": 2}
    assert payload["counts"]["dataset_queue_blocked_start_users"] == []
    assert payload["counts"]["browser_mutation_target_contract_all_users_met"] is True
    assert payload["counts"]["browser_mutation_target_contract_not_met_users"] == []
    assert payload["counts"]["browser_mutation_target_contract_not_met_user_count"] == 0
    assert payload["counts"]["browser_mutation_target_total_mismatch_count"] == 0
    assert payload["counts"]["direct_browser_start_contract_all_users_met"] is True
    assert payload["counts"]["direct_browser_start_contract_not_met_users"] == []
    assert payload["counts"]["direct_browser_start_contract_not_met_user_count"] == 0
    assert payload["counts"]["direct_browser_start_total_mismatch_count"] == 0
    assert payload["counts"]["single_owner_policy_contract_all_users_met"] is True
    assert payload["counts"]["single_owner_policy_contract_not_met_users"] == []
    assert payload["counts"]["single_owner_policy_contract_not_met_user_count"] == 0
    assert (
        payload["counts"][
            "labeler_route_authorization_runtime_checklist_gate_all_users_met"
        ]
        is True
    )
    assert (
        payload["counts"][
            "labeler_route_authorization_runtime_checklist_not_met_users"
        ]
        == []
    )
    assert (
        payload["counts"][
            "labeler_route_authorization_runtime_checklist_not_met_user_count"
        ]
        == 0
    )
    assert (
        payload["counts"][
            "labeler_route_authorization_runtime_checklist_total_mismatch_count"
        ]
        == 0
    )
    assert payload["dataset_queue_start_readiness"] == {
        "schema": "palette.web_labeling_dataset_queue_start_readiness.v1",
        "gate_id": "dataset_queue_start_readiness",
        "status": "passed",
        "ready": True,
        "dataset_queue_states": {"has_open_dataset_work": 2},
        "dataset_queue_blocked_start_users": [],
        "blocked_start_user_count": 0,
        "direct_browser_start_contract_not_ready_users": [],
        "direct_browser_start_contract_not_ready_user_count": 0,
        "direct_browser_start_contract_not_ready_reason_counts": {},
        "browser_mutation_target_contract_all_users_met": True,
        "browser_mutation_target_contract_not_met_users": [],
        "browser_mutation_target_contract_not_met_user_count": 0,
        "browser_mutation_target_total_mismatch_count": 0,
        "direct_browser_start_contract_all_users_met": True,
        "direct_browser_start_contract_all_users_ready": True,
        "direct_browser_start_contract_ready_users": ["alice", "bob"],
        "direct_browser_start_contract_ready_user_count": 2,
        "direct_browser_start_contract_task_count": 2,
        "direct_browser_start_contract_ready_task_count": 2,
        "direct_browser_start_contract_not_ready_task_count": 0,
        "direct_browser_start_contract_operator_action_counts": {},
        "direct_browser_start_contract_missing_summary_users": [],
        "direct_browser_start_contract_missing_summary_user_count": 0,
        "direct_browser_start_contract_not_met_users": [],
        "direct_browser_start_contract_not_met_user_count": 0,
        "direct_browser_start_total_mismatch_count": 0,
        "single_owner_policy_contract_all_users_met": True,
        "single_owner_policy_contract_not_met_users": [],
        "single_owner_policy_contract_not_met_user_count": 0,
        "labeler_route_authorization_runtime_checklist_gate_all_users_met": True,
        "labeler_route_authorization_runtime_checklist_not_met_users": [],
        "labeler_route_authorization_runtime_checklist_not_met_user_count": 0,
        "labeler_route_authorization_runtime_checklist_total_mismatch_count": 0,
        "reassignment_session_safety_blocked_users": [],
        "reassignment_session_safety_blocked_user_count": 0,
        "reassignment_session_safety_mismatch_count": 0,
        "reassignment_session_safety_blocked_recording_ids": [],
        "operator_action": "No dataset queue start blockers are visible.",
    }
    assert payload["counts"]["dataset_queue_start_readiness"] == payload["dataset_queue_start_readiness"]
    assert payload["counts"]["dataset_queue_preview_users"] == ["alice", "bob"]
    assert payload["counts"]["personalized_dataset_queue_preview_users"] == ["alice", "bob"]
    assert payload["counts"]["canonical_dataset_queue_preview_users"] == ["alice", "bob"]
    assert payload["counts"]["missing_personalized_dataset_queue_preview_users"] == []
    assert payload["counts"]["all_users_have_personalized_dataset_queue_preview"] is True
    assert payload["counts"]["preferred_personal_queue_match_users"] == ["alice", "bob"]
    assert payload["counts"]["missing_preferred_personal_queue_match_users"] == []
    assert payload["counts"]["all_users_have_preferred_personal_queue_match"] is True
    assert payload["counts"]["personalized_personal_queue_match_users"] == ["alice", "bob"]
    assert payload["counts"]["missing_personalized_personal_queue_match_users"] == []
    assert payload["counts"]["all_users_have_personalized_personal_queue_match"] is True
    assert payload["counts"]["dataset_queue_preferred_entrypoint_counts"] == {
        "personal_datasets_waiting_queue": 2
    }
    assert payload["counts"]["dataset_queue_link_role_counts"] == {
        "canonical_queue_fallback": 2
    }
    assert payload["status_report"]["report_kind"] == "multi_user_labeling_status"
    assert payload["status_report"]["ok"] is False
    assert payload["status_report"]["operator_validation"]["operator_validation_status"] == (
        "pending_operator_evidence"
    )
    assert payload["status_report"]["operator_validation"]["operator_validation_source"] == "none"
    assert payload["status_report"]["operator_validation"][
        "safe_share_checklist_gate_evidence_complete"
    ] is False
    assert "browser_smoke" in payload["status_report"]["operator_validation"][
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert payload["status_report"]["operator_validation_command_templates"] == payload[
        "operator_validation_command_templates"
    ]
    assert payload["status_report"]["safe_share_gate"]["schema"] == (
        "palette.web_labeling_safe_share_gate.v1"
    )
    assert payload["status_report"]["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert payload["status_report"]["safe_share_ready_to_send_is_sufficient"] is False
    assert payload["status_report"]["safe_share_checklist_gate_evidence_complete"] is False
    assert "mutable_zarr_backup_confirmation" in payload["status_report"][
        "safe_share_launch_blocking_missing_evidence_gate_ids"
    ]
    assert (
        payload["status_report"]["safe_share_required_inspection_field"]
        == "labeler_links_safe_to_share"
    )
    assert (
        "operator_validation_checklist_path"
        not in payload["status_report"]["operator_validation"]
    )
    assert payload["status_report"]["operator_validation_visibility_policy"][
        "operator_only_fields"
    ] == ["operator_validation_checklist_path"]
    assert payload["status_report"]["operator_recovery_contract"][
        "reassignment_closes_previous_owner_sessions_before_assignment_update"
    ] is True
    assert payload["status_report"]["operator_recovery_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert payload["status_report"]["operator_recovery_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert payload["status_report"]["users"] == 2
    assert payload["status_report"]["ready_to_invite"] == 0
    assert payload["status_report"]["not_ready_to_invite"] == 2
    assert payload["status_report"]["open_tasks"] == 2
    assert payload["status_report"]["waiting_datasets"] == 2
    assert payload["status_report"]["dataset_open_tasks"] == 2
    assert payload["status_report"]["users_with_waiting_datasets"] == 2
    assert payload["status_report"]["dataset_queue_states"] == {"has_open_dataset_work": 2}
    assert {
        row["labeler_work_completion_status"]
        for row in payload["status_report"]["user_statuses"]
    } == {"waiting"}
    assert {
        row["labeler_work_completion_has_waiting_work"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["labeler_work_completion_completed"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["labeler_work_completion_ready_for_more_labeling"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert payload["status_report"]["dataset_queue_preferred_entrypoint_counts"] == {
        "personal_datasets_waiting_queue": 2
    }
    assert payload["status_report"]["dataset_queue_link_role_counts"] == {
        "canonical_queue_fallback": 2
    }
    assert payload["status_report"]["personalized_dataset_queue_preview_users"] == [
        "alice",
        "bob",
    ]
    assert payload["status_report"]["canonical_dataset_queue_preview_users"] == [
        "alice",
        "bob",
    ]
    assert payload["status_report"]["missing_personalized_dataset_queue_preview_users"] == []
    assert payload["status_report"]["all_users_have_personalized_dataset_queue_preview"] is True
    assert payload["status_report"]["preferred_personal_queue_match_users"] == [
        "alice",
        "bob",
    ]
    assert payload["status_report"]["missing_preferred_personal_queue_match_users"] == []
    assert payload["status_report"]["all_users_have_preferred_personal_queue_match"] is True
    assert payload["status_report"]["personalized_personal_queue_match_users"] == [
        "alice",
        "bob",
    ]
    assert payload["status_report"]["missing_personalized_personal_queue_match_users"] == []
    assert payload["status_report"]["all_users_have_personalized_personal_queue_match"] is True
    assert payload["status_report"]["dataset_queue_blocked_start_users"] == []
    assert payload["status_report"]["dataset_queue_start_readiness"] == payload["dataset_queue_start_readiness"]
    assert payload["status_report"]["browser_mutation_target_contract_all_users_met"] is True
    assert payload["status_report"]["browser_mutation_target_contract_not_met_users"] == []
    assert payload["status_report"]["browser_mutation_target_contract_not_met_user_count"] == 0
    assert payload["status_report"]["browser_mutation_target_total_mismatch_count"] == 0
    assert payload["status_report"]["direct_browser_start_contract_all_users_met"] is True
    assert payload["status_report"]["direct_browser_start_contract_not_met_users"] == []
    assert payload["status_report"]["direct_browser_start_contract_not_met_user_count"] == 0
    assert payload["status_report"]["direct_browser_start_total_mismatch_count"] == 0
    assert payload["status_report"]["single_owner_policy_contract_all_users_met"] is True
    assert payload["status_report"]["single_owner_policy_contract_not_met_users"] == []
    assert payload["status_report"]["single_owner_policy_contract_not_met_user_count"] == 0
    assert (
        payload["status_report"][
            "labeler_route_authorization_runtime_checklist_gate_all_users_met"
        ]
        is True
    )
    assert (
        payload["status_report"][
            "labeler_route_authorization_runtime_checklist_not_met_users"
        ]
        == []
    )
    assert (
        payload["status_report"][
            "labeler_route_authorization_runtime_checklist_not_met_user_count"
        ]
        == 0
    )
    assert (
        payload["status_report"][
            "labeler_route_authorization_runtime_checklist_total_mismatch_count"
        ]
        == 0
    )
    assert payload["dataset_queue_start_readiness"][
        "direct_browser_start_contract_not_ready_users"
    ] == []
    assert payload["dataset_queue_start_readiness"][
        "direct_browser_start_contract_not_ready_user_count"
    ] == 0
    assert payload["dataset_queue_start_readiness"][
        "direct_browser_start_contract_not_ready_reason_counts"
    ] == {}
    assert payload["status_report"]["direct_browser_start_contract_ready_users"] == [
        "alice",
        "bob",
    ]
    assert payload["status_report"]["direct_browser_start_contract_ready_user_count"] == 2
    assert payload["status_report"]["direct_browser_start_contract_not_ready_users"] == []
    assert payload["status_report"]["direct_browser_start_contract_not_ready_user_count"] == 0
    assert payload["status_report"]["direct_browser_start_contract_all_users_ready"] is True
    assert payload["status_report"]["direct_browser_start_contract_task_count"] == 2
    assert payload["status_report"]["direct_browser_start_contract_ready_task_count"] == 2
    assert payload["status_report"]["direct_browser_start_contract_not_ready_task_count"] == 0
    assert payload["status_report"]["direct_browser_start_contract_not_ready_reason_counts"] == {}
    assert payload["status_report"]["direct_browser_start_contract_operator_action_counts"] == {}
    assert payload["status_report"]["completion_states"] == {"not_started": 2}
    assert payload["status_report"]["identity_probe_required"] == 2
    assert payload["status_report"]["identity_probe_available"] == 2
    assert payload["status_report"]["identity_probe_missing"] == 0
    assert payload["status_report"]["identity_probe_missing_users"] == []
    assert {row["user"] for row in payload["status_report"]["user_statuses"]} == {"alice", "bob"}
    assert {
        row["dataset_queue_state_code"] for row in payload["status_report"]["user_statuses"]
    } == {"has_open_dataset_work"}
    assert {
        row["safe_share_gate_id"] for row in payload["status_report"]["user_statuses"]
    } == {"labeler_links_safe_to_share"}
    assert {
        row["safe_share_ready_to_send_is_sufficient"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert all(
        "disposable_zarr_mutation_smoke" in row["safe_share_launch_blocking_evidence_gate_ids"]
        for row in payload["status_report"]["user_statuses"]
    )
    assert {
        row["safe_share_checklist_gate_evidence_complete"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert all(
        "identity_probe_verification" in row["safe_share_launch_blocking_missing_evidence_gate_ids"]
        for row in payload["status_report"]["user_statuses"]
    )
    assert {
        row["dataset_queue_blocks_labeler_start"] for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["dataset_queue_start_status"] for row in payload["status_report"]["user_statuses"]
    } == {"passed"}
    assert {
        row["dataset_queue_start_ready"] for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_enabled"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_method"]
        for row in payload["status_report"]["user_statuses"]
    } == {"POST"}
    assert {
        row["dataset_queue_direct_start_endpoint_route_template"]
        for row in payload["status_report"]["user_statuses"]
    } == {"/api/tasks/{task_id}/open"}
    assert {
        row["dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_post_body_expected_user_required"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_post_body_expected_user_field"]
        for row in payload["status_report"]["user_statuses"]
    } == {"expected_user"}
    assert {
        row["dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_support_includes_authorization_context"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_contract_reports_no_session_created"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_label_mutation_target_kind"]
        for row in payload["status_report"]["user_statuses"]
    } == {"task_scoped_training_zarr"}
    assert {
        row["dataset_queue_direct_start_browser_label_write_target"]
        for row in payload["status_report"]["user_statuses"]
    } == {"training_zarr"}
    assert {
        row["dataset_queue_direct_start_csv_handoff_artifact_role"]
        for row in payload["status_report"]["user_statuses"]
    } == {"metadata_only_control_plane"}
    assert {
        row["dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_writes_csv_or_handoff_files"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_writes_handoff_csv"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_writes_intermediate_csv"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_receives_zarr_write_authority"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_has_direct_zarr_write_authority"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["browser_mutation_target_contract_met"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["browser_mutation_target_mismatch_count"]
        for row in payload["status_report"]["user_statuses"]
    } == {0}
    assert {
        row["direct_browser_start_contract_met"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["direct_browser_start_mismatch_count"]
        for row in payload["status_report"]["user_statuses"]
    } == {0}
    assert {
        row["single_owner_policy_contract_met"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "preferred_require_flag"
    ] == "--require-operator-validation-for-browser-work"
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_start_open"
    ] is True
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_mutations"
    ] is True
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_target_token_check"
    ] is True
    assert payload["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_zarr_write"
    ] is True
    assert payload[
        "runtime_operator_validation_gate_cli_policy_preferred_require_flag"
    ] == "--require-operator-validation-for-browser-work"
    assert payload[
        "runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write"
    ] is True
    assert {
        row["runtime_operator_validation_gate_cli_policy_preferred_require_flag"]
        for row in payload["status_report"]["user_statuses"]
    } == {"--require-operator-validation-for-browser-work"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_protects_browser_start_open"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_protects_browser_mutations"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["direct_browser_start_contract_summary_schema"]
        for row in payload["status_report"]["user_statuses"]
    } == {"palette.web_labeling_direct_browser_start_contract_summary.v1"}
    assert {
        row["direct_browser_start_contract_summary_ready"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["direct_browser_start_contract_summary_browser_label_write_target"]
        for row in payload["status_report"]["user_statuses"]
    } == {"training_zarr"}
    assert {
        row["direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority"]
        for row in payload["status_report"]["user_statuses"]
    } == {False}
    assert {
        row["operator_validation_status"] for row in payload["status_report"]["user_statuses"]
    } == {"pending_operator_evidence"}
    assert {
        row["operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["operator_recovery_reassignment_target_validated_before_session_closure"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["operator_recovery_session_closure_and_assignment_update_atomic"]
        for row in payload["status_report"]["user_statuses"]
    } == {True}
    assert payload["counts"]["ready_to_invite_users"] == []
    assert payload["counts"]["not_ready_to_invite_users"] == ["alice", "bob"]
    ready_invitations_by_user = {
        row["user"]: row["message"] for row in payload["ready_invitations"]
    }
    assert ready_invitations_by_user == {}
    assert {
        row["expected_user_dashboard_url"] for row in payload["users"]
    } == {
        "https://labeling.example.org/work?expected_user=alice",
        "https://labeling.example.org/work?expected_user=bob",
    }
    assert {
        row["expected_user_identity_probe_url"] for row in payload["users"]
    } == {
        "https://labeling.example.org/identity?expected_user=alice",
        "https://labeling.example.org/identity?expected_user=bob",
    }
    assert {
        row["expected_user_dataset_queue_url"] for row in payload["users"]
    } == {
        "https://labeling.example.org/datasets?expected_user=alice",
        "https://labeling.example.org/datasets?expected_user=bob",
    }
    assert {row["preferred_labeler_entrypoint"] for row in payload["users"]} == {
        "personal_datasets_waiting_queue"
    }
    assert {row["preferred_labeler_entry_url"] for row in payload["users"]} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["personal_dataset_queue_link_role"] for row in payload["users"]} == {
        "preferred_queue"
    }
    assert {row["dataset_queue_link_role"] for row in payload["users"]} == {
        "canonical_queue_fallback"
    }
    assert {row["canonical_dataset_queue_link_role"] for row in payload["users"]} == {
        "canonical_queue_fallback"
    }
    assert {row["dashboard_link_role"] for row in payload["users"]} == {"fallback_dashboard"}
    assert {row["identity_probe_link_role"] for row in payload["users"]} == {"identity_check"}
    assert {row["task_links_role"] for row in payload["users"]} == {"convenience_entry_hints"}
    assert {row["single_owner_policy_assignment_scope"] for row in payload["users"]} == {
        "recording"
    }
    assert {
        row["single_owner_policy_recording_assignment_key"] for row in payload["users"]
    } == {"recording_id"}
    assert {
        row["single_owner_policy_multiple_labelers_per_recording_allowed"]
        for row in payload["users"]
    } == {False}
    assert {
        row["single_owner_policy_browser_mutation_requires_current_assignment_owner"]
        for row in payload["users"]
    } == {True}
    assert {
        row["single_owner_policy_browser_mutation_target_resolved_server_side"]
        for row in payload["users"]
    } == {True}
    assert {
        row["single_owner_policy_browser_mutation_target_source"]
        for row in payload["users"]
    } == {"recording_assignments.active_assignment"}
    assert {
        row["single_owner_policy_labelers_mutate_assigned_training_zarrs"]
        for row in payload["users"]
    } == {True}
    assert {
        row["single_owner_policy_labelers_mutate_intermediate_csvs"]
        for row in payload["users"]
    } == {False}
    assert {row["assignment_ownership_contract_ready"] for row in payload["users"]} == {
        True
    }
    assert {
        row["assignment_ownership_contract_assignment_scope"] for row in payload["users"]
    } == {"recording"}
    assert {
        row["assignment_ownership_contract_recording_assignment_key"]
        for row in payload["users"]
    } == {"recording_id"}
    assert {
        row["assignment_ownership_contract_one_active_owner"] for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_multiple_labelers_per_recording_allowed"]
        for row in payload["users"]
    } == {False}
    assert {
        row["assignment_ownership_contract_browser_mutation_requires_current_assignment_owner"]
        for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_browser_mutation_target_resolved_server_side"]
        for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_browser_mutation_target_source"]
        for row in payload["users"]
    } == {"recording_assignments.active_assignment"}
    assert {
        row["assignment_ownership_contract_labelers_mutate_assigned_training_zarrs"]
        for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_labelers_mutate_intermediate_csvs"]
        for row in payload["users"]
    } == {False}
    assert {
        row["assignment_ownership_contract_store_single_owner_assignment_contract_present"]
        for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_store_single_owner_assignment_contract_ready"]
        for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_store_single_owner_assignment_contract_met"]
        for row in payload["users"]
    } == {True}
    assert {
        row["assignment_ownership_contract_store_single_owner_assignment_contract_schema"]
        for row in payload["users"]
    } == {"palette.web_labeling_assignment_single_owner_contract.v1"}
    assert {
        row["assignment_ownership_contract_duplicate_active_owner_count"]
        for row in payload["users"]
    } == {0}
    assert {
        row["preferred_labeler_entry_url_matches_personal_dataset_queue"]
        for row in payload["users"]
    } == {True}
    assert {
        row["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_preview_url"] for row in payload["users"]
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {
        row["canonical_dataset_queue_preview_url"] for row in payload["users"]
    } == {
        "https://labeling.example.org/datasets?expected_user=alice",
        "https://labeling.example.org/datasets?expected_user=bob",
    }
    assert {row["waiting_datasets"] for row in payload["users"]} == {1}
    assert {row["dataset_open_tasks"] for row in payload["users"]} == {1}
    assert {row["dataset_queue_state_code"] for row in payload["users"]} == {"has_open_dataset_work"}
    assert {row["labeler_work_completion_status"] for row in payload["users"]} == {
        "waiting"
    }
    assert {row["labeler_work_completion_has_waiting_work"] for row in payload["users"]} == {
        True
    }
    assert {row["labeler_work_completion_completed"] for row in payload["users"]} == {
        False
    }
    assert {row["dataset_queue_blocks_labeler_start"] for row in payload["users"]} == {False}
    assert {row["dataset_queue_start_status"] for row in payload["users"]} == {"passed"}
    assert {row["dataset_queue_start_ready"] for row in payload["users"]} == {True}
    assert {
        row["operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update"]
        for row in payload["users"]
    } == {True}
    assert {
        row["operator_recovery_reassignment_target_validated_before_session_closure"]
        for row in payload["users"]
    } == {True}
    assert {
        row["operator_recovery_session_closure_and_assignment_update_atomic"]
        for row in payload["users"]
    } == {True}
    assert {row["identity_probe_required"] for row in payload["users"]} == {True}
    assert {row["identity_probe_available"] for row in payload["users"]} == {True}
    assert payload["ready_invitations_text"] == ""
    assert "carol" not in payload["ready_invitations_text"]
    assert {row["user"] for row in payload["users"]} == {"alice", "bob"}
    assert {row["dashboard_url"] for row in payload["users"]} == {"https://labeling.example.org/work"}
    assert {row["labeler_landing_url"] for row in payload["users"]} == {"https://labeling.example.org"}
    assert {row["expected_user_labeler_landing_url"] for row in payload["users"]} == {
        "https://labeling.example.org?expected_user=alice",
        "https://labeling.example.org?expected_user=bob",
    }
    assert {row["labeler_safety"]["dashboard_identity_check_required"] for row in payload["users"]} == {True}
    assert {row["dashboard_identity_check_required"] for row in payload["users"]} == {True}
    assert {row["browser_only"] for row in payload["users"]} == {True}
    assert {row["labeler_runtime_surface"] for row in payload["users"]} == {"browser"}
    assert {row["requires_local_palette_installation"] for row in payload["users"]} == {False}
    assert {row["requires_local_crimson_installation"] for row in payload["users"]} == {False}
    assert {row["requires_local_conda_environment"] for row in payload["users"]} == {False}
    assert {row["requires_local_project_dependencies"] for row in payload["users"]} == {False}
    assert {row["no_direct_zarr_edits"] for row in payload["users"]} == {True}
    assert {row["dataset_queue_direct_start_enabled"] for row in payload["users"]} == {True}
    assert {row["dataset_queue_direct_start_method"] for row in payload["users"]} == {"POST"}
    assert {row["dataset_queue_direct_start_endpoint_route_template"] for row in payload["users"]} == {
        "/api/tasks/{task_id}/open"
    }
    assert {
        row["dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_post_body_expected_user_required"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_post_body_expected_user_field"]
        for row in payload["users"]
    } == {"expected_user"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_preferred_require_flag"]
        for row in payload["users"]
    } == {"--require-operator-validation-for-browser-work"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_protects_browser_start_open"]
        for row in payload["users"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_protects_browser_mutations"]
        for row in payload["users"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check"]
        for row in payload["users"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write"]
        for row in payload["users"]
    } == {True}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_support_includes_authorization_context"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_contract_reports_no_session_created"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint"]
        for row in payload["users"]
    } == {True}
    assert {
        row["dataset_queue_direct_start_label_mutation_target_kind"]
        for row in payload["users"]
    } == {"task_scoped_training_zarr"}
    assert {
        row["dataset_queue_direct_start_browser_label_write_target"]
        for row in payload["users"]
    } == {"training_zarr"}
    assert {
        row["dataset_queue_direct_start_csv_handoff_artifact_role"]
        for row in payload["users"]
    } == {"metadata_only_control_plane"}
    assert {
        row["dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets"]
        for row in payload["users"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_writes_csv_or_handoff_files"]
        for row in payload["users"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_writes_handoff_csv"]
        for row in payload["users"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_writes_intermediate_csv"]
        for row in payload["users"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_receives_zarr_write_authority"]
        for row in payload["users"]
    } == {False}
    assert {
        row["dataset_queue_direct_start_browser_has_direct_zarr_write_authority"]
        for row in payload["users"]
    } == {False}
    assert {row["browser_mutation_target_contract_met"] for row in payload["users"]} == {
        True
    }
    assert {row["browser_mutation_target_mismatch_count"] for row in payload["users"]} == {
        0
    }
    assert {row["direct_browser_start_contract_met"] for row in payload["users"]} == {
        True
    }
    assert {row["direct_browser_start_mismatch_count"] for row in payload["users"]} == {
        0
    }
    assert {row["single_owner_policy_contract_met"] for row in payload["users"]} == {
        True
    }
    assert {row["direct_browser_start_contract_summary_schema"] for row in payload["users"]} == {
        "palette.web_labeling_direct_browser_start_contract_summary.v1"
    }
    assert {row["direct_browser_start_contract_summary_ready"] for row in payload["users"]} == {True}
    assert {row["direct_browser_start_contract_summary_browser_label_write_target"] for row in payload["users"]} == {
        "training_zarr"
    }
    assert {row["direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority"] for row in payload["users"]} == {
        False
    }
    assert {row["ready_to_invite"] for row in payload["users"]} == {False}
    assert {row["ready_state"] for row in payload["users"]} == {"not_ready_to_invite"}
    assert {row["copy_label"] for row in payload["users"]} == {"Copy not-ready note"}
    assert {row["copy_intent"] for row in payload["users"]} == {"diagnostic_note"}
    assert {tuple(row["invite_reasons"]) for row in payload["users"]} == {("operator_validation_pending",)}
    assert {
        "Approve required operator validation evidence" in row["invite_actions"][0]
        for row in payload["users"]
    } == {True}
    assert {row["operator_validation_required_before_invite"] for row in payload["users"]} == {True}
    assert {row["operator_validation_all_complete"] for row in payload["users"]} == {False}
    assert {row["operator_validation_status"] for row in payload["users"]} == {
        "pending_operator_evidence"
    }
    assert alice["expected_user_personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert alice["expected_user_personal_work_url"] == (
        "https://labeling.example.org/my-work?expected_user=alice"
    )
    assert alice["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert alice["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert bob["expected_user_personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=bob"
    )
    assert "Your Palette labeling work is ready." not in alice["invitation_message"]
    assert "Dashboard ready-row draft is not ready: operator_validation_pending" in alice["invitation_message"]
    assert "Preview only:" in alice["invitation_message"]
    assert "start page: https://labeling.example.org?expected_user=alice" in alice["invitation_message"]
    assert "labeling home: https://labeling.example.org/labeling?expected_user=alice" in alice[
        "invitation_message"
    ]
    assert "personalized work queue: https://labeling.example.org/my-datasets?expected_user=alice" in alice[
        "invitation_message"
    ]
    assert "dataset queue: https://labeling.example.org/datasets?expected_user=alice" in alice[
        "invitation_message"
    ]
    assert "dashboard: https://labeling.example.org/work?expected_user=alice" in alice["invitation_message"]
    assert "identity check: https://labeling.example.org/identity?expected_user=alice" in alice[
        "invitation_message"
    ]
    assert "Preview only:" in alice["invitation_message"]
    assert "Next action: Approve required operator validation evidence" in alice[
        "invitation_message"
    ]
    assert alice["recordings"] == 2
    assert alice["visible_tasks"] == 1
    assert alice["open_tasks"] == 1
    assert alice["waiting_datasets"] == 1
    assert alice["dataset_open_tasks"] == 1
    assert alice["dataset_queue_summary"]["waiting_dataset_count"] == 1
    assert alice["dataset_queue_state"]["code"] == "has_open_dataset_work"
    assert alice["dataset_queue_blocks_labeler_start"] is False
    assert alice["dataset_queue_preview_url"] == "https://labeling.example.org/my-datasets?expected_user=alice"
    assert (
        alice["canonical_dataset_queue_preview_url"]
        == "https://labeling.example.org/datasets?expected_user=alice"
    )
    assert alice["expected_user_dataset_queue_url"] == "https://labeling.example.org/datasets?expected_user=alice"
    assert alice["total_tasks"] == 1
    assert alice["complete_tasks"] == 0
    assert alice["completion_percent"] == 0.0
    assert alice["completion_state"] == "not_started"
    assert alice["recordings_without_open_tasks"] == 1
    assert alice["recordings_without_open_tasks_by_reason"] == {"tasks_not_generated": 1}
    assert "Generate or import browser-labeling tasks" in alice["recordings_without_open_tasks_actions"][0]
    assert bob["open_tasks"] == 1
    assert bob["waiting_datasets"] == 1
    assert bob["dataset_open_tasks"] == 1
    assert bob["dataset_queue_summary"]["waiting_dataset_count"] == 1
    assert bob["dataset_queue_preview_url"] == "https://labeling.example.org/my-datasets?expected_user=bob"
    assert (
        bob["canonical_dataset_queue_preview_url"]
        == "https://labeling.example.org/datasets?expected_user=bob"
    )
    assert bob["expected_user_dataset_queue_url"] == "https://labeling.example.org/datasets?expected_user=bob"
    assert bob["total_tasks"] == 1
    assert bob["complete_tasks"] == 0
    assert bob["completion_percent"] == 0.0
    assert bob["completion_state"] == "not_started"
    assert bob["recordings_without_open_tasks_by_reason"] == {}
    assert "carol" not in {row["user"] for row in payload["users"]}

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--format",
            "html",
            "--output",
            str(html_path),
        ]
    )

    html_summary = json.loads(capsys.readouterr().out)
    html = html_path.read_text()
    assert rc == 2
    assert html_summary["ok"] is False
    assert html_summary["format"] == "html"
    assert html_summary["dataset_queue_start_readiness"]["ready"] is True
    assert html_summary["browser_mutation_target_contract_all_users_met"] is True
    assert html_summary["browser_mutation_target_contract_not_met_users"] == []
    assert html_summary["browser_mutation_target_contract_not_met_user_count"] == 0
    assert html_summary["browser_mutation_target_total_mismatch_count"] == 0
    assert html_summary["direct_browser_start_contract_all_users_met"] is True
    assert html_summary["direct_browser_start_contract_not_met_users"] == []
    assert html_summary["direct_browser_start_contract_not_met_user_count"] == 0
    assert html_summary["direct_browser_start_total_mismatch_count"] == 0
    assert html_summary["single_owner_policy_contract_all_users_met"] is True
    assert html_summary["single_owner_policy_contract_not_met_users"] == []
    assert html_summary["single_owner_policy_contract_not_met_user_count"] == 0
    assert (
        html_summary[
            "labeler_route_authorization_runtime_checklist_gate_all_users_met"
        ]
        is True
    )
    assert (
        html_summary[
            "labeler_route_authorization_runtime_checklist_not_met_users"
        ]
        == []
    )
    assert (
        html_summary[
            "labeler_route_authorization_runtime_checklist_not_met_user_count"
        ]
        == 0
    )
    assert (
        html_summary[
            "labeler_route_authorization_runtime_checklist_total_mismatch_count"
        ]
        == 0
    )
    assert "Palette dashboard ready-row draft roster" in html
    assert "Status report: multi_user_labeling_status" in html
    assert "Dashboard URL: https://labeling.example.org/work" in html
    assert "no local install" in html
    assert "Landing URL: https://labeling.example.org" in html
    assert "Labeling home URL: https://labeling.example.org/labeling" in html
    assert "Dataset queue URL: https://labeling.example.org/datasets" in html
    assert "https://labeling.example.org?expected_user=alice" in html
    assert "https://labeling.example.org/my-datasets?expected_user=alice" in html
    assert "Canonical fallback: https://labeling.example.org/datasets?expected_user=alice" in html
    assert "Ready-row draft users: none" in html
    assert "Diagnostic-note users: alice, bob" in html
    assert 'Ready states: {&quot;not_ready_to_invite&quot;: 2}' in html
    assert "Identity probes: 2 / 2 available; missing 0 (none)" in html
    assert "Waiting datasets: 2" in html
    assert 'Dataset queue states: {&quot;has_open_dataset_work&quot;: 2}' in html
    assert "Blocked-start users: none" in html
    assert "Queue start readiness: passed" in html
    assert "Blocked users: none" in html
    assert "Compact contract gates: browser_mutation_target_contract_all_users_met=True" in html
    assert "browser_mutation_target_total_mismatch_count=0" in html
    assert "browser_mutation_target_contract_not_met_users=none" in html
    assert "direct_browser_start_contract_all_users_met=True" in html
    assert "direct_browser_start_total_mismatch_count=0" in html
    assert "direct_browser_start_contract_not_met_users=none" in html
    assert "single_owner_policy_contract_all_users_met=True" in html
    assert "single_owner_policy_contract_not_met_users=none" in html
    assert (
        "labeler_route_authorization_runtime_checklist_gate_all_users_met=True"
        in html
    )
    assert (
        "labeler_route_authorization_runtime_checklist_total_mismatch_count=0"
        in html
    )
    assert (
        "labeler_route_authorization_runtime_checklist_not_met_users=none"
        in html
    )
    assert (
        "Operator recovery: ready=True; validates target before session closure=True; "
        "session closure/update atomic=True; "
        "reassignment closes previous-owner sessions=True; "
        "before owner update=True"
    ) in html
    assert "validates reassignment target before closing sessions" in html
    assert "session closure and assignment update are atomic" in html
    assert "reassignment closes previous-owner sessions before owner update" in html
    assert "compact contract gates: browser_mutation_target_contract_met=True" in html
    assert "browser_mutation_target_mismatch_count=0" in html
    assert "direct_browser_start_contract_met=True" in html
    assert "direct_browser_start_mismatch_count=0" in html
    assert "single_owner_policy_contract_met=True" in html
    assert "https://labeling.example.org/work?expected_user=alice" in html
    assert "https://labeling.example.org/datasets?expected_user=alice" in html
    assert "https://labeling.example.org/datasets?expected_user=bob" in html
    assert "datasets_waiting_queue" in html
    assert "entry=personal_datasets_waiting_queue" in html
    assert "personal_queue=preferred_queue" in html
    assert "queue=canonical_queue_fallback" in html
    assert "canonical_queue=canonical_queue_fallback" in html
    assert "dashboard=fallback_dashboard" in html
    assert "identity=identity_check" in html
    assert "tasks=convenience_entry_hints" in html
    assert "Completion: 0 / 2 tasks complete (0.0%)" in html
    assert 'Completion states: {&quot;not_started&quot;: 2}' in html
    assert 'Copy intents: {&quot;diagnostic_note&quot;: 2}' in html
    assert "Ready-row drafts only; safe-share review required" in html
    assert "No ready-row draft text is available to copy" in html
    assert "Copy ready-row draft text" not in html
    assert "Operator validation commands" in html
    assert "Safe-share next actions: Safe-share next actions: 6;" in html
    assert "Safe-share blocker action detail fields:" in html
    assert "operator_validation_record_command_ids" in html
    assert "Safe-share blocker action command fields:" in html
    assert "operator_validation_evidence_template_path" in html
    assert "Runtime validation gate CLI" in html
    assert "--require-operator-validation-for-browser-work" in html
    assert "Operator validation start gate" in html
    assert "Operator validation mutation gate" in html
    assert "blocks task open" in html
    assert "blocks browser mutation" in html
    assert "record-browser-smoke-evidence" in html
    assert "apply-operator-evidence-templates" in html
    assert "These are operator-only next steps, not labeler instructions." in html
    assert "<th>Completion</th>" in html
    assert "<th>Queue state</th>" in html
    assert "<th>Preferred entry</th>" in html
    assert "<th>Dataset queue</th>" in html
    assert "<th>Guarded queue page</th>" in html
    assert "<th>Next action</th>" in html
    assert "<th>Safety</th>" in html
    assert "<th>Identity probe</th>" in html
    assert "https://labeling.example.org/identity?expected_user=alice" in html
    assert "https://labeling.example.org/identity?expected_user=bob" in html
    assert "confirm dashboard user" in html
    assert "browser only" in html
    assert "has_open_dataset_work" in html
    assert "blocks start: False" in html
    assert "Your Palette labeling work is ready." not in html
    assert "Dashboard ready-row draft is not ready: operator_validation_pending" in html
    assert "dataset queue: https://labeling.example.org/datasets" in html
    assert "dashboard: https://labeling.example.org/work" in html
    assert "<textarea readonly>" in html
    assert "Copy not-ready note" in html
    assert "Copy ready-row draft" not in html
    assert "copyInvitation" in html

    approved_output_path = tmp_path / "dashboard-roster-approved.json"
    approved_checklist_path = tmp_path / "approved-validation-checklist.json"
    approved_checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "bundle_label": "dashboard roster launch approval",
                "all_validation_complete": True,
                "ready_for_operator_validation": True,
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "passed",
                        "required": True,
                        "evidence": ["Operator approved deployed identity verification."],
                    },
                    {
                        "id": "browser_response_security_headers",
                        "status": "passed",
                        "required": True,
                        "evidence": ["Operator approved deployed browser response security headers."],
                    },
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--operator-validation-checklist",
            str(approved_checklist_path),
            "--output",
            str(approved_output_path),
        ]
    )

    approved_summary = json.loads(capsys.readouterr().out)
    approved_payload = json.loads(approved_output_path.read_text())
    approved_alice = next(row for row in approved_payload["users"] if row["user"] == "alice")

    assert rc == 0
    assert approved_summary["ok"] is True
    assert approved_payload["ok"] is True
    assert approved_payload["operator_validation_all_complete"] is True
    assert approved_payload["operator_validation_declared_all_complete"] is True
    assert approved_payload["operator_validation_gate_count"] == 2
    assert approved_payload["operator_validation_status"] == "passed"
    assert approved_payload["operator_validation_source"] == "validation_checklist"
    assert approved_payload["operator_validation_required_before_invite_legacy_semantics"] == (
        "operator_validation_required_before_ready_row_draft_not_safe_share_approval"
    )
    assert approved_payload["operator_validation_required_before_invite_is_safe_share_approval"] is False
    assert approved_payload["operator_validation_required_before_invite_safe_share_field"] == (
        "labeler_links_safe_to_share"
    )
    assert approved_payload["operator_launch_approved_legacy_semantics"] == (
        "operator_validation_evidence_only_not_safe_share_approval"
    )
    assert approved_payload["operator_launch_approved_is_safe_share_approval"] is False
    assert approved_payload["operator_launch_approved_requires_safe_share_inspection"] is True
    assert approved_payload["operator_launch_approved_required_safe_share_field"] == (
        "labeler_links_safe_to_share"
    )
    assert approved_payload["operator_launch_approved_required_safe_share_value"] is True
    assert approved_payload["operator_validation_checklist_path"] == str(approved_checklist_path)
    assert approved_summary["operator_validation_status"] == "passed"
    assert approved_summary["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert "operator_validation_checklist_path" not in approved_alice
    assert approved_payload["status_report"]["operator_validation"]["operator_validation_status"] == "passed"
    assert approved_payload["status_report"]["operator_validation"]["operator_validation_source"] == (
        "validation_checklist"
    )
    assert approved_payload["status_report"]["operator_validation"][
        "operator_launch_approved_is_safe_share_approval"
    ] is False
    assert approved_payload["status_report"]["operator_validation"][
        "operator_validation_required_before_invite_is_safe_share_approval"
    ] is False
    for gate_id in ("identity_probe_verification", "browser_response_security_headers"):
        assert approved_payload[f"operator_validation_gate_{gate_id}_status"] == "passed"
        assert approved_payload[f"operator_validation_gate_{gate_id}_pending"] is False
        assert approved_payload[f"operator_validation_gate_{gate_id}_passed"] is True
        assert approved_summary[f"operator_validation_gate_{gate_id}_status"] == "passed"
        assert approved_summary[f"operator_validation_gate_{gate_id}_pending"] is False
        assert approved_summary[f"operator_validation_gate_{gate_id}_passed"] is True
        assert approved_payload["status_report"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_status"
        ] == "passed"
        assert approved_payload["status_report"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_pending"
        ] is False
        assert approved_payload["status_report"]["operator_validation"][
            f"operator_validation_gate_{gate_id}_passed"
        ] is True
        assert approved_alice[f"operator_validation_gate_{gate_id}_status"] == "passed"
        assert approved_alice[f"operator_validation_gate_{gate_id}_pending"] is False
        assert approved_alice[f"operator_validation_gate_{gate_id}_passed"] is True
    assert (
        "operator_validation_checklist_path"
        not in approved_payload["status_report"]["operator_validation"]
    )
    assert approved_payload["status_report"]["operator_validation_visibility_policy"][
        "per_user_payloads_use_public_fields_only"
    ] is True
    assert approved_payload["counts"]["ready_to_invite"] == 2
    assert approved_payload["counts"]["not_ready_to_invite"] == 0
    assert approved_payload["ready_to_invite_legacy_semantics"] == (
        "row_readiness_not_safe_share_approval"
    )
    assert approved_payload["ready_row_state_values"] == [
        "ready_row_draft",
        "diagnostic_note",
    ]
    assert approved_payload["copy_intent_values"] == [
        "ready_row_draft",
        "diagnostic_note",
    ]
    assert approved_payload["counts"]["ready_row_draft_count"] == 2
    assert approved_payload["counts"]["diagnostic_note_count"] == 0
    assert approved_payload["counts"]["ready_row_draft_users"] == ["alice", "bob"]
    assert approved_payload["counts"]["diagnostic_note_users"] == []
    assert approved_payload["counts"]["copy_intents"] == {"ready_row_draft": 2}
    assert approved_payload["status_report"]["ready_to_invite_legacy_semantics"] == (
        "row_readiness_not_safe_share_approval"
    )
    assert approved_payload["status_report"]["ready_row_state_values"] == [
        "ready_row_draft",
        "diagnostic_note",
    ]
    assert approved_payload["status_report"]["copy_intent_values"] == [
        "ready_row_draft",
        "diagnostic_note",
    ]
    assert approved_payload["status_report"]["ready_row_draft_count"] == 2
    assert approved_payload["status_report"]["diagnostic_note_count"] == 0
    assert approved_payload["status_report"]["ready_row_draft_users"] == ["alice", "bob"]
    assert approved_payload["status_report"]["diagnostic_note_users"] == []
    assert approved_payload["counts"]["ready_states"] == {"ready_to_invite": 2}
    assert approved_payload["counts"]["ready_to_invite_users"] == ["alice", "bob"]
    assert approved_payload["counts"]["not_ready_to_invite_users"] == []
    assert approved_payload["counts"]["invite_reasons"] == {}
    assert {row["operator_validation_all_complete"] for row in approved_payload["users"]} == {True}
    assert {row["operator_validation_status"] for row in approved_payload["users"]} == {"passed"}
    assert {row["operator_validation_source"] for row in approved_payload["users"]} == {
        "validation_checklist"
    }
    assert {row["operator_validation_gate_count"] for row in approved_payload["users"]} == {2}
    assert {
        row["operator_validation_source"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"validation_checklist"}
    assert {
        row["operator_validation_gate_count"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {2}
    for status_row in approved_payload["status_report"]["user_statuses"]:
        for gate_id in ("identity_probe_verification", "browser_response_security_headers"):
            assert status_row[f"operator_validation_gate_{gate_id}_status"] == "passed"
            assert status_row[f"operator_validation_gate_{gate_id}_pending"] is False
            assert status_row[f"operator_validation_gate_{gate_id}_passed"] is True
    assert {
        row["preferred_labeler_entrypoint"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"personal_datasets_waiting_queue"}
    assert {
        row["preferred_labeler_entry_url_matches_personal_dataset_queue"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {True}
    assert {
        row["personal_dataset_queue_link_role"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"preferred_queue"}
    assert {
        row["dataset_queue_link_role"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"canonical_queue_fallback"}
    assert {
        row["canonical_dataset_queue_link_role"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"canonical_queue_fallback"}
    assert {
        row["dataset_queue_preview_url"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {
        row["canonical_dataset_queue_preview_url"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {
        "https://labeling.example.org/datasets?expected_user=alice",
        "https://labeling.example.org/datasets?expected_user=bob",
    }
    assert {
        "operator_validation_checklist_path" in row
        for row in approved_payload["status_report"]["user_statuses"]
    } == {False}
    assert {row["ready_to_invite"] for row in approved_payload["users"]} == {True}
    assert {row["ready_to_invite_legacy_semantics"] for row in approved_payload["users"]} == {
        "row_readiness_not_safe_share_approval"
    }
    assert {row["ready_row_state"] for row in approved_payload["users"]} == {
        "ready_row_draft"
    }
    assert {row["ready_row_draft_bundle_schema"] for row in approved_payload["users"]} == {
        "palette.web_labeling_ready_row_draft_bundle.v1"
    }
    assert {row["ready_row_draft_bundle_kind"] for row in approved_payload["users"]} == {
        "ready_row_draft_text"
    }
    assert {
        tuple(row["ready_row_state_values"])
        for row in approved_payload["users"]
    } == {("ready_row_draft", "diagnostic_note")}
    assert {
        tuple(row["copy_intent_values"])
        for row in approved_payload["users"]
    } == {("ready_row_draft", "diagnostic_note")}
    assert {
        row["ready_row_draft_requires_safe_share_inspection"]
        for row in approved_payload["users"]
    } == {True}
    assert {row["ready_row_draft_required_safe_share_field"] for row in approved_payload["users"]} == {
        "labeler_links_safe_to_share"
    }
    assert {row["ready_row_draft_required_safe_share_value"] for row in approved_payload["users"]} == {
        True
    }
    assert {row["copy_intent"] for row in approved_payload["users"]} == {"ready_row_draft"}
    assert {
        row["ready_to_invite_legacy_semantics"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"row_readiness_not_safe_share_approval"}
    assert {
        row["ready_row_state"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"ready_row_draft"}
    assert {
        row["ready_row_draft_bundle_schema"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"palette.web_labeling_ready_row_draft_bundle.v1"}
    assert {
        tuple(row["ready_row_state_values"])
        for row in approved_payload["status_report"]["user_statuses"]
    } == {("ready_row_draft", "diagnostic_note")}
    assert {
        tuple(row["copy_intent_values"])
        for row in approved_payload["status_report"]["user_statuses"]
    } == {("ready_row_draft", "diagnostic_note")}
    assert {
        row["ready_row_draft_required_safe_share_field"]
        for row in approved_payload["status_report"]["user_statuses"]
    } == {"labeler_links_safe_to_share"}
    assert {row["copy_intent"] for row in approved_payload["status_report"]["user_statuses"]} == {
        "ready_row_draft"
    }
    assert "Your Palette labeling work is ready." in approved_alice["invitation_message"]
    assert "Start here: https://labeling.example.org/my-datasets?expected_user=alice" in approved_alice[
        "invitation_message"
    ]
    assert "Open your personalized dataset queue: https://labeling.example.org/my-datasets?expected_user=alice" in approved_alice[
        "invitation_message"
    ]
    assert "Canonical dataset queue fallback: https://labeling.example.org/datasets?expected_user=alice" in approved_alice[
        "invitation_message"
    ]
    assert "Queue-first start page: https://labeling.example.org?expected_user=alice" in approved_alice[
        "invitation_message"
    ]
    assert "Human-readable labeling home alias: https://labeling.example.org/labeling?expected_user=alice" in approved_alice[
        "invitation_message"
    ]
    assert "Full dashboard fallback: https://labeling.example.org/work?expected_user=alice" in approved_alice[
        "invitation_message"
    ]
    assert approved_alice["invitation_message"] in approved_payload["ready_invitations_text"]
    assert approved_payload["ready_row_draft_bundle_schema"] == (
        "palette.web_labeling_ready_row_draft_bundle.v1"
    )
    assert approved_payload["ready_row_draft_bundle_kind"] == "ready_row_draft_text"
    assert approved_payload["ready_invitations_legacy_semantics"] == (
        "draft_text_only_safe_share_required"
    )
    assert approved_payload["ready_invitations_legacy_field_names"] == [
        "ready_invitations",
        "ready_invitations_text",
    ]
    assert approved_payload["ready_row_drafts"] == approved_payload["ready_invitations"]
    assert approved_payload["ready_row_draft_text"] == approved_payload["ready_invitations_text"]
    assert "labeler_links_safe_to_share=true" in approved_payload["ready_row_draft_share_rule"]

    stale_checklist_path = tmp_path / "stale-validation-checklist.json"
    stale_output_path = tmp_path / "dashboard-roster-stale-checklist.json"
    stale_checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "bundle_label": "stale dashboard roster launch approval",
                "all_validation_complete": True,
                "ready_for_operator_validation": True,
                "gates": [
                    {
                        "id": "browser_response_security_headers",
                        "status": "pending_operator_evidence",
                        "required": True,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--operator-validation-checklist",
            str(stale_checklist_path),
            "--output",
            str(stale_output_path),
        ]
    )

    stale_summary = json.loads(capsys.readouterr().out)
    stale_payload = json.loads(stale_output_path.read_text())

    assert rc == 2
    assert stale_summary["ok"] is False
    assert stale_payload["ok"] is False
    assert stale_payload["operator_validation_declared_all_complete"] is True
    assert stale_payload["operator_validation_all_complete"] is False
    assert stale_payload["operator_validation_gate_count"] == 1
    assert stale_payload["operator_validation_status"] == "pending_operator_evidence"
    assert stale_payload["operator_validation_pending_gate_ids"] == [
        "browser_response_security_headers"
    ]
    assert stale_payload["counts"]["ready_to_invite"] == 0
    assert stale_payload["counts"]["not_ready_to_invite"] == 2
    assert stale_payload["counts"]["invite_reasons"] == {"operator_validation_pending": 2}
    assert stale_payload["ready_invitations_text"] == ""

    empty_gate_checklist_path = tmp_path / "empty-gate-validation-checklist.json"
    empty_gate_output_path = tmp_path / "dashboard-roster-empty-gate-checklist.json"
    empty_gate_checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "bundle_label": "empty-gate dashboard roster launch approval",
                "all_validation_complete": True,
                "ready_for_operator_validation": True,
                "gates": [],
            },
            sort_keys=True,
        )
        + "\n"
    )
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--operator-validation-checklist",
            str(empty_gate_checklist_path),
            "--output",
            str(empty_gate_output_path),
        ]
    )

    empty_gate_summary = json.loads(capsys.readouterr().out)
    empty_gate_payload = json.loads(empty_gate_output_path.read_text())

    assert rc == 2
    assert empty_gate_summary["ok"] is False
    assert empty_gate_payload["ok"] is False
    assert empty_gate_payload["operator_validation_declared_all_complete"] is True
    assert empty_gate_payload["operator_validation_all_complete"] is False
    assert empty_gate_payload["operator_validation_gate_count"] == 0
    assert empty_gate_payload["operator_validation_status"] == "pending_operator_evidence"
    assert empty_gate_payload["counts"]["ready_to_invite"] == 0
    assert empty_gate_payload["counts"]["invite_reasons"] == {"operator_validation_pending": 2}

    with pytest.raises(ValueError, match="--operator-validation-checklist.*--operator-launch-approved"):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "dashboard-roster",
                "--base-url",
                "https://labeling.example.org",
                "--operator-validation-checklist",
                str(approved_checklist_path),
                "--operator-launch-approved",
            ]
        )


def test_dashboard_roster_cli_marks_users_without_open_tasks_not_ready(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    html_path = tmp_path / "dashboard-roster-no-open.html"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-done", assignee_user="alice")
        store.upsert_task(task_id="task-done", recording_id="rec-done", workflow_kind="keypoints", state="complete")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    row = payload["users"][0]

    assert rc == 2
    assert payload["ok"] is False
    assert payload["counts"]["ready_to_invite"] == 0
    assert payload["counts"]["not_ready_to_invite"] == 1
    assert payload["counts"]["invite_reasons"] == {"no_open_tasks": 1}
    assert payload["counts"]["copy_intents"] == {"diagnostic_note": 1}
    assert payload["counts"]["ready_states"] == {"not_ready_to_invite": 1}
    assert payload["counts"]["ready_to_invite_users"] == []
    assert payload["counts"]["not_ready_to_invite_users"] == ["alice"]
    assert payload["counts"]["dataset_queue_blocked_start_users"] == ["alice"]
    assert payload["dataset_queue_start_readiness"]["status"] == "needs_review"
    assert payload["dataset_queue_start_readiness"]["ready"] is False
    assert payload["dataset_queue_start_readiness"]["dataset_queue_blocked_start_users"] == ["alice"]
    assert payload["status_report"]["dataset_queue_start_readiness"] == payload["dataset_queue_start_readiness"]
    assert payload["counts"]["dashboard_warnings"] == {
        "dataset_queue_blocks_labeler_start": 1,
        "no_open_tasks": 1,
    }
    assert payload["ready_invitations"] == []
    assert payload["ready_invitations_text"] == ""
    assert payload["warning_codes"] == ["dataset_queue_blocks_labeler_start", "no_open_tasks"]
    assert payload["status_report"]["warning_codes"] == ["dataset_queue_blocks_labeler_start", "no_open_tasks"]
    assert payload["status_report"]["dashboard_warnings"] == {
        "dataset_queue_blocks_labeler_start": 1,
        "no_open_tasks": 1,
    }
    assert row["ready_to_invite"] is False
    assert row["ready_state"] == "not_ready_to_invite"
    assert row["dataset_queue_blocks_labeler_start"] is True
    assert row["dataset_queue_state"]["blocks_labeler_start"] is True
    assert row["dataset_queue_start_ready"] is False
    assert row["dataset_queue_start_status"] == "needs_review"
    assert row["copy_label"] == "Copy not-ready note"
    assert row["copy_intent"] == "diagnostic_note"
    assert row["invite_reasons"] == ["no_open_tasks"]
    assert row["invite_actions"] == [
        "Generate, import, or reopen browser-labeling tasks for this user's active recordings."
    ]
    assert "Dashboard ready-row draft is not ready: no_open_tasks" in row["invitation_message"]
    assert "Next action: Generate, import, or reopen browser-labeling tasks" in row["invitation_message"]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--format",
            "html",
            "--output",
            str(html_path),
        ]
    )

    html_summary = json.loads(capsys.readouterr().out)
    html = html_path.read_text()

    assert rc == 2
    assert html_summary["ok"] is False
    assert html_summary["warning_codes"] == ["dataset_queue_blocks_labeler_start", "no_open_tasks"]
    assert "Ready-row draft users: none" in html
    assert "Diagnostic-note users: alice" in html
    assert 'Ready states: {&quot;not_ready_to_invite&quot;: 1}' in html
    assert "Blocked-start users: alice" in html
    assert "blocks start: True" in html
    assert 'Copy intents: {&quot;diagnostic_note&quot;: 1}' in html
    assert "Ready-row drafts only; safe-share review required" in html
    assert "No ready-row draft text is available to copy" in html
    assert "Copy ready-row draft text" not in html
    assert "Invite actions:" in html
    assert "Generate, import, or reopen browser-labeling tasks" in html
    assert "Copy not-ready note" in html
    assert "Copy ready-row draft" not in html


def test_dashboard_roster_blocks_reassignment_session_safety_issue(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.create_session(task_id="task-a", user="alice", ttl_seconds=600)
        store.assign_recording(
            recording_id="rec-a",
            assignee_user="bob",
            allow_stale_open_sessions=True,
        )
        store.assign_recording(recording_id="rec-b", assignee_user="charlie")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--operator-launch-approved",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    rows_by_user = {row["user"]: row for row in payload["users"]}
    bob = rows_by_user["bob"]
    charlie = rows_by_user["charlie"]

    assert rc == 2
    assert payload["counts"]["ready_to_invite"] == 1
    assert payload["counts"]["not_ready_to_invite"] == 1
    assert payload["counts"]["invite_reasons"] == {"reassignment_session_safety_failed": 1}
    assert payload["counts"]["dataset_queue_states"] == {
        "has_open_dataset_work": 1,
        "reassignment_session_safety_failed": 1,
    }
    assert payload["counts"]["dataset_queue_blocked_start_users"] == ["bob"]
    assert payload["counts"]["reassignment_session_safety_blocked_users"] == ["bob"]
    assert payload["counts"]["reassignment_session_safety_blocked_user_count"] == 1
    assert payload["counts"]["reassignment_session_safety_mismatch_count"] == 1
    assert payload["counts"]["reassignment_session_safety_blocked_recording_ids"] == ["rec-a"]
    assert payload["counts"]["direct_browser_start_contract_not_ready_users"] == ["bob"]
    assert payload["counts"]["direct_browser_start_contract_not_ready_user_count"] == 1
    assert payload["counts"]["direct_browser_start_contract_all_users_ready"] is False
    assert payload["dataset_queue_start_readiness"]["status"] == "needs_review"
    assert payload["dataset_queue_start_readiness"]["reassignment_session_safety_blocked_users"] == [
        "bob"
    ]
    assert payload["dataset_queue_start_readiness"][
        "direct_browser_start_contract_not_ready_users"
    ] == ["bob"]
    assert payload["dataset_queue_start_readiness"][
        "direct_browser_start_contract_not_ready_user_count"
    ] == 1
    assert payload["status_report"]["reassignment_session_safety_blocked_users"] == ["bob"]
    assert payload["status_report"]["reassignment_session_safety_mismatch_count"] == 1
    assert payload["status_report"]["direct_browser_start_contract_not_ready_users"] == ["bob"]
    assert payload["status_report"]["direct_browser_start_contract_not_ready_user_count"] == 1
    assert bob["ready_to_invite"] is False
    assert bob["invite_reasons"] == ["reassignment_session_safety_failed"]
    assert "stale previous-owner sessions" in bob["invite_actions"][0]
    assert bob["dataset_queue_state_code"] == "reassignment_session_safety_failed"
    assert bob["dataset_queue_blocks_labeler_start"] is True
    assert bob["dataset_queue_start_ready"] is False
    assert bob["dataset_open_tasks"] == 0
    assert bob["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert bob["reassignment_session_safety_active_session_assignment_mismatch_count"] == 1
    assert bob["reassignment_session_safety"]["active_session_assignment_mismatch_recording_ids"] == [
        "rec-a"
    ]
    assert bob["operator_recovery_reassignment_session_repair_route"] == (
        "/api/admin/recordings/{recording_id}/repair-reassignment-sessions"
    )
    assert bob["operator_recovery_task_repair_route"] == "/api/admin/tasks/{task_id}/repair"
    assert charlie["ready_to_invite"] is True
    assert charlie["invite_reasons"] == []
    assert charlie["dataset_queue_state_code"] == "has_open_dataset_work"
    assert charlie["dataset_queue_blocks_labeler_start"] is False
    assert charlie["dataset_open_tasks"] == 1
    assert charlie["reassignment_session_safety"]["ok"] is True
    assert charlie["reassignment_session_safety_blocks_labeler_mutation"] is False
    status_by_user = {row["user"]: row for row in payload["status_report"]["user_statuses"]}
    assert status_by_user["bob"]["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert status_by_user["bob"]["operator_recovery_reassignment_session_repair_route"] == (
        "/api/admin/recordings/{recording_id}/repair-reassignment-sessions"
    )
    assert status_by_user["bob"]["operator_recovery_task_state_route"] == (
        "/api/admin/tasks/{task_id}/state"
    )
    assert status_by_user["bob"]["operator_recovery_task_repair_route"] == (
        "/api/admin/tasks/{task_id}/repair"
    )
    assert status_by_user["bob"]["operator_recovery_audit_event_lookup_route"] == (
        "/api/admin/events/{event_id}"
    )
    assert status_by_user["charlie"]["reassignment_session_safety_blocks_labeler_mutation"] is False
    admin_summary_config = ServerConfig(
        store_path=store_path,
        host="127.0.0.1",
        port=0,
        fixed_user=None,
        auth_header="X-User",
        session_ttl_seconds=3600,
        trust_auth_header=True,
        admin_users=("admin@example.org",),
        production=True,
    )
    admin_store = LabelingStore(store_path)
    try:
        admin_summary = _admin_summary_payload(admin_store, config=admin_summary_config)
        admin_bob = _admin_user_payload(admin_store, user="bob")
        admin_charlie = _admin_user_payload(admin_store, user="charlie")
    finally:
        admin_store.close()
    assert admin_summary["reassignment_session_safety_blocked_users"] == ["bob"]
    assert admin_summary["reassignment_session_safety_blocked_user_count"] == 1
    assert admin_summary["reassignment_session_safety_mismatch_count"] == 1
    assert admin_summary["reassignment_session_safety_blocked_recording_ids"] == ["rec-a"]
    assert admin_bob["reassignment_session_safety"]["ok"] is False
    assert admin_bob["personal_work_page_path"] == "/my-work"
    assert admin_bob["personal_dataset_queue_page_path"] == "/my-datasets"
    assert admin_bob["expected_user_personal_work_url"] == "/my-work?expected_user=bob"
    assert admin_bob["expected_user_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=bob"
    )
    assert admin_bob["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert admin_bob["reassignment_session_safety_active_session_assignment_mismatch_count"] == 1
    assert admin_bob["reassignment_session_safety_active_session_assignment_mismatch_recording_ids"] == [
        "rec-a"
    ]
    admin_bob_html = _admin_user_html(
        user="bob",
        work=admin_bob["work"],
        dashboard_row=admin_bob["dashboard_user"],
    ).decode("utf-8")
    assert "/my-work?expected_user=bob" in admin_bob_html
    assert "/my-datasets?expected_user=bob" in admin_bob_html
    assert "/api/admin/recordings/rec-a/repair-reassignment-sessions" in admin_bob_html
    assert "Repair sessions for rec-a" in admin_bob_html
    assert "repairReassignmentSessions" in admin_bob_html
    recording_store = LabelingStore(store_path)
    try:
        admin_recording = _admin_recording_payload(recording_store, recording_id="rec-a")
    finally:
        recording_store.close()
    assert admin_recording["reassignment_session_safety_blocks_labeler_mutation"] is True
    assert admin_recording["personal_work_page_path"] == "/my-work"
    assert admin_recording["personal_dataset_queue_page_path"] == "/my-datasets"
    assert admin_recording["expected_user_personal_work_url"] == "/my-work?expected_user=bob"
    assert admin_recording["expected_user_personal_dataset_queue_url"] == (
        "/my-datasets?expected_user=bob"
    )
    assert admin_recording["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert admin_recording["personalized_labeler_entry_url"] == (
        "/my-datasets?expected_user=bob"
    )
    assert admin_recording[
        "reassignment_session_safety_active_session_assignment_mismatch_count"
    ] == 1
    assert admin_recording["reassignment_session_repair_route"] == (
        "/api/admin/recordings/rec-a/repair-reassignment-sessions"
    )
    admin_recording_html = _admin_recording_html(admin_recording).decode("utf-8")
    assert "/my-work?expected_user=bob" in admin_recording_html
    assert "/my-datasets?expected_user=bob" in admin_recording_html
    assert "Repair reassignment sessions" in admin_recording_html
    assert "/api/admin/recordings/rec-a/repair-reassignment-sessions" in admin_recording_html
    assert admin_charlie["reassignment_session_safety"]["ok"] is True
    assert admin_charlie["reassignment_session_safety_blocks_labeler_mutation"] is False

    csv_path = tmp_path / "dashboard-roster-reassignment-session-safety.csv"
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--operator-launch-approved",
            "--format",
            "csv",
            "--output",
            str(csv_path),
        ]
    )

    csv_summary = json.loads(capsys.readouterr().out)
    csv_rows = list(csv.DictReader(csv_path.open()))
    csv_by_user = {row["user"]: row for row in csv_rows}
    assert rc == 2
    assert csv_summary["ok"] is False
    assert csv_by_user["bob"]["reassignment_session_safety_blocks_labeler_mutation"] == "True"
    assert csv_by_user["bob"][
        "reassignment_session_safety_active_session_assignment_mismatch_count"
    ] == "1"
    assert "rec-a" in csv_by_user["bob"]["reassignment_session_safety"]
    assert csv_by_user["bob"]["operator_recovery_reassignment_session_repair_route"] == (
        "/api/admin/recordings/{recording_id}/repair-reassignment-sessions"
    )
    assert csv_by_user["bob"]["operator_recovery_task_repair_route"] == (
        "/api/admin/tasks/{task_id}/repair"
    )
    assert csv_by_user["bob"]["operator_validation_command_template_schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert csv_by_user["bob"]["operator_validation_command_template_commands_are_operator_only"] == "True"
    assert csv_by_user["bob"][
        "operator_validation_command_template_commands_are_labeler_instructions"
    ] == "False"
    assert csv_by_user["bob"][
        "operator_validation_command_template_labelers_must_not_run_commands"
    ] == "True"
    assert csv_by_user["bob"]["operator_validation_command_template_command_count"] == "0"
    assert csv_by_user["bob"]["operator_validation_command_template_command_ids"] == "[]"
    assert csv_by_user["bob"][
        "operator_validation_command_template_launch_evidence_collection_plan_schema"
    ] == "palette.web_labeling_launch_evidence_collection_plan.v1"
    assert csv_by_user["bob"][
        "operator_validation_command_template_launch_evidence_collection_step_count"
    ] == "0"
    assert csv_by_user["bob"][
        "operator_validation_command_template_launch_evidence_collection_gate_ids"
    ] == "[]"
    assert csv_by_user["bob"][
        "operator_validation_command_template_launch_evidence_collection_record_command_ids"
    ] == "[]"
    assert csv_by_user["bob"][
        "operator_validation_command_template_launch_evidence_collection_required_final_field"
    ] == "labeler_links_safe_to_share"
    assert csv_by_user["charlie"]["reassignment_session_safety_blocks_labeler_mutation"] == "False"
    assert csv_by_user["charlie"][
        "reassignment_session_safety_active_session_assignment_mismatch_count"
    ] == "0"


def test_dashboard_roster_include_completed_does_not_make_completed_only_user_ready(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-done", assignee_user="alice")
        store.upsert_task(task_id="task-done", recording_id="rec-done", workflow_kind="keypoints", state="complete")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--include-completed",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    row = payload["users"][0]

    assert rc == 2
    assert payload["ok"] is False
    assert row["visible_tasks"] == 1
    assert row["open_tasks"] == 0
    assert row["total_tasks"] == 1
    assert row["complete_tasks"] == 1
    assert row["ready_to_invite"] is False
    assert row["invite_reasons"] == ["no_open_tasks"]


def test_dashboard_roster_cli_marks_empty_roster_not_ready(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--base-url",
            "https://labeling.example.org",
            "--user",
            "missing-user",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert payload["ok"] is False
    assert payload["counts"]["users"] == 0
    assert payload["counts"]["invite_reasons"] == {"no_users": 1}
    assert payload["warning_codes"] == ["no_users"]
    assert payload["users"] == []


def test_dashboard_roster_cli_without_base_url_is_not_ready_to_invite(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "dashboard-roster.csv"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "dashboard-roster",
            "--format",
            "csv",
            "--output",
            str(output_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    rows = list(csv.DictReader(output_path.open()))

    assert rc == 2
    assert summary["ok"] is False
    assert summary["warning_codes"] == ["missing_base_url"]
    assert summary["dataset_queue_start_readiness"]["ready"] is True
    assert summary["dataset_queue_start_readiness"][
        "browser_mutation_target_contract_all_users_met"
    ] is True
    assert summary["dataset_queue_start_readiness"][
        "direct_browser_start_contract_all_users_met"
    ] is True
    assert summary["dataset_queue_start_readiness"][
        "single_owner_policy_contract_all_users_met"
    ] is True
    assert summary["dataset_queue_start_readiness"][
        "labeler_route_authorization_runtime_checklist_gate_all_users_met"
    ] is True
    assert summary["browser_mutation_target_contract_all_users_met"] is True
    assert summary["browser_mutation_target_contract_not_met_users"] == []
    assert summary["browser_mutation_target_contract_not_met_user_count"] == 0
    assert summary["browser_mutation_target_total_mismatch_count"] == 0
    assert summary["direct_browser_start_contract_all_users_met"] is True
    assert summary["direct_browser_start_contract_not_met_users"] == []
    assert summary["direct_browser_start_contract_not_met_user_count"] == 0
    assert summary["direct_browser_start_total_mismatch_count"] == 0
    assert summary["single_owner_policy_contract_all_users_met"] is True
    assert summary["single_owner_policy_contract_not_met_users"] == []
    assert summary["single_owner_policy_contract_not_met_user_count"] == 0
    assert (
        summary[
            "labeler_route_authorization_runtime_checklist_gate_all_users_met"
        ]
        is True
    )
    assert (
        summary[
            "labeler_route_authorization_runtime_checklist_not_met_users"
        ]
        == []
    )
    assert (
        summary[
            "labeler_route_authorization_runtime_checklist_not_met_user_count"
        ]
        == 0
    )
    assert (
        summary[
            "labeler_route_authorization_runtime_checklist_total_mismatch_count"
        ]
        == 0
    )
    assert rows[0]["user"] == "alice"
    assert rows[0]["dashboard_path"] == "/work"
    assert rows[0]["dashboard_url"] == ""
    assert rows[0]["expected_user_identity_probe_url"] == ""
    assert rows[0]["assignment_ownership_contract_ready"] == "True"
    assert rows[0]["assignment_ownership_contract_assignment_scope"] == "recording"
    assert rows[0]["assignment_ownership_contract_recording_assignment_key"] == "recording_id"
    assert rows[0]["assignment_ownership_contract_primary_key_columns"] == (
        "[\"recording_id\"]"
    )
    assert rows[0]["assignment_ownership_contract_one_active_owner"] == "True"
    assert rows[0][
        "assignment_ownership_contract_multiple_labelers_per_recording_allowed"
    ] == "False"
    assert rows[0][
        "assignment_ownership_contract_browser_mutation_requires_current_assignment_owner"
    ] == "True"
    assert rows[0]["assignment_ownership_contract_duplicate_active_owner_count"] == "0"
    assert rows[0]["single_owner_policy_contract_met"] == "True"
    assert rows[0]["waiting_datasets"] == "1"
    assert rows[0]["dataset_open_tasks"] == "1"
    assert rows[0]["dataset_queue_state_code"] == "has_open_dataset_work"
    assert rows[0]["dataset_queue_blocks_labeler_start"] == "False"
    assert rows[0]["dataset_queue_start_ready"] == "True"
    assert rows[0]["dataset_queue_start_status"] == "passed"
    assert rows[0]["dataset_queue_preview_url"] == ""
    assert rows[0]["identity_probe_required"] == "True"
    assert rows[0]["identity_probe_available"] == "False"
    assert rows[0]["dashboard_identity_check_required"] == "True"
    assert rows[0]["browser_only"] == "True"
    assert rows[0]["labeler_runtime_surface"] == "browser"
    assert rows[0]["requires_local_palette_installation"] == "False"
    assert rows[0]["requires_local_crimson_installation"] == "False"
    assert rows[0]["requires_local_conda_environment"] == "False"
    assert rows[0]["requires_local_project_dependencies"] == "False"
    assert rows[0]["no_direct_zarr_edits"] == "True"
    assert rows[0]["browser_mutation_authoritative_label_state"] == "assigned_task_zarr_scope"
    assert rows[0]["browser_mutation_data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert rows[0]["browser_mutation_mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert rows[0]["browser_mutation_label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert rows[0]["browser_mutation_browser_label_write_target"] == "training_zarr"
    assert rows[0]["browser_mutation_server_mutates_task_scoped_zarr_targets"] == "True"
    assert rows[0]["browser_mutation_training_zarr_mutations_are_server_owned"] == "True"
    assert rows[0]["browser_mutation_promotion_training_zarr_requires_task_scope"] == "True"
    assert rows[0]["browser_mutation_handoff_artifacts_are_metadata_only"] == "True"
    assert rows[0]["browser_mutation_csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert rows[0]["browser_mutation_csv_handoff_artifacts_are_label_write_targets"] == "False"
    assert rows[0]["browser_mutation_handoff_csv_artifacts_are_label_write_targets"] == "False"
    assert rows[0]["browser_mutation_intermediate_csv_artifacts_are_label_write_targets"] == "False"
    assert rows[0]["browser_mutation_browser_writes_csv_or_handoff_files"] == "False"
    assert rows[0]["browser_mutation_browser_writes_handoff_csv"] == "False"
    assert rows[0]["browser_mutation_browser_writes_intermediate_csv"] == "False"
    assert rows[0]["browser_mutation_browser_receives_zarr_write_authority"] == "False"
    assert rows[0]["browser_mutation_browser_has_direct_zarr_write_authority"] == "False"
    assert rows[0]["browser_mutation_target_contract_met"] == "True"
    assert rows[0]["browser_mutation_target_mismatch_count"] == "0"
    assert rows[0]["browser_mutation_target_mismatch_users"] == "[]"
    assert rows[0]["dataset_queue_direct_start_enabled"] == "True"
    assert rows[0]["dataset_queue_direct_start_method"] == "POST"
    assert rows[0]["dataset_queue_direct_start_endpoint_route_template"] == "/api/tasks/{task_id}/open"
    assert rows[0]["dataset_queue_direct_start_same_origin_only"] == "True"
    assert rows[0]["dataset_queue_direct_start_exact_route_required"] == "True"
    assert rows[0]["dataset_queue_direct_start_endpoint_task_segment_must_match_row_task_id"] == "True"
    assert rows[0]["dataset_queue_direct_start_expected_user_guard_required"] == "True"
    assert rows[0]["dataset_queue_direct_start_post_body_expected_user_required"] == "True"
    assert rows[0]["dataset_queue_direct_start_post_body_expected_user_field"] == "expected_user"
    assert rows[0][
        "dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract"
    ] == "True"
    assert rows[0][
        "dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract"
    ] == "True"
    assert rows[0][
        "dataset_queue_direct_start_denied_start_support_includes_authorization_context"
    ] == "True"
    assert rows[0][
        "dataset_queue_direct_start_denied_start_contract_reports_no_session_created"
    ] == "True"
    assert rows[0][
        "dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false"
    ] == "True"
    assert rows[0]["dataset_queue_direct_start_non_startable_tasks_do_not_advertise_endpoint"] == "True"
    assert rows[0]["dataset_queue_direct_start_label_mutation_target_kind"] == "task_scoped_training_zarr"
    assert rows[0]["dataset_queue_direct_start_browser_label_write_target"] == "training_zarr"
    assert rows[0]["dataset_queue_direct_start_csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert rows[0]["dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets"] == "False"
    assert rows[0]["dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets"] == "False"
    assert rows[0][
        "dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets"
    ] == "False"
    assert rows[0]["dataset_queue_direct_start_browser_writes_csv_or_handoff_files"] == "False"
    assert rows[0]["dataset_queue_direct_start_browser_writes_handoff_csv"] == "False"
    assert rows[0]["dataset_queue_direct_start_browser_writes_intermediate_csv"] == "False"
    assert rows[0]["dataset_queue_direct_start_browser_receives_zarr_write_authority"] == "False"
    assert rows[0]["dataset_queue_direct_start_browser_has_direct_zarr_write_authority"] == "False"
    assert rows[0]["direct_browser_start_contract_met"] == "True"
    assert rows[0]["direct_browser_start_mismatch_count"] == "0"
    assert rows[0]["direct_browser_start_mismatch_users"] == "[]"
    assert rows[0]["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert rows[0]["safe_share_ready_to_send_is_sufficient"] == "False"
    assert rows[0]["safe_share_launch_blocking_gate_count"] == "6"
    assert rows[0]["safe_share_launch_blocking_missing_evidence_gate_count"] == "6"
    assert rows[0]["safe_share_launch_blocking_unsatisfied_gate_count"] == "6"
    assert rows[0]["safe_share_checklist_gate_evidence_complete"] == "False"
    assert rows[0]["safe_share_launch_blocking_next_action_count"] == "6"
    assert rows[0]["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 6;"
    )
    assert "browser_smoke" in rows[0]["safe_share_launch_blocking_next_actions"]
    assert (
        "operator_validation_record_command_ids"
        in rows[0]["safe_share_launch_blocking_next_action_detail_fields"]
    )
    assert (
        "operator_validation_evidence_template_path"
        in rows[0]["safe_share_launch_blocking_next_action_command_fields"]
    )
    assert "browser_smoke" in rows[0]["safe_share_launch_blocking_missing_evidence_gate_ids"]
    assert "operator_recovery_contract" in rows[0][
        "safe_share_launch_blocking_unsatisfied_gate_ids"
    ]
    assert rows[0]["operator_recovery_ready"] == "True"
    assert rows[0]["operator_recovery_reassignment_closes_previous_owner_sessions"] == "True"
    assert (
        rows[0][
            "operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update"
        ]
        == "True"
    )
    assert (
        rows[0][
            "operator_recovery_reassignment_target_validated_before_session_closure"
        ]
        == "True"
    )
    assert rows[0]["operator_recovery_session_closure_and_assignment_update_atomic"] == "True"
    assert rows[0]["operator_recovery_task_reopen_operator_only"] == "True"
    assert rows[0]["operator_validation_required_before_invite"] == "True"
    assert rows[0]["operator_validation_all_complete"] == "False"
    assert rows[0]["operator_validation_declared_all_complete"] == "False"
    assert rows[0]["operator_validation_ready_for_operator_validation"] == "False"
    assert rows[0]["operator_validation_status"] == "pending_operator_evidence"
    assert rows[0]["operator_validation_source"] == "none"
    assert "operator_validation_checklist_path" not in rows[0]
    assert rows[0]["operator_validation_operator_only_fields"] == (
        "[\"operator_validation_checklist_path\"]"
    )
    assert rows[0]["operator_validation_labeler_visible_payloads_include_operator_only_fields"] == "False"
    assert rows[0]["operator_validation_per_user_payloads_use_public_fields_only"] == "True"
    assert rows[0]["operator_validation_top_level_operator_reports_may_include_operator_only_fields"] == "True"
    assert rows[0]["operator_validation_gate_count"] == "6"
    assert json.loads(rows[0]["operator_validation_required_missing_evidence_gate_ids"]) == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert rows[0]["operator_validation_required_pending_gate_count"] == "6"
    assert rows[0]["operator_validation_needs_review_gate_count"] == "0"
    assert rows[0]["operator_validation_required_missing_evidence_gate_count"] == "6"
    assert "Approve required operator validation evidence" in rows[0]["operator_validation_operator_action"]
    assert rows[0]["ready_to_invite"] == "False"
    assert rows[0]["ready_state"] == "not_ready_to_invite"
    assert rows[0]["copy_label"] == "Copy not-ready note"
    assert rows[0]["copy_intent"] == "diagnostic_note"
    assert rows[0]["invite_reasons"] == "[\"missing_base_url\"]"
    assert "--base-url" in rows[0]["invite_actions"]
    assert "missing_base_url" in rows[0]["invitation_message"]
    assert "--base-url" in rows[0]["invitation_message"]


def test_export_user_handoff_cli_writes_preview_links_and_check(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "alice-handoff"
    zip_path = tmp_path / "alice-handoff.zip"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.assign_recording(recording_id="rec-empty", assignee_user="alice", notes="Waiting for task generation")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Alice task",
            priority=7,
            notes="Task-specific instructions",
            scope={"zarr_path": "/secret/alice-training.zarr", "target_frames": [1, 2]},
        )
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis", title="Bob task")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-user-handoff",
            "--user",
            "alice",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--output-dir",
            str(output_dir),
            "--zip-output",
            str(zip_path),
        ]
    )

    manifest = json.loads(capsys.readouterr().out)
    work_summary = json.loads((output_dir / "work-summary.json").read_text())
    dataset_queue = json.loads((output_dir / "dataset-queue.json").read_text())
    links = [json.loads(line) for line in (output_dir / "signed-links.jsonl").read_text().splitlines()]
    check = json.loads((output_dir / "check-store.json").read_text())
    manifest_file = json.loads((output_dir / "manifest.json").read_text())
    html_index = (output_dir / "index.html").read_text()
    message = (output_dir / "message.txt").read_text()
    quickstart = (output_dir / "labeler-quickstart.txt").read_text()
    validation_log = (output_dir / "validation-log-template.md").read_text()
    validation_checklist = json.loads((output_dir / "validation-checklist.json").read_text())
    validation_gate_statuses = {gate["id"]: gate["status"] for gate in validation_checklist["gates"]}

    assert rc == 0
    assert manifest["ok"] is True
    assert manifest["files"]["html_index"] == str(output_dir / "index.html")
    assert manifest["files"]["dataset_queue"] == str(output_dir / "dataset-queue.json")
    assert manifest["files"]["message"] == str(output_dir / "message.txt")
    assert manifest["files"]["quickstart"] == str(output_dir / "labeler-quickstart.txt")
    assert manifest["files"]["validation_log"] == str(output_dir / "validation-log-template.md")
    assert manifest["files"]["validation_checklist"] == str(output_dir / "validation-checklist.json")
    assert manifest["files"]["bundle_zip"] == str(zip_path)
    assert manifest["generated_at_utc"]
    assert manifest["links_expire_at_utc"]
    assert manifest["labeler_landing_page_path"] == "/"
    assert manifest["labeler_landing_url"] == "https://labeling.example.org"
    assert manifest["expected_user_labeler_landing_url"] == "https://labeling.example.org?expected_user=alice"
    assert manifest["dashboard_path"] == "/work"
    assert manifest["dashboard_url"] == "https://labeling.example.org/work"
    assert manifest["expected_user_dashboard_url"] == "https://labeling.example.org/work?expected_user=alice"
    assert manifest["expected_user_dataset_queue_url"] == "https://labeling.example.org/datasets?expected_user=alice"
    assert manifest["expected_user_personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert manifest["expected_user_personal_work_url"] == (
        "https://labeling.example.org/my-work?expected_user=alice"
    )
    assert manifest["expected_user_identity_probe_url"] == "https://labeling.example.org/identity?expected_user=alice"
    assert manifest["preferred_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert manifest["preferred_labeler_entry_url"] == "https://labeling.example.org/my-datasets?expected_user=alice"
    assert manifest["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert manifest["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert manifest["personal_dataset_queue_link_role"] == "preferred_queue"
    assert manifest["dataset_queue_link_role"] == "canonical_queue_fallback"
    assert manifest["canonical_dataset_queue_link_role"] == "canonical_queue_fallback"
    assert manifest["dashboard_link_role"] == "fallback_dashboard"
    assert manifest["identity_probe_link_role"] == "identity_check"
    assert manifest["task_links_role"] == "convenience_entry_hints"
    assert manifest["preferred_labeler_entry_url_matches_dataset_queue"] is True
    assert manifest["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert manifest["known_user_status"]["is_known_labeler"] is True
    assert manifest["known_user_status"]["active_assignment_count"] == 2
    assert manifest["guarded_links_ready"] is True
    assert manifest["missing_guarded_links"] == []
    assert manifest["handoff_artifacts_ready"] is True
    assert manifest["missing_handoff_artifacts"] == []
    assert manifest["handoff_entry_readiness"] == "passed"
    assert manifest["operator_validation_visibility_policy"]["operator_only_fields"] == [
        "operator_validation_checklist_path"
    ]
    assert work_summary["operator_validation_visibility_policy"] == manifest[
        "operator_validation_visibility_policy"
    ]
    assert work_summary["work"]["operator_validation_visibility_policy"] == manifest[
        "operator_validation_visibility_policy"
    ]
    assert dataset_queue["operator_validation_visibility_policy"] == manifest[
        "operator_validation_visibility_policy"
    ]
    assert manifest["operator_validation_command_templates"]["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert manifest["operator_validation_command_templates"]["commands_are_operator_only"] is True
    assert manifest["operator_validation_command_templates"][
        "commands_are_labeler_instructions"
    ] is False
    assert manifest["operator_validation_command_templates"]["labelers_must_not_run_commands"] is True
    assert "record_browser_smoke_evidence" in manifest[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert "apply_operator_evidence_templates" in manifest[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert manifest["operator_validation_command_templates"][
        "launch_evidence_collection_plan_schema"
    ] == "palette.web_labeling_launch_evidence_collection_plan.v1"
    assert manifest["operator_validation_command_templates"][
        "launch_evidence_collection_step_count"
    ] == 6
    assert manifest["operator_validation_command_templates"][
        "launch_evidence_collection_required_final_field"
    ] == "labeler_links_safe_to_share"
    assert manifest["operator_validation_command_templates"][
        "launch_evidence_collection_final_inspection_command"
    ] == "inspect-handoff --path PACKAGE --require-shareable"
    assert manifest["operator_validation_command_templates"][
        "launch_evidence_collection_plan"
    ]["steps_by_gate_id"]["browser_smoke"]["record_command_id"] == (
        "record_browser_smoke_evidence"
    )
    assert work_summary["operator_validation_command_templates"] == manifest[
        "operator_validation_command_templates"
    ]
    assert work_summary["work"]["operator_validation_command_templates"] == manifest[
        "operator_validation_command_templates"
    ]
    assert dataset_queue["operator_validation_command_templates"] == manifest[
        "operator_validation_command_templates"
    ]
    assert work_summary["operator_validation_pending_gate_ids"] == manifest[
        "operator_validation_pending_gate_ids"
    ]
    assert dataset_queue["operator_validation_pending_gate_ids"] == manifest[
        "operator_validation_pending_gate_ids"
    ]
    for artifact in (manifest, work_summary, work_summary["work"], dataset_queue):
        assert artifact["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
        assert artifact["safe_share_gate_id"] == "labeler_links_safe_to_share"
        assert artifact["labeler_links_safe_to_share"] is False
        assert artifact["ready_to_send_is_sufficient_for_safe_share"] is False
        assert artifact["safe_share_launch_blocking_gate_count"] == 6
        assert artifact["safe_share_launch_blocking_unsatisfied_gate_count"] == 6
        assert artifact["safe_share_checklist_gate_evidence_complete"] is False
        assert set(artifact["safe_share_launch_blocking_missing_evidence_gate_ids"]) == {
            "mutable_zarr_backup_confirmation",
            "browser_response_security_headers",
            "identity_probe_verification",
            "browser_smoke",
            "disposable_zarr_mutation_smoke",
            "operator_recovery_contract",
        }
        for gate_id in (
            "mutable_zarr_backup_confirmation",
            "browser_response_security_headers",
            "identity_probe_verification",
            "browser_smoke",
            "disposable_zarr_mutation_smoke",
            "operator_recovery_contract",
        ):
            assert artifact[f"operator_validation_gate_{gate_id}_status"] == (
                "missing_evidence"
            )
            assert artifact[f"operator_validation_gate_{gate_id}_pending"] is True
            assert artifact[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
            assert artifact[f"operator_validation_gate_{gate_id}_needs_review"] is False
            assert artifact[f"operator_validation_gate_{gate_id}_passed"] is False
    assert manifest["labeler_safety"]["dashboard_identity_check_required"] is True
    assert manifest["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert manifest["queue_first_entry_contract"]["ready"] is True
    assert manifest["queue_first_entry_contract"]["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert work_summary["queue_first_entry_contract"] == manifest[
        "queue_first_entry_contract"
    ]
    assert work_summary["work"]["queue_first_entry_contract"] == manifest[
        "queue_first_entry_contract"
    ]
    assert dataset_queue["queue_first_entry_contract"] == manifest[
        "queue_first_entry_contract"
    ]
    assert manifest["browser_response_security_policy"]["headers"]["Referrer-Policy"] == "no-referrer"
    assert work_summary["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert work_summary["browser_response_security_policy"]["headers"]["Permissions-Policy"] == (
        "camera=(), microphone=(), geolocation=()"
    )
    assert work_summary["dataset_queue_direct_start_policy"]["enabled"] is True
    assert work_summary["dataset_queue_direct_start_policy"]["endpoint_route_template"] == (
        "/api/tasks/{task_id}/open"
    )
    assert work_summary["runtime_operator_validation_gate_cli_policy"][
        "preferred_require_flag"
    ] == "--require-operator-validation-for-browser-work"
    assert work_summary["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_start_open"
    ] is True
    assert work_summary["runtime_operator_validation_gate_cli_policy"][
        "protects_browser_mutations"
    ] is True
    assert work_summary["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_zarr_write"
    ] is True
    assert work_summary["browser_mutation_write_checklist"]["ready"] is True
    assert work_summary["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert work_summary["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert work_summary["browser_mutation_write_checklist"] == work_summary["work"][
        "browser_mutation_write_checklist"
    ]
    assert work_summary["browser_mutation_write_policy"]["browser_has_direct_zarr_write_authority"] is False
    assert work_summary["labeler_route_authorization_checklist"] == work_summary["work"][
        "labeler_route_authorization_checklist"
    ]
    assert work_summary["labeler_route_authorization_checklist"]["ready"] is True
    assert work_summary["labeler_route_authorization_checklist"][
        "expected_user_matches_resolved_user"
    ] is True
    assert work_summary["labeler_route_authorization_checklist"][
        "known_assignment_store_user"
    ] is True
    assert work_summary["labeler_route_authorization_checklist"][
        "mutation_requires_current_target_token"
    ] is True
    assert dataset_queue["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert dataset_queue["browser_response_security_policy"]["clickjacking_protection"] is True
    assert validation_checklist["dataset_queue_state"]["code"] == "has_open_dataset_work"
    assert validation_checklist["dataset_queue_blocked_start_users"] == []
    assert validation_checklist["browser_response_security_policy"]["proxy_must_preserve_headers"] is True
    assert validation_gate_statuses["dataset_queue_start_readiness"] == "passed"
    assert validation_gate_statuses["browser_response_security_headers"] == "pending_operator_evidence"
    assert validation_checklist["ready_for_operator_validation"] is True
    assert validation_checklist["all_validation_complete"] is False
    assert manifest["operator_validation_required_before_invite"] is True
    assert manifest["operator_validation_all_complete"] is False
    assert manifest["operator_validation_status"] == "pending_operator_evidence"
    assert "browser_response_security_headers" in manifest["operator_validation_pending_gate_ids"]
    assert manifest["operator_validation_needs_review_gate_ids"] == []
    assert manifest["ready_to_send"] is False
    assert manifest["sendability_reasons"] == ["operator_validation_pending"]
    assert "Complete required operator validation evidence" in manifest["sendability_actions"][0]
    assert manifest["sendability_warnings"]
    assert manifest["labeler_safety_policy_present"] is True
    assert manifest["labeler_safety_ready"] is True
    assert manifest["labeler_safety_readiness"] == "passed"
    assert manifest["labeler_safety_expected_user_guards_ready"] is True
    assert manifest["labeler_safety_browser_payload_redaction_ready"] is True
    assert manifest["labeler_route_authorization_policy_present"] is True
    assert manifest["labeler_route_authorization_ready"] is True
    assert manifest["labeler_route_authorization_readiness"] == "passed"
    assert manifest["labeler_route_authorization_active_assignment_required"] is True
    assert manifest["labeler_route_authorization_checklist"]["ready"] is True
    assert manifest["labeler_route_authorization_checklist"][
        "expected_user_matches_resolved_user"
    ] is True
    assert manifest["labeler_route_authorization_checklist"][
        "known_assignment_store_user"
    ] is True
    assert manifest["labeler_route_authorization_checklist"][
        "task_open_requires_active_assignment"
    ] is True
    assert manifest["labeler_route_authorization_checklist"][
        "mutation_requires_current_target_token"
    ] is True
    assert manifest["signed_link_policy_present"] is True
    assert manifest["signed_link_policy_ready"] is True
    assert manifest["signed_link_policy_readiness"] == "passed"
    assert manifest["signed_link_authorization_grant"] is False
    assert manifest["signed_link_forwarded_links_recheck_identity"] is True
    assert manifest["signed_link_runtime_operator_validation_start_gate_enforced"] is True
    assert (
        manifest["signed_link_operator_validation_start_gate_checked_before_session_create"]
        is True
    )
    assert manifest["session_guard_policy_present"] is True
    assert manifest["session_guard_policy_ready"] is True
    assert manifest["session_guard_policy_readiness"] == "passed"
    assert manifest["session_guard_stale_tab_save_rejected"] is True
    assert manifest["session_guard_rejects_after_reassignment"] is True
    assert manifest["session_guard_non_startable_task_sessions_rejected"] is True
    assert manifest["task_state_policy_present"] is True
    assert manifest["task_state_policy_ready"] is True
    assert manifest["task_state_policy_readiness"] == "passed"
    assert manifest["task_state_startable_task_states"] == '["pending", "in_progress"]'
    assert manifest["task_state_completed_tasks_read_only"] is True
    assert manifest["task_state_non_startable_task_open_requests"] == "reject_task_not_startable"
    assert manifest["task_state_non_startable_task_save_requests"] == "reject_task_not_startable"
    assert manifest["task_state_operator_reopen_required_before_more_labeling"] is True
    assert manifest["task_state_requires_current_target_token"] is True
    assert manifest["zarr_backup_policy_present"] is True
    assert manifest["zarr_backup_ready"] is True
    assert manifest["zarr_backup_readiness"] == "passed"
    assert manifest["zarr_backup_copy_before_labeling"] is True
    assert manifest["zarr_backup_labelers_do_not_receive_backup_paths"] is True
    assert manifest["zarr_backup_rollback_owner"] == "operator"
    assert manifest["mutation_audit_policy_present"] is True
    assert manifest["mutation_audit_ready"] is True
    assert manifest["mutation_audit_readiness"] == "passed"
    assert manifest["mutation_audit_event_store"] == "labeling_task_events"
    assert manifest["mutation_audit_server_records_events"] is True
    assert manifest["mutation_audit_browser_records_events_directly"] is False
    assert manifest["mutation_audit_required_event_fields_present"] is True
    assert manifest["browser_response_security_policy_present"] is True
    assert manifest["browser_response_security_ready"] is True
    assert manifest["browser_response_security_readiness"] == "passed"
    assert manifest["browser_response_security_no_store_cache"] is True
    assert manifest["browser_response_security_clickjacking_protection"] is True
    assert manifest["browser_response_security_proxy_must_preserve_headers"] is True
    assert manifest["browser_mutation_write_policy_present"] is True
    assert manifest["browser_mutation_write_ready"] is True
    assert manifest["browser_mutation_write_readiness"] == "passed"
    assert manifest["progress_summary"]["waiting_recording_count"] == 1
    assert manifest["progress_summary"]["blocked_recording_count"] == 1
    assert manifest["dataset_queue_summary"]["waiting_dataset_count"] == 1
    assert manifest["counts"]["waiting_datasets"] == 1
    assert manifest["counts"]["dataset_open_tasks"] == 1
    assert manifest["dataset_queue_preview_url"] == "https://labeling.example.org/my-datasets?expected_user=alice"
    assert manifest["canonical_dataset_queue_preview_url"] == "https://labeling.example.org/datasets?expected_user=alice"
    assert manifest["personalized_launch_readiness"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert manifest["personalized_launch_readiness"][
        "browser_writes_csv_or_handoff_files"
    ] is False
    assert manifest["personalized_launch_readiness"][
        "browser_has_direct_zarr_write_authority"
    ] is False
    assert work_summary["work"]["personalized_launch_readiness"][
        "browser_label_write_target"
    ] == "training_zarr"
    assert manifest_file["operator_validation_required_before_invite"] is True
    assert manifest_file["operator_validation_all_complete"] is False
    assert manifest_file["operator_validation_status"] == "pending_operator_evidence"
    assert "browser_response_security_headers" in manifest_file["operator_validation_pending_gate_ids"]
    assert manifest_file["operator_validation_needs_review_gate_ids"] == []
    assert manifest_file["ready_to_send"] is False
    assert manifest_file["personalized_launch_readiness"][
        "browser_label_write_target"
    ] == "training_zarr"
    assert manifest_file["personalized_launch_readiness"][
        "browser_writes_csv_or_handoff_files"
    ] is False
    assert manifest_file["personalized_launch_readiness"][
        "browser_has_direct_zarr_write_authority"
    ] is False
    assert manifest_file["known_user_status"]["is_known_labeler"] is True
    assert manifest_file["known_user_status"]["active_assignment_count"] == 2
    assert manifest_file["labeler_route_authorization_checklist"] == manifest[
        "labeler_route_authorization_checklist"
    ]
    assert manifest_file["labeler_landing_page_path"] == "/"
    assert manifest_file["labeler_landing_url"] == "https://labeling.example.org"
    assert manifest_file["expected_user_labeler_landing_url"] == "https://labeling.example.org?expected_user=alice"
    assert manifest_file["dashboard_path"] == "/work"
    assert manifest_file["dashboard_url"] == "https://labeling.example.org/work"
    assert manifest_file["expected_user_dashboard_url"] == "https://labeling.example.org/work?expected_user=alice"
    assert manifest_file["expected_user_dataset_queue_url"] == "https://labeling.example.org/datasets?expected_user=alice"
    assert manifest_file["expected_user_identity_probe_url"] == "https://labeling.example.org/identity?expected_user=alice"
    assert manifest_file["preferred_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert manifest_file["preferred_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert manifest_file["preferred_labeler_entry_url_matches_dataset_queue"] is True
    assert manifest_file["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert manifest_file["guarded_links_ready"] is True
    assert manifest_file["handoff_artifacts_ready"] is True
    assert manifest_file["handoff_entry_readiness"] == "passed"
    assert manifest_file["labeler_safety_policy_present"] is True
    assert manifest_file["labeler_safety_ready"] is True
    assert manifest_file["labeler_safety_readiness"] == "passed"
    assert manifest_file["labeler_route_authorization_policy_present"] is True
    assert manifest_file["labeler_route_authorization_ready"] is True
    assert manifest_file["labeler_route_authorization_readiness"] == "passed"
    assert manifest_file["labeler_route_authorization_active_assignment_required"] is True
    assert manifest_file["signed_link_policy_present"] is True
    assert manifest_file["signed_link_policy_ready"] is True
    assert manifest_file["signed_link_policy_readiness"] == "passed"
    assert manifest_file["session_guard_policy_present"] is True
    assert manifest_file["session_guard_policy_ready"] is True
    assert manifest_file["session_guard_policy_readiness"] == "passed"
    assert manifest_file["task_state_policy_present"] is True
    assert manifest_file["task_state_policy_ready"] is True
    assert manifest_file["task_state_policy_readiness"] == "passed"
    assert manifest_file["task_state_completed_tasks_read_only"] is True
    assert manifest_file["task_state_requires_current_target_token"] is True
    assert manifest_file["zarr_backup_policy_present"] is True
    assert manifest_file["zarr_backup_ready"] is True
    assert manifest_file["zarr_backup_readiness"] == "passed"
    assert manifest_file["zarr_backup_copy_before_labeling"] is True
    assert manifest_file["zarr_backup_labelers_do_not_receive_backup_paths"] is True
    assert manifest_file["mutation_audit_policy_present"] is True
    assert manifest_file["mutation_audit_ready"] is True
    assert manifest_file["mutation_audit_readiness"] == "passed"
    assert manifest_file["mutation_audit_event_store"] == "labeling_task_events"
    assert manifest_file["mutation_audit_server_records_events"] is True
    assert manifest_file["mutation_audit_browser_records_events_directly"] is False
    assert manifest_file["browser_response_security_policy_present"] is True
    assert manifest_file["browser_response_security_ready"] is True
    assert manifest_file["browser_response_security_readiness"] == "passed"
    assert manifest_file["browser_response_security_no_store_cache"] is True
    assert manifest_file["browser_response_security_clickjacking_protection"] is True
    assert manifest_file["browser_mutation_write_policy_present"] is True
    assert manifest_file["browser_mutation_write_ready"] is True
    assert manifest_file["browser_mutation_write_readiness"] == "passed"
    assert manifest_file["labeler_safety"]["dashboard_path"] == "/work"
    assert manifest_file["labeler_safety"]["dataset_queue_page_path"] == "/datasets"
    assert manifest_file["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert manifest_file["sendability_reasons"] == ["operator_validation_pending"]
    assert "Complete required operator validation evidence" in manifest_file["sendability_actions"][0]
    assert manifest_file["sendability_warnings"]
    assert manifest_file["files"]["validation_log"] == str(output_dir / "validation-log-template.md")
    assert manifest_file["files"]["validation_checklist"] == str(output_dir / "validation-checklist.json")
    assert manifest_file["files"]["dataset_queue"] == str(output_dir / "dataset-queue.json")
    assert manifest_file["progress_summary"] == manifest["progress_summary"]
    assert manifest_file["dataset_queue_summary"] == manifest["dataset_queue_summary"]
    assert manifest_file["counts"]["recordings"] == 2
    assert manifest_file["counts"]["tasks"] == 1
    assert manifest_file["counts"]["signed_links"] == 1
    assert manifest_file["counts"]["ready_to_share_links"] == 1
    assert manifest_file["counts"]["recordings_without_open_tasks"] == 1
    assert manifest_file["counts"]["recordings_without_open_tasks_by_reason"] == {"tasks_not_generated": 1}
    assert "Generate or import browser-labeling tasks" in manifest_file["counts"][
        "recordings_without_open_tasks_actions"
    ][0]
    assert manifest_file["counts"]["redacted_summary_fields"] == 1
    assert work_summary["work"]["user"] == "alice"
    assert work_summary["work"]["recordings"][0]["recording_id"] == "rec-a"
    assert work_summary["work"]["recordings"][1]["recording_id"] == "rec-empty"
    assert work_summary["work"]["recordings"][1]["tasks"] == []
    assert work_summary["work"]["progress_summary"] == manifest["progress_summary"]
    assert dataset_queue["schema"] == "palette.web_labeling_dataset_queue.v1"
    assert dataset_queue["user"] == "alice"
    assert dataset_queue["empty_state"] == work_summary["work"]["empty_state"]
    assert dataset_queue["progress_summary"] == manifest["progress_summary"]
    assert dataset_queue["dataset_queue_summary"] == manifest["dataset_queue_summary"]
    assert dataset_queue["direct_browser_start_contract_summary"] == work_summary["work"][
        "direct_browser_start_contract_summary"
    ]
    assert dataset_queue["direct_browser_start_contract_summary"]["ready"] is True
    assert dataset_queue["direct_browser_start_contract_summary"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert dataset_queue["direct_browser_start_contract_summary"][
        "browser_has_direct_zarr_write_authority"
    ] is False
    assert dataset_queue["dataset_queue_state"]["schema"] == "palette.web_labeling_dataset_queue_state.v1"
    assert dataset_queue["dataset_queue_state"]["code"] == "has_open_dataset_work"
    assert dataset_queue["dataset_queue_state"]["has_open_dataset_work"] is True
    assert dataset_queue["dataset_queue_state"] == manifest["dataset_queue_state"]
    assert dataset_queue["labeler_work_completion"] == manifest["labeler_work_completion"]
    assert dataset_queue["labeler_work_completion"]["schema"] == (
        "palette.web_labeling_labeler_work_completion.v1"
    )
    assert dataset_queue["labeler_work_completion"]["status"] == "waiting"
    assert dataset_queue["labeler_work_completion"]["has_waiting_work"] is True
    assert dataset_queue["labeler_work_completion"]["completed"] is False
    assert dataset_queue["labeler_work_completion_ready_for_more_labeling"] is True
    assert dataset_queue["labeler_work_completion_waiting_dataset_count"] == 1
    assert work_summary["work"]["labeler_start_ready"] is True
    assert work_summary["work"]["labeler_start_status"] == "has_open_dataset_work"
    assert work_summary["work"]["labeler_action"] == "open_dataset_queue"
    assert work_summary["work"]["dataset_queue_direct_start_policy"] == work_summary[
        "dataset_queue_direct_start_policy"
    ]
    assert work_summary["work"]["runtime_operator_validation_gate_cli_policy"] == work_summary[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert work_summary["work"]["browser_mutation_write_checklist"] == work_summary[
        "browser_mutation_write_checklist"
    ]
    assert dataset_queue["labeler_start_ready"] is True
    assert dataset_queue["labeler_start_status"] == "has_open_dataset_work"
    assert dataset_queue["labeler_action"] == "open_dataset_queue"
    assert dataset_queue["labeler_start_message"] == dataset_queue["dataset_queue_state"]["message"]
    assert dataset_queue["labeler_start_operator_action"] == ""
    assert dataset_queue["dataset_queue_direct_start_policy"]["enabled"] is True
    assert dataset_queue["dataset_queue_direct_start_policy"]["method"] == "POST"
    assert dataset_queue["dataset_queue_direct_start_policy"]["endpoint_route_template"] == (
        "/api/tasks/{task_id}/open"
    )
    assert dataset_queue["dataset_queue_direct_start_policy"]["same_origin_only"] is True
    assert dataset_queue["dataset_queue_direct_start_policy"]["exact_route_required"] is True
    assert dataset_queue["dataset_queue_direct_start_policy"][
        "endpoint_task_segment_must_match_row_task_id"
    ] is True
    assert dataset_queue["dataset_queue_direct_start_policy"][
        "non_startable_tasks_do_not_advertise_endpoint"
    ] is True
    assert manifest["dataset_queue_direct_start_policy"] == dataset_queue[
        "dataset_queue_direct_start_policy"
    ]
    assert manifest["runtime_operator_validation_gate_cli_policy"] == dataset_queue[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert dataset_queue["runtime_operator_validation_gate_cli_policy"][
        "blocks_before_audit_event_creation"
    ] is True
    assert manifest["labeler_start_ready"] is True
    assert manifest["labeler_start_status"] == "has_open_dataset_work"
    assert manifest["labeler_action"] == "open_dataset_queue"
    assert dataset_queue["dataset_queue_preview_url"] == manifest["dataset_queue_preview_url"]
    assert dataset_queue["expected_user_labeler_landing_url"] == manifest["expected_user_labeler_landing_url"]
    assert dataset_queue["expected_user_dataset_queue_url"] == manifest["expected_user_dataset_queue_url"]
    assert dataset_queue["known_user_status"] == manifest["known_user_status"]
    assert dataset_queue["dataset_queue"] == work_summary["work"]["dataset_queue"]
    assert dataset_queue["datasets"] == work_summary["work"]["dataset_queue"]
    queue_dataset = dataset_queue["dataset_queue"][0]
    queue_recording = queue_dataset["recordings"][0]
    queue_task = queue_recording["tasks"][0]
    assert queue_dataset["labeler_start_ready"] is True
    assert queue_dataset["labeler_action"] == "open_dataset"
    assert queue_dataset["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert queue_dataset["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert queue_dataset["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert queue_dataset["browser_label_write_target"] == "training_zarr"
    assert queue_dataset["training_zarr_mutations_are_server_owned"] is True
    assert queue_dataset["handoff_artifacts_are_metadata_only"] is True
    assert queue_dataset["browser_writes_csv_or_handoff_files"] is False
    assert queue_dataset["browser_writes_handoff_csv"] is False
    assert queue_dataset["browser_writes_intermediate_csv"] is False
    assert queue_dataset["browser_receives_zarr_write_authority"] is False
    assert queue_dataset["browser_has_direct_zarr_write_authority"] is False
    assert queue_recording["labeler_start_ready"] is True
    assert queue_recording["labeler_action"] == "open_recording"
    assert queue_recording["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert queue_recording["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert queue_recording["training_zarr_mutations_are_server_owned"] is True
    assert queue_recording["browser_receives_zarr_write_authority"] is False
    assert queue_recording["browser_has_direct_zarr_write_authority"] is False
    assert queue_task["labeler_start_ready"] is True
    assert queue_task["labeler_action"] == "open_task"
    assert queue_task["direct_browser_start_endpoint"] == "/api/tasks/task-a/open"
    assert queue_task["direct_browser_start_method"] == "POST"
    assert queue_task["direct_browser_start_uses_existing_task_open_api"] is True
    assert queue_task["direct_browser_start_requires_expected_user_guard"] is True
    work_task = work_summary["work"]["recordings"][0]["tasks"][0]
    assert work_task["direct_browser_start_endpoint"] == "/api/tasks/task-a/open"
    assert work_task["direct_browser_start_authorization_contract_ready"] is True
    assert work_task["direct_browser_start_not_ready_reason"] == ""
    assert work_task["direct_browser_start_authorization_contract"]["ready"] is True
    assert work_task["direct_browser_start_authorization_contract"]["expected_user"] == "alice"
    assert work_task["direct_browser_start_authorization_contract"][
        "expected_user_guard_enforced_by_api"
    ] is True
    assert work_task["direct_browser_start_authorization_contract"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert work_task["direct_browser_start_authorization_contract"][
        "browser_has_direct_zarr_write_authority"
    ] is False
    assert work_task["operator_support"]["direct_browser_start_authorization_contract_ready"] is True
    assert work_task["operator_support"]["direct_browser_start_not_ready_reason"] == ""
    assert work_task["operator_support"]["direct_browser_start_not_ready_reasons"] == []
    assert work_task["operator_support"]["direct_browser_start_operator_action"] == ""
    assert work_task["operator_support"]["direct_browser_start_expected_user_guard_required"] is True
    assert work_task["operator_support"]["direct_browser_start_expected_user_guard_enforced_by_api"] is True
    assert work_task["operator_support"]["direct_browser_start_server_rechecks_on_post"] is True
    assert work_task["operator_support"]["browser_label_write_target"] == "training_zarr"
    assert work_task["operator_support"]["browser_writes_csv_or_handoff_files"] is False
    assert work_task["operator_support"]["browser_has_direct_zarr_write_authority"] is False
    direct_start_summary = work_summary["work"]["direct_browser_start_contract_summary"]
    assert direct_start_summary["schema"] == (
        "palette.web_labeling_direct_browser_start_contract_summary.v1"
    )
    assert direct_start_summary["ready"] is True
    assert direct_start_summary["task_count"] >= 1
    assert direct_start_summary["ready_task_count"] >= 1
    assert direct_start_summary["expected_user_guard_enforced_by_api"] is True
    assert direct_start_summary["server_rechecks_on_post"] is True
    assert direct_start_summary["browser_label_write_target"] == "training_zarr"
    assert direct_start_summary["csv_handoff_artifacts_are_label_write_targets"] is False
    assert direct_start_summary["browser_writes_csv_or_handoff_files"] is False
    assert direct_start_summary["browser_has_direct_zarr_write_authority"] is False
    assert queue_task["operator_support"]["direct_browser_start_endpoint"] == "/api/tasks/task-a/open"
    assert queue_task["operator_support"]["direct_browser_start_method"] == "POST"
    assert queue_task["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert queue_task["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert queue_task["browser_label_write_target"] == "training_zarr"
    assert queue_task["training_zarr_mutations_are_server_owned"] is True
    assert queue_task["handoff_artifacts_are_metadata_only"] is True
    assert queue_task["browser_writes_csv_or_handoff_files"] is False
    assert queue_task["browser_writes_handoff_csv"] is False
    assert queue_task["browser_writes_intermediate_csv"] is False
    assert queue_task["browser_receives_zarr_write_authority"] is False
    assert queue_task["browser_has_direct_zarr_write_authority"] is False
    assert dataset_queue["zarr_backup_policy"]["validation_gate"] == "mutable_zarr_backup_confirmation"
    assert dataset_queue["zarr_backup_policy"]["labelers_do_not_receive_backup_paths"] is True
    assert dataset_queue["mutation_audit_policy"]["event_store"] == "labeling_task_events"
    assert dataset_queue["mutation_audit_policy"]["server_records_events"] is True
    assert dataset_queue["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert dataset_queue["browser_mutation_write_policy"]["handoff_artifacts_are_metadata_only"] is True
    assert dataset_queue["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert dataset_queue["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert dataset_queue["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert dataset_queue["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert dataset_queue["browser_mutation_write_policy"]["browser_has_direct_zarr_write_authority"] is False
    assert dataset_queue["browser_mutation_write_checklist"]["schema"] == (
        "palette.web_labeling_browser_mutation_write_checklist.v1"
    )
    assert dataset_queue["browser_mutation_write_checklist"]["ready"] is True
    assert dataset_queue["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert dataset_queue["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert dataset_queue["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert dataset_queue["browser_mutation_write_checklist"]["browser_has_direct_zarr_write_authority"] is False
    assert dataset_queue["labeler_route_authorization_policy"]["expected_user_must_match_resolved_user"] is True
    assert dataset_queue["labeler_route_authorization_policy"]["known_assignment_store_user_required"] is True
    assert dataset_queue["labeler_route_authorization_policy"]["task_open_requires_active_assignment"] is True
    assert dataset_queue["labeler_route_authorization_policy"]["task_open_requires_startable_task_state"] is True
    assert dataset_queue["labeler_route_authorization_policy"]["startable_task_states"] == [
        "pending",
        "in_progress",
    ]
    assert dataset_queue["labeler_route_authorization_checklist"]["schema"] == (
        "palette.web_labeling_labeler_route_authorization_runtime_checklist.v1"
    )
    assert dataset_queue["labeler_route_authorization_checklist"]["ready"] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "expected_user_must_match_resolved_user"
    ] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "expected_user_matches_resolved_user"
    ] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "known_assignment_store_user"
    ] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "task_open_requires_active_assignment"
    ] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "mutation_requires_current_target_token"
    ] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "signed_links_are_entry_hints_not_authorization"
    ] is True
    assert dataset_queue["labeler_route_authorization_checklist"][
        "forwarded_expected_user_links_recheck_identity"
    ] is True
    assert dataset_queue["labeler_route_authorization_policy"]["signed_links_are_entry_hints_not_authorization"] is True
    assert dataset_queue["session_guard_policy"]["stale_tab_save_rejected"] is True
    assert dataset_queue["session_guard_policy"]["target_token_required_for_mutation"] is True
    assert dataset_queue["session_guard_policy"]["labeler_promotion_retry_requires_current_session"] is True
    assert dataset_queue["session_guard_policy"]["session_closure_event_support"] is True
    summary_task = work_summary["work"]["recordings"][0]["tasks"][0]
    assert summary_task["task_id"] == "task-a"
    assert "scope" not in summary_task
    assert summary_task["redacted_fields"] == ["scope"]
    assert "/secret/alice-training.zarr" not in json.dumps(work_summary)
    assert "/secret/alice-training.zarr" not in html_index
    assert links[0]["task_id"] == "task-a"
    assert links[0]["url"].startswith("https://labeling.example.org/t/")
    assert links[0]["url_is_absolute"] is True
    assert links[0]["task_launchable"] is True
    assert links[0]["ready_to_share"] is True
    assert links[0]["shareability_warnings"] == []
    assert links[0]["issued_at_utc"] == manifest["generated_at_utc"]
    assert links[0]["expires_at_utc"] == manifest["links_expire_at_utc"]
    assert check["ok"] is True
    assert "Your Palette labeling work" in html_index
    assert "Waiting recordings: 1" in html_index
    assert "Blocked/no-open recordings: 1" in html_index
    assert "labeler-quickstart.txt" in html_index
    assert "validation-log-template.md" in html_index
    assert "Alice task" in html_index
    assert "Task-specific instructions" in html_index
    assert "rec-empty" in html_index
    assert "no browser-labeling tasks have been generated yet" in html_index
    assert "No open signed task link" in html_index
    assert "<th>Priority</th>" in html_index
    assert "<td>7</td>" in html_index
    assert "https://labeling.example.org/t/" in html_index
    assert "confirm the dashboard shows you as <b>alice</b>" in html_index
    assert "Preview your datasets-waiting landing page" in html_index
    assert "https://labeling.example.org?expected_user=alice" in html_index
    assert "Preview your personalized dataset queue" in html_index
    assert "https://labeling.example.org/datasets?expected_user=alice" in html_index
    assert "Dataset queue state:" in html_index
    assert "has_open_dataset_work" in html_index
    assert "Labeler start allowed" in html_index
    assert "https://labeling.example.org/work?expected_user=alice" in html_index
    assert "Your Palette labeling work is ready." not in message
    assert "Your Palette labeling handoff needs operator review before starting." in message
    assert "Identity check:" in message
    assert "https://labeling.example.org/identity?expected_user=alice" in message
    assert "confirm the dashboard shows you as alice" in message
    assert "No Palette or Crimson installation is needed" in message
    assert "Dataset queue state: has_open_dataset_work" in message
    assert "Dataset queue start: allowed" in message
    assert "Preview queue-first entry point:" in message
    assert "Start here:" in message
    assert "Queue-first start page:" in message
    assert "https://labeling.example.org?expected_user=alice" in message
    assert "https://labeling.example.org/my-datasets?expected_user=alice" in message
    assert "https://labeling.example.org/datasets?expected_user=alice" in message
    assert "https://labeling.example.org/work?expected_user=alice" in message
    assert "Links expire at UTC:" in message
    assert "still require authenticated access" in message
    assert "Do not edit zarr files directly" in quickstart
    assert "Wait for operator review before starting." in quickstart
    assert "Dataset queue state: has_open_dataset_work" in quickstart
    assert "Dataset queue start: allowed" in quickstart
    assert "Open the identity check and confirm it reports you as alice" in quickstart
    assert "https://labeling.example.org/identity?expected_user=alice" in quickstart
    assert "Open your datasets-waiting landing page" in quickstart
    assert "https://labeling.example.org?expected_user=alice" in quickstart
    assert "https://labeling.example.org/my-datasets?expected_user=alice" in quickstart
    assert "https://labeling.example.org/datasets?expected_user=alice" in quickstart
    assert "Canonical dataset queue fallback" in quickstart
    assert "preferred queue-first view" in quickstart
    assert "Confirm the dashboard shows you as alice" in quickstart
    assert "https://labeling.example.org/work?expected_user=alice" in quickstart
    assert "# Web Labeling Validation Log" in validation_log
    assert "single-user handoff bundle" in validation_log
    assert "## Recording Evidence" in validation_log
    assert "update-validation-checklist" in validation_log
    assert f"--path {output_dir / 'validation-checklist.json'}" in validation_log
    assert f"--append-log {output_dir / 'validation-log-template.md'}" in validation_log
    assert "Operator Authorization Boundary" in validation_log
    assert "Browser Response Security Headers" in validation_log
    assert "`Cache-Control` preserved" in validation_log
    assert "Identity Probe Links" in validation_log
    assert "Dataset Queue Start Readiness" in validation_log
    assert "dataset_queue_blocked_start_users" in validation_log
    assert "Assignment Transition Evidence" in validation_log
    assert "- User: alice" in validation_log
    assert f"- Manifest: {output_dir / 'manifest.json'}" in validation_log
    assert f"- HTML index: {output_dir / 'index.html'}" in validation_log
    gate_statuses = {gate["id"]: gate["status"] for gate in validation_checklist["gates"]}
    assert validation_checklist["schema"] == "palette.web_labeling_validation_checklist.v1"
    assert validation_checklist["bundle_label"] == "single-user handoff bundle"
    assert validation_checklist["validation_log"] == str(output_dir / "validation-log-template.md")
    assert validation_checklist["labeler_landing_page_path"] == "/"
    assert validation_checklist["expected_user_labeler_landing_url"] == (
        "https://labeling.example.org?expected_user=alice"
    )
    assert validation_checklist["dataset_queue_page_path"] == "/datasets"
    assert validation_checklist["expected_user_dataset_queue_url"] == (
        "https://labeling.example.org/datasets?expected_user=alice"
    )
    assert validation_checklist["personal_dataset_queue_page_path"] == "/my-datasets"
    assert validation_checklist["personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets"
    )
    assert validation_checklist["expected_user_personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert validation_checklist["personal_work_page_path"] == "/my-work"
    assert validation_checklist["personal_work_url"] == "https://labeling.example.org/my-work"
    assert validation_checklist["expected_user_personal_work_url"] == (
        "https://labeling.example.org/my-work?expected_user=alice"
    )
    assert validation_checklist["personalized_labeler_entrypoint"] == (
        "personal_datasets_waiting_queue"
    )
    assert validation_checklist["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert validation_checklist["operator_validation_visibility_policy"] == manifest[
        "operator_validation_visibility_policy"
    ]
    assert validation_checklist["operator_validation_command_templates"]["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert validation_checklist["safe_share_gate"]["schema"] == (
        "palette.web_labeling_safe_share_gate.v1"
    )
    assert validation_checklist["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert validation_checklist["safe_share_ready_to_send_is_sufficient"] is False
    assert validation_checklist["safe_share_required_inspection_field"] == (
        "labeler_links_safe_to_share"
    )
    assert "disposable_zarr_mutation_smoke" in validation_checklist[
        "safe_share_launch_blocking_evidence_gate_ids"
    ]
    assert "operator_validation_record_command_ids" in validation_checklist[
        "safe_share_launch_blocking_next_action_detail_fields"
    ]
    assert "operator_validation_evidence_template_path" in validation_checklist[
        "safe_share_launch_blocking_next_action_command_fields"
    ]
    assert "record_browser_smoke_evidence" in validation_checklist[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert validation_checklist["operator_validation_command_templates"][
        "launch_evidence_collection_plan_schema"
    ] == "palette.web_labeling_launch_evidence_collection_plan.v1"
    assert validation_checklist["operator_validation_command_templates"][
        "launch_evidence_collection_plan"
    ]["required_final_field"] == "labeler_links_safe_to_share"
    assert validation_checklist["operator_validation_command_templates"][
        "launch_evidence_collection_plan"
    ]["steps_by_gate_id"]["disposable_zarr_mutation_smoke"]["record_command_id"] == (
        "record_disposable_zarr_mutation_smoke_evidence"
    )
    assert "apply_operator_evidence_templates" in validation_checklist[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert "final_signoff" in validation_checklist[
        "operator_validation_command_templates"
    ]["missing_command_gate_ids"]
    assert validation_checklist["operator_validation_command_templates"]["commands_by_gate_id"][
        "browser_smoke"
    ] == [
        "record_browser_smoke_evidence",
        "apply_operator_evidence_templates",
    ]
    assert validation_checklist["queue_first_entry_contract"]["preferred_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert validation_checklist["queue_first_entry_contract"]["personal_dataset_queue_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["personal_work_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["personalized_labeler_entrypoint"] == (
        "personal_datasets_waiting_queue"
    )
    assert validation_checklist["queue_first_entry_contract"]["personalized_entry_required"] is True
    assert validation_checklist["queue_first_entry_contract"]["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert validation_checklist["queue_first_entry_contract"][
        "personalized_labeler_entry_url_matches_personal_dataset_queue"
    ] is True
    assert validation_checklist["queue_first_entry_contract"][
        "personalized_labeler_entry_url_is_expected_user_guarded"
    ] is True
    assert validation_checklist["queue_first_entry_contract"]["preferred_labeler_entry_url_is_expected_user_guarded"] is True
    assert validation_checklist["expected_user_identity_probe_url"] == (
        "https://labeling.example.org/identity?expected_user=alice"
    )
    assert validation_checklist["single_owner_policy"]["one_active_owner"] is True
    assert validation_checklist["queue_first_entry_contract"]["ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["landing_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["labeling_home_ready"] is True
    assert validation_checklist["labeling_home_page_path"] == "/labeling"
    assert validation_checklist["expected_user_labeling_home_url"] == validation_checklist[
        "queue_first_entry_contract"
    ]["expected_user_labeling_home_url"]
    assert validation_checklist["queue_first_entry_contract"]["dataset_queue_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["queue_first_paths_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["labeler_landing_page_kind"] == (
        "datasets_waiting_queue"
    )
    assert validation_checklist["queue_first_entry_contract"]["landing_serves_datasets_waiting_queue"] is True
    assert validation_checklist["queue_first_entry_contract"]["datasets_waiting_alias_paths"] == [
        "/",
        "/me",
        "/labeling",
        "/datasets",
        "/my-datasets",
    ]
    assert validation_checklist["queue_first_entry_contract"]["datasets_waiting_aliases_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["dashboard_is_fallback"] is True
    assert validation_checklist["queue_first_entry_contract"]["preferred_labeler_entrypoint"] == (
        "personal_datasets_waiting_queue"
    )
    assert (
        validation_checklist["queue_first_entry_contract"][
            "preferred_labeler_entry_url_matches_dataset_queue"
        ]
        is True
    )
    assert (
        validation_checklist["queue_first_entry_contract"]["personal_dataset_queue_link_role"]
        == "preferred_queue"
    )
    assert (
        validation_checklist["queue_first_entry_contract"]["dataset_queue_link_role"]
        == "canonical_queue_fallback"
    )
    assert (
        validation_checklist["queue_first_entry_contract"]["canonical_dataset_queue_link_role"]
        == "canonical_queue_fallback"
    )
    assert validation_checklist["queue_first_entry_contract"]["dashboard_link_role"] == "fallback_dashboard"
    assert validation_checklist["queue_first_entry_contract"]["task_links_role"] == "convenience_entry_hints"
    assert validation_checklist["queue_first_entry_contract"]["identity_check_required"] is True
    assert validation_checklist["identity_probe_link_contract"]["ready"] is True
    assert validation_checklist["identity_probe_link_contract"]["identity_check_required"] is True
    assert validation_checklist["identity_probe_link_contract"]["expected_user_identity_probe_url_present"] is True
    assert validation_checklist["identity_probe_link_contract"]["batch_identity_probe_evidence_present"] is False
    assert validation_checklist["identity_probe_link_contract"]["operator_verification_still_required"] is True
    assert validation_checklist["labeler_safety"]["labeler_runtime_surface"] == "browser"
    assert validation_checklist["labeler_safety"]["requires_local_palette_installation"] is False
    assert validation_checklist["labeler_safety"]["requires_local_crimson_installation"] is False
    assert validation_checklist["labeler_safety"]["requires_local_conda_environment"] is False
    assert validation_checklist["labeler_safety"]["requires_local_project_dependencies"] is False
    assert validation_checklist["labeler_safety"]["browser_receives_raw_zarr_paths"] is False
    assert validation_checklist["browser_payload_redaction_contract"]["ready"] is True
    assert validation_checklist["browser_payload_redaction_contract"]["browser_receives_raw_zarr_paths"] is False
    assert validation_checklist["browser_payload_redaction_contract"]["browser_receives_task_scope"] is False
    assert validation_checklist["browser_payload_redaction_contract"]["redacts_direct_storage_paths"] is True
    assert validation_checklist["browser_payload_redaction_contract"]["labeler_support_text_redacted"] is True
    assert validation_checklist["operator_authorization_policy"]["admin_routes_require_operator"] is True
    assert validation_checklist["operator_authorization_policy"]["operator_boundary_required_for_launch"] is True
    assert validation_checklist["operator_authorization_policy"]["operator_boundary_known"] is False
    assert validation_checklist["operator_authorization_policy"]["runtime_preflight_required"] is True
    assert validation_checklist["operator_authorization_contract"]["ready"] is True
    assert validation_checklist["operator_authorization_contract"]["admin_routes_require_operator"] is True
    assert validation_checklist["operator_authorization_contract"]["labelers_are_not_operators_by_default"] is True
    assert validation_checklist["operator_authorization_contract"]["operator_authorization_grants_labeler_mutation"] is False
    assert validation_checklist["labeler_route_authorization_policy"]["expected_user_must_match_resolved_user"] is True
    assert validation_checklist["labeler_route_authorization_policy"][
        "single_owner_store_proof_required_for_browser_work"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"]["ready"] is True
    assert validation_checklist["labeler_route_authorization_contract"]["known_assignment_store_user_required"] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_required_for_browser_work"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_requires_integrity_ok"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_requires_zero_duplicate_active_owners"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_requires_training_zarr_target"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_rejects_intermediate_csv_mutation"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "personal_work_page_expected_user_guarded"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "personal_dataset_queue_page_expected_user_guarded"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "personal_aliases_route_to_canonical_browser_surfaces"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"]["task_open_requires_startable_task_state"] is True
    assert validation_checklist["labeler_route_authorization_contract"]["startable_task_states"] == [
        "pending",
        "in_progress",
    ]
    assert validation_checklist["labeler_route_authorization_contract"]["forwarded_links_are_not_authorization_grants"] is True
    assert validation_checklist["signed_link_policy"]["authorization_grant"] is False
    assert validation_checklist["signed_link_contract"]["ready"] is True
    assert validation_checklist["signed_link_contract"]["binds_expected_user_in_new_links"] is True
    assert validation_checklist["signed_link_contract"]["signed_links_are_entry_hints_not_authorization"] is True
    assert validation_checklist["signed_link_contract"]["forwarded_signed_links_recheck_identity"] is True
    assert validation_checklist["signed_link_contract"]["runtime_operator_validation_start_gate_enforced"] is True
    assert validation_checklist["signed_link_contract"]["operator_validation_start_gate_checked_before_session_create"] is True
    assert validation_checklist["expected_user_guard_contract"]["ready"] is True
    assert validation_checklist["expected_user_guard_contract"]["missing_or_mismatched_guards"] == []
    assert "personal_work_page" in validation_checklist["expected_user_guard_contract"][
        "guarded_labeler_entrypoints"
    ]
    assert "personal_dataset_queue_page" in validation_checklist["expected_user_guard_contract"][
        "guarded_labeler_entrypoints"
    ]
    assert validation_checklist["expected_user_guard_contract"]["signed_links_expected_user_bound"] is True
    assert validation_checklist["expected_user_guard_contract"]["forwarded_links_stop_on_expected_user_mismatch"] is True
    assert validation_checklist["expected_user_guard_contract"]["promotion_retry_guarded_support_only"] is True
    assert validation_checklist["expected_user_guard_contract"]["promotion_retry_labeler_mutation_enabled"] is False
    assert validation_checklist["expected_user_guard_contract"]["promotion_retry_labeler_rejection_error"] == (
        "operator_support_required"
    )
    assert validation_checklist["session_guard_contract"]["ready"] is True
    assert validation_checklist["session_guard_contract"]["stale_tab_save_rejected"] is True
    assert validation_checklist["session_guard_contract"]["non_startable_task_sessions_rejected"] is True
    assert validation_checklist["session_guard_contract"]["rejects_after_reassignment"] is True
    assert validation_checklist["session_guard_contract"]["rejects_after_target_navigation"] is True
    assert validation_checklist["browser_response_security_policy"]["content_security_policy_scope"] == (
        "frame_ancestors_base_uri_form_action_object_src"
    )
    assert validation_checklist["browser_response_security_contract"]["ready"] is True
    assert validation_checklist["browser_response_security_contract"]["no_store_cache"] is True
    assert validation_checklist["browser_response_security_contract"]["clickjacking_protection"] is True
    assert validation_checklist["browser_response_security_contract"]["content_security_policy_scope_ready"] is True
    assert validation_checklist["browser_response_security_contract"]["permissions_policy_ready"] is True
    assert validation_checklist["task_state_policy"]["browser_mutation_target_token"] == "required_current_target_token"
    assert validation_checklist["task_state_contract"]["ready"] is True
    assert validation_checklist["task_state_contract"]["startable_task_states"] == ["pending", "in_progress"]
    assert validation_checklist["task_state_contract"]["completed_tasks_read_only"] is True
    assert validation_checklist["task_state_contract"]["completed_task_open_requests"] == "reject_task_complete"
    assert validation_checklist["task_state_contract"]["completed_task_save_requests"] == "reject_task_complete"
    assert validation_checklist["task_state_contract"]["non_startable_task_open_requests"] == "reject_task_not_startable"
    assert validation_checklist["task_state_contract"]["non_startable_task_save_requests"] == "reject_task_not_startable"
    assert validation_checklist["task_state_contract"]["labeler_promotion_retry_mutation_enabled"] is False
    assert validation_checklist["task_state_contract"]["labeler_promotion_retry_rejection_error"] == (
        "operator_support_required"
    )
    assert validation_checklist["task_state_contract"]["ordinary_labeler_promotion_retry_mutation"] == (
        "operator_support_required"
    )
    assert validation_checklist["task_state_contract"]["operator_reopen_required_before_more_labeling"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["ready"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["absolute_navigation_out_of_scope"] == "reject_nav_error"
    assert validation_checklist["browser_workflow_scope_contract"]["browser_mutation_target_selectors"] == (
        "server_owned_reject_client_fields"
    )
    assert {
        "target_zarr",
        "csv_path",
        "data_plane_write_target",
        "browser_label_write_target",
    }.issubset(
        set(validation_checklist["browser_workflow_scope_contract"]["target_selector_fields_rejected"])
    )
    assert validation_checklist["browser_workflow_scope_contract"]["target_indices_components_labels_frames_server_owned"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["workflow_contracts_server_owned_zarr_targets"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["workflow_contracts_training_zarr_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert validation_checklist["browser_workflow_scope_contract"]["workflow_training_zarr_write_modes"][
        "detect_training"
    ] == "direct"
    assert validation_checklist["browser_workflow_scope_contract"]["workflow_training_zarr_write_modes"][
        "detect_analysis"
    ] == "promotion_when_configured"
    assert validation_checklist["browser_workflow_scope_contract"]["workflow_contracts_csv_handoff_metadata_only"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["workflows_missing_server_owned_zarr_target"] == []
    assert validation_checklist["browser_workflow_scope_contract"]["workflows_missing_csv_metadata_contract"] == []
    assert validation_checklist["browser_workflow_scope_contract"]["workflows_with_csv_handoff_label_targets"] == []
    assert validation_checklist["browser_workflow_scope_contract"]["workflows_with_browser_csv_handoff_writes"] == []
    assert validation_checklist["browser_workflow_scope_contract"]["workflows_with_browser_zarr_authority"] == []
    assert validation_checklist["browser_mutation_target_contract"]["ready"] is True
    assert validation_checklist["browser_mutation_target_contract"]["workflow_contracts_require_target_token"] is True
    assert validation_checklist["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert validation_checklist["browser_mutation_write_policy"]["mutable_label_data_plane"] == (
        "task_scoped_training_zarr"
    )
    assert validation_checklist["browser_mutation_write_policy"]["training_zarr_mutations_are_server_owned"] is True
    assert validation_checklist["browser_mutation_write_policy"]["browser_has_direct_zarr_write_authority"] is False
    assert validation_checklist["browser_mutation_write_contract"]["ready"] is True
    assert validation_checklist["browser_mutation_write_contract"]["data_plane_write_target"] == (
        "server_owned_assigned_task_zarr_scope"
    )
    assert validation_checklist["browser_mutation_write_contract"]["mutable_label_data_plane"] == (
        "task_scoped_training_zarr"
    )
    assert validation_checklist["browser_mutation_write_contract"]["training_zarr_mutations_are_server_owned"] is True
    assert validation_checklist["browser_mutation_write_contract"]["handoff_artifacts_are_metadata_only"] is True
    assert validation_checklist["browser_mutation_write_contract"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert validation_checklist["browser_mutation_write_contract"]["browser_writes_csv_or_handoff_files"] is False
    assert validation_checklist["browser_mutation_write_contract"]["browser_writes_handoff_csv"] is False
    assert validation_checklist["browser_mutation_write_contract"]["browser_writes_intermediate_csv"] is False
    assert validation_checklist["browser_mutation_write_contract"]["browser_has_direct_zarr_write_authority"] is False
    assert validation_checklist["dataset_queue_direct_start_policy"]["enabled"] is True
    assert validation_checklist["dataset_queue_direct_start_policy"]["endpoint_route_template"] == (
        "/api/tasks/{task_id}/open"
    )
    assert validation_checklist["dataset_queue_direct_start_policy"][
        "endpoint_task_segment_must_match_row_task_id"
    ] is True
    assert validation_checklist["mutation_audit_contract"]["ready"] is True
    assert validation_checklist["mutation_audit_contract"]["event_store"] == "labeling_task_events"
    assert validation_checklist["mutation_audit_contract"]["server_records_events"] is True
    assert validation_checklist["mutation_audit_contract"]["browser_records_events_directly"] is False
    assert validation_checklist["mutation_audit_contract"]["required_event_fields_present"] is True
    assert validation_checklist["zarr_backup_contract"]["ready"] is True
    assert validation_checklist["zarr_backup_contract"]["operator_only"] is True
    assert validation_checklist["zarr_backup_contract"]["labelers_do_not_receive_backup_paths"] is True
    assert validation_checklist["zarr_backup_contract"]["rollback_owner"] == "operator"
    assert "identity_probe_verification" in validation_checklist["operator_evidence_gate_ids"]
    assert "browser_smoke" in validation_checklist["operator_evidence_gate_ids"]
    assert "static_readiness" in validation_checklist["generated_contract_gate_ids"]
    assert "queue_first_entry_contract" in validation_checklist["generated_contract_gate_ids"]
    assert "identity_probe_verification" in validation_checklist["operator_evidence_pending_gate_ids"]
    assert "browser_response_security_headers" in validation_checklist["operator_evidence_pending_gate_ids"]
    assert validation_checklist["operator_evidence_needs_review_gate_ids"] == []
    assert validation_checklist["generated_contract_failed_gate_ids"] == []
    assert validation_checklist["validation_gate_classification"]["operator_evidence_gate_ids"] == (
        validation_checklist["operator_evidence_gate_ids"]
    )
    assert validation_checklist["counts"]["browser_workflow_contracts_missing_target_token"] == 0
    assert validation_checklist["counts"]["operator_evidence_gates"] == len(
        validation_checklist["operator_evidence_gate_ids"]
    )
    assert validation_checklist["counts"]["generated_contract_gates"] == len(
        validation_checklist["generated_contract_gate_ids"]
    )
    assert validation_checklist["counts"]["operator_evidence_pending_gates"] == len(
        validation_checklist["operator_evidence_pending_gate_ids"]
    )
    assert validation_checklist["counts"]["operator_evidence_needs_review_gates"] == 0
    assert validation_checklist["counts"]["generated_contract_failed_gates"] == 0
    assert validation_checklist["counts"]["queue_first_entry_contract_ready"] == 1
    assert validation_checklist["counts"]["identity_probe_link_contract_ready"] == 1
    assert validation_checklist["counts"]["browser_payload_redaction_contract_ready"] == 1
    assert validation_checklist["counts"]["operator_authorization_contract_ready"] == 1
    assert validation_checklist["counts"]["browser_response_security_contract_ready"] == 1
    assert validation_checklist["counts"]["browser_mutation_write_contract_ready"] == 1
    assert validation_checklist["counts"]["labeler_route_authorization_contract_ready"] == 1
    assert validation_checklist["counts"]["signed_link_contract_ready"] == 1
    assert validation_checklist["counts"]["expected_user_guard_contract_ready"] == 1
    assert validation_checklist["counts"]["mutation_audit_contract_ready"] == 1
    assert validation_checklist["counts"]["zarr_backup_contract_ready"] == 1
    assert validation_checklist["counts"]["session_guard_contract_ready"] == 1
    assert validation_checklist["counts"]["task_state_contract_ready"] == 1
    assert validation_checklist["counts"]["browser_workflow_scope_contract_ready"] == 1
    assert validation_checklist["assignment_ownership_integrity"]["ok"] is True
    assert validation_checklist["assignment_ownership_contract"]["ready"] is True
    assert validation_checklist["assignment_ownership_contract"]["assignment_scope"] == "recording"
    assert validation_checklist["assignment_ownership_contract"]["recording_assignment_key"] == (
        "recording_id"
    )
    assert validation_checklist["assignment_ownership_contract"][
        "one_current_assignment_row_per_recording"
    ] is True
    assert validation_checklist["assignment_ownership_contract"]["one_active_owner"] is True
    assert validation_checklist["assignment_ownership_contract"][
        "multiple_labelers_per_recording_allowed"
    ] is False
    assert validation_checklist["assignment_ownership_contract"]["store_recording_id_primary_key"] is True
    assert validation_checklist["assignment_ownership_contract"]["store_schema_enforced_recording_primary_key"] is True
    assert validation_checklist["assignment_ownership_contract"]["schema_integrity_source"] == "store_pragma"
    assert validation_checklist["assignment_ownership_contract"]["primary_key_columns"] == ["recording_id"]
    assert validation_checklist["assignment_ownership_contract"]["duplicate_active_owner_count"] == 0
    assert validation_checklist["assignment_ownership_contract"][
        "assignment_manifests_are_control_plane"
    ] is True
    assert validation_checklist["assignment_ownership_contract"][
        "duplicate_manifest_rows_do_not_create_multiple_owners"
    ] is True
    assert validation_checklist["assignment_ownership_contract"][
        "browser_mutation_requires_current_assignment_owner"
    ] is True
    assert validation_checklist["assignment_ownership_contract"]["stale_sessions_closed_on_reassignment"] is True
    assert validation_checklist["assignment_ownership_contract"][
        "stale_sessions_closed_before_assignment_update"
    ] is True
    assert validation_checklist["assignment_ownership_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert validation_checklist["assignment_ownership_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert validation_checklist["counts"]["assignment_ownership_duplicate_active_owners"] == 0
    assert validation_checklist["counts"]["assignment_ownership_contract_ready"] == 1
    assert validation_checklist["ready_for_operator_validation"] is True
    assert validation_checklist["all_validation_complete"] is False
    assert gate_statuses["static_readiness"] == "passed"
    assert gate_statuses["queue_first_entry_contract"] == "passed"
    assert gate_statuses["identity_probe_link_contract"] == "passed"
    assert gate_statuses["assignment_ownership_contract"] == "passed"
    assert gate_statuses["browser_payload_redaction_contract"] == "passed"
    assert gate_statuses["labeler_route_authorization"] == "passed"
    assert gate_statuses["signed_link_contract"] == "passed"
    assert gate_statuses["expected_user_guard_contract"] == "passed"
    assert gate_statuses["session_guard_contract"] == "passed"
    assert gate_statuses["task_state_contract"] == "passed"
    assert gate_statuses["operator_authorization_contract"] == "passed"
    assert gate_statuses["browser_response_security_contract"] == "passed"
    assert gate_statuses["browser_workflow_scope_contract"] == "passed"
    assert gate_statuses["browser_mutation_target_contract"] == "passed"
    assert gate_statuses["browser_mutation_write_policy"] == "passed"
    assert gate_statuses["mutation_audit_contract"] == "passed"
    assert gate_statuses["zarr_backup_contract"] == "passed"
    assert gate_statuses["identity_probe_verification"] == "pending_operator_evidence"
    assert gate_statuses["operator_authorization_boundary"] == "pending_operator_evidence"
    assert gate_statuses["dashboard_visibility"] == "pending_operator_evidence"
    assert gate_statuses["browser_smoke"] == "pending_operator_evidence"
    assert gate_statuses["mutable_zarr_backup_confirmation"] == "pending_operator_evidence"
    backup_gate = next(gate for gate in validation_checklist["gates"] if gate["id"] == "mutable_zarr_backup_confirmation")
    assert backup_gate["required"] is True
    assert "does not include a top-level zarr backup plan" in backup_gate["details"]
    assert "Reference the launch bundle backup plan" in backup_gate["operator_evidence"][0]
    with zipfile.ZipFile(zip_path) as archive:
        assert "alice-handoff/index.html" in archive.namelist()
        assert "alice-handoff/dataset-queue.json" in archive.namelist()
        assert "alice-handoff/message.txt" in archive.namelist()
        assert "alice-handoff/labeler-quickstart.txt" in archive.namelist()
        assert "alice-handoff/validation-log-template.md" in archive.namelist()
        assert "alice-handoff/validation-checklist.json" in archive.namelist()

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-user-handoff",
                "--user",
                "alice",
                "--link-secret",
                "test-secret",
                "--output-dir",
                str(output_dir),
            ]
        )


def test_export_user_handoff_without_base_url_is_preview_not_ready(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "alice-handoff-no-base-url"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints", title="Alice task")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-user-handoff",
            "--user",
            "alice",
            "--link-secret",
            "test-secret",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads(capsys.readouterr().out)
    manifest_file = json.loads((output_dir / "manifest.json").read_text())
    html_index = (output_dir / "index.html").read_text()
    message = (output_dir / "message.txt").read_text()
    quickstart = (output_dir / "labeler-quickstart.txt").read_text()

    assert rc == 0
    assert manifest["ok"] is True
    assert manifest["ready_to_send"] is False
    assert manifest["dashboard_path"] == "/work"
    assert manifest["dashboard_url"] == ""
    assert manifest["expected_user_dashboard_url"] == ""
    assert manifest["labeler_safety"]["dashboard_identity_check_required"] is True
    assert manifest["labeler_safety_policy_present"] is True
    assert manifest["labeler_safety_ready"] is True
    assert manifest["labeler_safety_readiness"] == "passed"
    assert manifest["labeler_route_authorization_policy_present"] is True
    assert manifest["labeler_route_authorization_ready"] is True
    assert manifest["labeler_route_authorization_readiness"] == "passed"
    assert manifest["signed_link_policy_present"] is True
    assert manifest["signed_link_policy_ready"] is True
    assert manifest["signed_link_policy_readiness"] == "passed"
    assert manifest["session_guard_policy_present"] is True
    assert manifest["session_guard_policy_ready"] is True
    assert manifest["session_guard_policy_readiness"] == "passed"
    assert manifest["task_state_policy_present"] is True
    assert manifest["task_state_policy_ready"] is True
    assert manifest["task_state_policy_readiness"] == "passed"
    assert manifest["task_state_completed_tasks_read_only"] is True
    assert manifest["task_state_requires_current_target_token"] is True
    assert manifest["browser_mutation_write_policy_present"] is True
    assert manifest["browser_mutation_write_ready"] is True
    assert manifest["browser_mutation_write_readiness"] == "passed"
    assert manifest["sendability_reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert "--base-url" in manifest["sendability_actions"][0]
    assert manifest["sendability_warnings"][0]["reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert "--base-url" in manifest["sendability_warnings"][0]["actions"][0]
    assert manifest_file["ready_to_send"] is False
    assert manifest_file["sendability_reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert "--base-url" in manifest_file["sendability_actions"][0]
    assert manifest_file["sendability_warnings"][0]["reasons"] == [
        "missing_base_url",
        "operator_validation_needs_review",
    ]
    assert "Wait for operator review" in html_index
    assert "Review reasons: missing_base_url, operator_validation_needs_review" in html_index
    assert "Operator repair action" in html_index
    assert "--base-url" in html_index
    assert "This handoff was generated without a service URL" in html_index
    assert "Needs service URL: /t/" in html_index
    assert "Your Palette labeling handoff needs operator review before starting." in message
    assert "Review reasons: missing_base_url" in message
    assert "Repair actions:" in message
    assert "--base-url" in message
    assert "Your Palette labeling work is ready." not in message
    assert "This handoff was generated without a service base URL." in message
    assert "Wait for operator review before starting." in quickstart
    assert "Review reasons: missing_base_url" in quickstart
    assert "Repair actions:" in quickstart
    assert "--base-url" in quickstart
    assert "How to preview while waiting:" in quickstart
    assert "the handoff was generated without a service URL" in quickstart


def test_export_user_handoff_include_completed_is_not_ready_without_open_work(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "completed-handoff"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-complete",
            recording_id="rec-a",
            workflow_kind="keypoints",
            state="complete",
            title="Completed task",
        )
        store.upsert_task(
            task_id="task-blocked",
            recording_id="rec-a",
            workflow_kind="keypoints",
            state="blocked",
            title="Blocked task",
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-user-handoff",
            "--user",
            "alice",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--include-completed",
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads(capsys.readouterr().out)
    links = [json.loads(line) for line in (output_dir / "signed-links.jsonl").read_text().splitlines()]
    html_index = (output_dir / "index.html").read_text()
    message = (output_dir / "message.txt").read_text()
    quickstart = (output_dir / "labeler-quickstart.txt").read_text()

    assert rc == 0
    assert manifest["ok"] is True
    assert manifest["ready_to_send"] is False
    assert manifest["counts"]["signed_links"] == 2
    assert manifest["counts"]["ready_to_share_links"] == 0
    assert manifest["dataset_queue_state"]["blocks_labeler_start"] is True
    assert manifest["sendability_reasons"] == [
        "no_ready_to_share_links",
        "dataset_queue_blocks_labeler_start",
        "operator_validation_needs_review",
    ]
    assert manifest["sendability_warnings"][0]["reasons"] == [
        "no_ready_to_share_links",
        "dataset_queue_blocks_labeler_start",
        "operator_validation_needs_review",
    ]
    assert "Resolve the user's dataset queue state" in manifest["sendability_actions"][1]
    links_by_task = {link["task_id"]: link for link in links}
    assert links_by_task["task-complete"]["url_is_absolute"] is True
    assert links_by_task["task-complete"]["task_launchable"] is False
    assert links_by_task["task-complete"]["ready_to_share"] is False
    assert [warning["code"] for warning in links_by_task["task-complete"]["shareability_warnings"]] == [
        "task_completed"
    ]
    assert links_by_task["task-blocked"]["url_is_absolute"] is True
    assert links_by_task["task-blocked"]["task_launchable"] is False
    assert links_by_task["task-blocked"]["ready_to_share"] is False
    assert links_by_task["task-blocked"]["startable_task_states"] == ["pending", "in_progress"]
    assert [warning["code"] for warning in links_by_task["task-blocked"]["shareability_warnings"]] == [
        "task_not_startable"
    ]
    assert "Wait for operator review" in html_index
    assert "Review reasons: no_ready_to_share_links, dataset_queue_blocks_labeler_start" in html_index
    assert "Dataset queue state:" in html_index
    assert "Labeler start blocked" in html_index
    assert "Not ready to open: task_completed" in html_index
    assert "Not ready to open: task_not_startable" in html_index
    assert ">Open task</a>" not in html_index
    assert "Your Palette labeling handoff needs operator review before starting." in message
    assert "Review reasons: no_ready_to_share_links, dataset_queue_blocks_labeler_start" in message
    assert "Dataset queue start: blocked" in message
    assert "Do not start new labeling from the dataset queue until the operator resolves this state." in message
    assert "Preview queue-first entry point:" in message
    assert "Preview dashboard:" in message
    assert "Your Palette labeling work is ready." not in message
    assert "Wait for operator review before starting." in quickstart
    assert "Review reasons: no_ready_to_share_links, dataset_queue_blocks_labeler_start" in quickstart
    assert "Dataset queue start: blocked" in quickstart
    assert "Do not start new labeling from the dataset queue until the operator resolves this state." in quickstart
    assert "How to preview while waiting:" in quickstart
    assert "Do not open or save task work until the operator confirms this handoff is ready." in quickstart

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )

    inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert "handoff_not_ready" in inspection["failure_reasons"]
    assert "dataset_queue_blocks_labeler_start" in inspection["failure_reasons"]
    assert "Resolve blocked dataset queue states" in "\n".join(inspection["failure_actions"])
    assert inspection["counts"]["dataset_queue_blocked_start_users"] == ["alice"]
    assert inspection["counts"]["sendability_reasons"]["dataset_queue_blocks_labeler_start"] == 1
    assert inspection["handoffs"][0]["sendability_reasons"] == [
        "no_ready_to_share_links",
        "dataset_queue_blocks_labeler_start",
        "operator_validation_needs_review",
    ]


def test_batch_readiness_cli_reports_active_launch_counts_and_warnings(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_path = tmp_path / "batch-readiness.json"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.assign_recording(recording_id="rec-done", assignee_user="dana")
        store.assign_recording(recording_id="rec-c", assignee_user="carol", status="inactive")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-a-complete", recording_id="rec-a", workflow_kind="keypoints", state="complete")
        store.upsert_task(task_id="task-done", recording_id="rec-done", workflow_kind="keypoints", state="complete")
        store.upsert_task(task_id="task-c", recording_id="rec-c", workflow_kind="detect_analysis")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "batch-readiness",
            "--output",
            str(output_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    archived = json.loads(output_path.read_text())
    warning_codes = {warning["code"] for warning in payload["readiness_warnings"]}

    assert rc == 0
    assert payload["ok"] is True
    assert archived["counts"] == payload["counts"]
    assert payload["single_owner_policy"]["one_active_owner"] is True
    assert payload["assignment_ownership_integrity"]["ok"] is True
    assert payload["assignment_ownership_integrity"]["active_assignment_count"] == 3
    assert payload["assignment_ownership_integrity"]["unique_active_recording_count"] == 3
    assert payload["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    assert payload["counts"]["active_users"] == 3
    assert payload["counts"]["active_open_tasks"] == 1
    assert payload["counts"]["open_tasks"] == 2
    assert payload["counts"]["active_recordings_without_tasks"] == 1
    assert payload["counts"]["active_recordings_without_open_tasks"] == 2
    assert payload["counts"]["active_recordings_without_open_tasks_by_reason"] == {
        "all_tasks_complete": 1,
        "tasks_not_generated": 1,
    }
    assert "Generate or import browser-labeling tasks" in " ".join(
        payload["counts"]["active_recordings_without_open_tasks_actions"]
    )
    assert "Reopen a completed task" in " ".join(payload["counts"]["active_recordings_without_open_tasks_actions"])
    assert payload["counts"]["active_users_without_open_tasks"] == 2
    assert payload["readiness_issues"] == []
    assert "active_assignment_without_tasks" in warning_codes
    assert "active_assignment_without_open_tasks" in warning_codes
    assert "active_user_without_open_tasks" in warning_codes
    readiness_reasons = {
        warning["recording_id"]: warning.get("no_open_task_reason")
        for warning in payload["readiness_warnings"]
        if warning["code"] in {"active_assignment_without_tasks", "active_assignment_without_open_tasks"}
    }
    assert readiness_reasons == {
        "rec-b": "tasks_not_generated",
        "rec-done": "all_tasks_complete",
    }
    readiness_actions = {
        warning["recording_id"]: warning.get("no_open_task_actions")
        for warning in payload["readiness_warnings"]
        if warning["code"] in {"active_assignment_without_tasks", "active_assignment_without_open_tasks"}
    }
    assert "Generate or import browser-labeling tasks" in readiness_actions["rec-b"][0]
    assert "Reopen a completed task" in readiness_actions["rec-done"][0]
    assert payload["store_consistency"]["warning_count"] == 1
    assert payload["store_consistency"]["assignment_ownership_integrity"] == payload["assignment_ownership_integrity"]
    assert payload["warnings_as_errors"] is False
    assert payload["blocking_warning_count"] == 0
    assert payload["blocking_warning_codes"] == []

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "batch-readiness",
            "--warnings-as-errors",
        ]
    )

    strict_payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert strict_payload["ok"] is False
    assert strict_payload["warnings_as_errors"] is True
    assert strict_payload["warning_count"] >= 1
    assert strict_payload["blocking_warning_count"] == strict_payload["warning_count"]
    assert "active_assignment_without_tasks" in strict_payload["blocking_warning_codes"]

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "batch-readiness",
                "--output",
                str(output_path),
            ]
        )


def test_batch_readiness_cli_treats_non_startable_tasks_as_not_open(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.upsert_task(
            task_id="task-blocked",
            recording_id="rec-a",
            workflow_kind="keypoints",
            state="blocked",
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "batch-readiness",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    readiness_reasons = {
        warning["recording_id"]: warning.get("no_open_task_reason")
        for warning in payload["readiness_warnings"]
        if warning["code"] == "active_assignment_without_open_tasks"
    }
    assert rc == 2
    assert payload["ok"] is False
    assert payload["counts"]["open_tasks"] == 0
    assert payload["counts"]["active_open_tasks"] == 0
    assert payload["counts"]["active_recordings_without_open_tasks"] == 1
    assert payload["counts"]["active_recordings_without_open_tasks_by_reason"] == {
        "non_startable_task_state": 1
    }
    assert readiness_reasons == {"rec-a": "non_startable_task_state"}
    assert "no_open_tasks" in {issue["code"] for issue in payload["readiness_issues"]}


def test_export_assignment_and_task_snapshots_cli_writes_filtered_archives(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    assignments_path = tmp_path / "assignments.json"
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_csv_path = tmp_path / "tasks.csv"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.assign_recording(recording_id="rec-b", assignee_user="bob", status="inactive")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
        store.upsert_task(task_id="task-a-complete", recording_id="rec-a", workflow_kind="keypoints", state="complete")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-assignments",
            "--status",
            "active",
            "--output",
            str(assignments_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assignment_payload = json.loads(assignments_path.read_text())
    assert rc == 0
    assert summary["count"] == 1
    assert assignment_payload["filters"]["status"] == "active"
    assert assignment_payload["single_owner_policy"] == _assignment_ownership_policy()
    assert [row["recording_id"] for row in assignment_payload["assignments"]] == ["rec-a"]

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-tasks",
            "--user",
            "alice",
            "--open-only",
            "--format",
            "jsonl",
            "--output",
            str(tasks_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    task_rows = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    assert rc == 0
    assert summary["count"] == 1
    assert task_rows[0]["task_id"] == "task-a"
    assert task_rows[0]["assignee_user"] == "alice"

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-tasks",
            "--user",
            "alice",
            "--open-only",
            "--format",
            "csv",
            "--output",
            str(tasks_csv_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    task_csv_rows = list(csv.DictReader(tasks_csv_path.open()))
    assert rc == 0
    assert summary["count"] == 1
    assert task_csv_rows[0]["task_id"] == "task-a"
    assert task_csv_rows[0]["assignee_user"] == "alice"

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-tasks",
                "--output",
                str(tasks_path),
            ]
        )


def test_export_launch_bundle_cli_writes_plan_readiness_handoffs_and_zip(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "launch-bundle"
    zip_path = tmp_path / "launch-bundle.zip"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.assign_recording(recording_id="rec-b", assignee_user="bob", notes="Bob instructions")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints", title="Alice task")
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis", title="Bob task")
        store.record_event(
            task_id="task-a",
            recording_id="rec-a",
            user="alice",
            event_type="save_keypoints",
            target={"frame_idx": 1},
        )
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-launch-bundle",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--output-dir",
            str(output_dir),
            "--zip-output",
            str(zip_path),
            "--include-audit-events",
        ]
    )

    manifest = json.loads(capsys.readouterr().out)
    manifest_file = json.loads((output_dir / "manifest.json").read_text())
    assignments = json.loads((output_dir / "assignments.json").read_text())
    tasks = json.loads((output_dir / "tasks.json").read_text())
    zarr_backup_plan = json.loads((output_dir / "zarr-backup-plan.json").read_text())
    zarr_backup_evidence = json.loads(
        (output_dir / "zarr-backup-evidence-template.json").read_text()
    )
    response_security_evidence = json.loads(
        (output_dir / "browser-response-security-evidence-template.json").read_text()
    )
    identity_source_evidence = json.loads(
        (output_dir / "identity-source-evidence-template.json").read_text()
    )
    browser_smoke_evidence = json.loads(
        (output_dir / "browser-smoke-evidence-template.json").read_text()
    )
    disposable_zarr_smoke_evidence = json.loads(
        (output_dir / "disposable-zarr-mutation-smoke-evidence-template.json").read_text()
    )
    readiness = json.loads((output_dir / "batch-readiness.json").read_text())
    handoffs_index = json.loads((output_dir / "handoffs" / "index.json").read_text())
    handoffs_roster_rows = list(csv.DictReader((output_dir / "handoffs" / "labeler-roster.csv").open()))
    handoffs_html_index = (output_dir / "handoffs" / "index.html").read_text()
    readme = (output_dir / "launch-readme.txt").read_text()
    implementation_status = (output_dir / "implementation-status.txt").read_text()
    inspect_command = (output_dir / "inspect-command.txt").read_text()
    operator_evidence_commands = (output_dir / "operator-evidence-commands.txt").read_text()
    launch_evidence_execution_checklist = (
        output_dir / "launch-evidence-execution-checklist.txt"
    ).read_text()
    inspection_targets = json.loads((output_dir / "inspection-targets.json").read_text())
    html_index = (output_dir / "index.html").read_text()
    checksums = json.loads((output_dir / "checksums.json").read_text())
    validation_checklist = json.loads((output_dir / "validation-checklist.json").read_text())
    task_events = [json.loads(line) for line in (output_dir / "audit" / "task-events.jsonl").read_text().splitlines()]
    assignment_events = [json.loads(line) for line in (output_dir / "audit" / "assignment-events.jsonl").read_text().splitlines()]
    task_definition_events = [json.loads(line) for line in (output_dir / "audit" / "task-definition-events.jsonl").read_text().splitlines()]

    assert rc == 2
    assert manifest["ok"] is False
    assert manifest["readiness_ok"] is True
    assert manifest["handoffs_ok"] is False
    assert manifest["labeler_landing_page_path"] == "/"
    assert manifest["labeler_landing_url"] == "https://labeling.example.org"
    assert manifest["dashboard_path"] == "/work"
    assert manifest["dashboard_url"] == "https://labeling.example.org/work"
    assert manifest["dataset_queue_page_path"] == "/datasets"
    assert manifest["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert manifest["single_owner_policy"]["one_active_owner"] is True
    assert manifest["assignment_ownership_integrity"]["ok"] is True
    assert manifest["assignment_ownership_integrity"]["duplicate_active_owner_count"] == 0
    assert manifest["assignment_snapshot"]["recording_ids"] == ["rec-a", "rec-b"]
    assert manifest["assignment_snapshot"]["recording_count"] == 2
    assert manifest["labeler_safety"]["dashboard_identity_check_required"] is True
    assert manifest["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert "labeler_safety_identity_probe_success_launch_ctas_rendered" in (
        inspection_targets["shareability_labeler_safety_policy_fields"]
    )
    assert inspection_targets["shareability_labeler_safety_policy_required_values"][
        "labeler_safety_identity_probe_expected_user_guard_required"
    ] is True
    assert inspection_targets["shareability_labeler_safety_policy_required_values"][
        "labeler_safety_identity_probe_success_launch_ctas_rendered"
    ] is True
    assert inspection_targets["shareability_labeler_safety_policy_required_values"][
        "labeler_safety_identity_probe_failed_launch_ctas_suppressed"
    ] is True
    assert inspection_targets["shareability_labeler_safety_policy_required_values"][
        "labeler_safety_identity_probe_failed_support_urls_diagnostic_only"
    ] is True
    assert manifest["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert manifest["browser_mutation_write_policy"]["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert manifest["browser_mutation_write_policy"]["training_zarr_mutations_are_server_owned"] is True
    assert manifest["browser_mutation_write_policy"]["handoff_artifacts_are_metadata_only"] is True
    assert manifest["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert manifest["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert manifest["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert manifest["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert manifest["browser_mutation_write_policy"]["browser_has_direct_zarr_write_authority"] is False
    assert manifest["files"]["zarr_backup_evidence_template"] == str(
        output_dir / "zarr-backup-evidence-template.json"
    )
    assert zarr_backup_evidence["schema"] == "palette.web_labeling_zarr_backup_evidence_template.v1"
    assert zarr_backup_evidence["operator_only"] is True
    assert zarr_backup_evidence["labelers_do_not_receive"] is True
    assert zarr_backup_evidence["source_plan_schema"] == zarr_backup_plan["schema"]
    assert all(
        "backup_execution_manifest_path" in row
        for row in zarr_backup_evidence["targets"]
    )
    assert zarr_backup_evidence["counts"]["backup_required_targets"] == zarr_backup_plan["counts"][
        "backup_required_targets"
    ]
    assert manifest["files"]["browser_response_security_evidence_template"] == str(
        output_dir / "browser-response-security-evidence-template.json"
    )
    assert response_security_evidence["schema"] == (
        "palette.web_labeling_browser_response_security_evidence_template.v1"
    )
    assert response_security_evidence["operator_only"] is True
    assert response_security_evidence["labelers_do_not_receive"] is True
    assert response_security_evidence["validation_gate"] == "browser_response_security_headers"
    assert response_security_evidence["expected_headers"]["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    assert response_security_evidence["captured_headers"]["Cache-Control"] == ""
    assert response_security_evidence["operator_approval"]["status"] == "pending_operator_confirmation"
    assert manifest["files"]["identity_source_evidence_template"] == str(
        output_dir / "identity-source-evidence-template.json"
    )
    assert identity_source_evidence["schema"] == "palette.web_labeling_identity_source_evidence_template.v1"
    assert identity_source_evidence["operator_only"] is True
    assert identity_source_evidence["labelers_do_not_receive"] is True
    assert identity_source_evidence["validation_gate"] == "identity_probe_verification"
    assert identity_source_evidence["counts"]["users"] == 2
    assert {row["expected_user"] for row in identity_source_evidence["users"]} == {"alice", "bob"}
    assert {
        row["expected_user_identity_probe_url"] for row in identity_source_evidence["users"]
    } == {
        "https://labeling.example.org/identity?expected_user=alice",
        "https://labeling.example.org/identity?expected_user=bob",
    }
    assert {
        row["expected_user_dataset_queue_url"] for row in identity_source_evidence["users"]
    } == {
        "https://labeling.example.org/datasets?expected_user=alice",
        "https://labeling.example.org/datasets?expected_user=bob",
    }
    assert {
        row["expected_user_personal_dataset_queue_url"]
        for row in identity_source_evidence["users"]
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert identity_source_evidence["labeling_home_page_path"] == "/labeling"
    assert identity_source_evidence["labeling_home_url"] == "https://labeling.example.org/labeling"
    assert {
        row["expected_user_labeling_home_url"] for row in identity_source_evidence["users"]
    } == {
        "https://labeling.example.org/labeling?expected_user=alice",
        "https://labeling.example.org/labeling?expected_user=bob",
    }
    assert {row["preferred_labeler_entrypoint"] for row in identity_source_evidence["users"]} == {
        "personal_datasets_waiting_queue"
    }
    assert {row["preferred_labeler_entry_url"] for row in identity_source_evidence["users"]} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["personalized_labeler_entrypoint"] for row in identity_source_evidence["users"]} == {
        "personal_datasets_waiting_queue"
    }
    assert {row["personalized_labeler_entry_url"] for row in identity_source_evidence["users"]} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["labeling_home_link_role"] for row in identity_source_evidence["users"]} == {
        "human_readable_queue_alias"
    }
    assert {row["labeling_home_is_preferred_entrypoint"] for row in identity_source_evidence["users"]} == {
        False
    }
    assert {
        row["preferred_labeler_entry_url_matches_dataset_queue"]
        for row in identity_source_evidence["users"]
    } == {True}
    assert {
        row["preferred_labeler_entry_url_matches_personal_dataset_queue"]
        for row in identity_source_evidence["users"]
    } == {True}
    assert {
        row["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        for row in identity_source_evidence["users"]
    } == {True}
    assert manifest["files"]["browser_smoke_evidence_template"] == str(
        output_dir / "browser-smoke-evidence-template.json"
    )
    assert browser_smoke_evidence["schema"] == "palette.web_labeling_browser_smoke_evidence_template.v1"
    assert browser_smoke_evidence["operator_only"] is True
    assert browser_smoke_evidence["validation_gate"] == "browser_smoke"
    assert browser_smoke_evidence["recommended_representative_user"] == "alice"
    assert "browser_only_runtime_verified" in browser_smoke_evidence["required_checks"]
    assert "no_local_palette_install_verified" in browser_smoke_evidence["required_checks"]
    assert "no_local_crimson_install_verified" in browser_smoke_evidence["required_checks"]
    assert "no_local_conda_or_project_dependencies_verified" in browser_smoke_evidence["required_checks"]
    assert "personalized_dataset_queue_verified" in browser_smoke_evidence["required_checks"]
    assert (
        "preferred_labeler_entry_url_matches_personal_dataset_queue"
        in browser_smoke_evidence["required_checks"]
    )
    assert (
        "personalized_labeler_entry_url_matches_personal_dataset_queue"
        in browser_smoke_evidence["required_checks"]
    )
    assert "personalized_work_dashboard_verified" in browser_smoke_evidence["required_checks"]
    assert browser_smoke_evidence["personalized_route_smoke_contract"] == {
        "schema": "palette.web_labeling_browser_smoke_route_contract.v1",
        "preferred_queue_entrypoint": "personal_datasets_waiting_queue",
        "preferred_queue_path": "/my-datasets",
        "human_readable_queue_alias_path": "/labeling",
        "human_readable_queue_alias_url_field": "labeling_home_url",
        "fallback_dashboard_entrypoint": "personal_work_dashboard",
        "fallback_dashboard_path": "/my-work",
        "canonical_queue_fallback_path": "/datasets",
        "canonical_dashboard_fallback_path": "/work",
        "expected_user_query_required": True,
        "required_check_fields": [
            "personalized_dataset_queue_verified",
            "preferred_labeler_entry_url_matches_personal_dataset_queue",
            "personalized_labeler_entry_url_matches_personal_dataset_queue",
            "personalized_work_dashboard_verified",
        ],
        "per_user_route_url_fields": [
            "personalized_dataset_queue_url",
            "labeling_home_url",
            "personalized_work_url",
        ],
        "per_user_route_link_roles": {
            "personalized_dataset_queue_url": "preferred_queue",
            "labeling_home_url": "human_readable_queue_alias",
            "personalized_work_url": "fallback_dashboard",
        },
        "canonical_fallback_url_fields": [
            "dataset_queue_url",
            "dashboard_url",
        ],
        "canonical_fallback_link_roles": {
            "dataset_queue_url": "canonical_queue_fallback",
            "dashboard_url": "canonical_dashboard_fallback",
        },
        "identity_probe_url_field": "identity_probe_url",
        "identity_probe_link_role": "identity_check",
        "wrong_expected_user_url_fields": [
            "wrong_expected_user_personalized_dataset_queue_url",
            "wrong_expected_user_labeling_home_url",
            "wrong_expected_user_personalized_work_url",
        ],
        "wrong_expected_user_link_role": "expected_user_mismatch_check",
        "expected_user_mismatch_check_field": "expected_user_mismatch_rejected",
        "labelers_need_local_palette_or_crimson_install": False,
    }
    assert "personalized /my-work dashboard fallback" in "\n".join(
        browser_smoke_evidence["instructions"]
    )
    assert "human-readable /labeling alias" in "\n".join(
        browser_smoke_evidence["instructions"]
    )
    browser_smoke_status = _operator_evidence_template_status(
        gate_id="browser_smoke",
        template_path="browser-smoke-evidence-template.json",
        template=browser_smoke_evidence,
        present=True,
        valid=True,
    )
    assert browser_smoke_status["personalized_route_smoke_contract"] == (
        browser_smoke_evidence["personalized_route_smoke_contract"]
    )
    assert browser_smoke_status["actual_personalized_route_smoke_contract"] == (
        browser_smoke_evidence["personalized_route_smoke_contract"]
    )
    assert browser_smoke_status["expected_personalized_route_smoke_contract"] == (
        _browser_smoke_personalized_route_contract()
    )
    assert browser_smoke_status["personalized_route_smoke_contract_ready"] is True
    assert browser_smoke_status["personalized_route_smoke_contract_missing_fields"] == []
    assert {row["expected_user"] for row in browser_smoke_evidence["users"]} == {"alice", "bob"}
    assert {
        row["personalized_dataset_queue_url"] for row in browser_smoke_evidence["users"]
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {
        row["personalized_work_url"] for row in browser_smoke_evidence["users"]
    } == {
        "https://labeling.example.org/my-work?expected_user=alice",
        "https://labeling.example.org/my-work?expected_user=bob",
    }
    assert {
        row["labeling_home_url"] for row in browser_smoke_evidence["users"]
    } == {
        "https://labeling.example.org/labeling?expected_user=alice",
        "https://labeling.example.org/labeling?expected_user=bob",
    }
    assert {
        row["wrong_expected_user_personalized_dataset_queue_url"]
        for row in browser_smoke_evidence["users"]
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {
        row["wrong_expected_user_labeling_home_url"]
        for row in browser_smoke_evidence["users"]
    } == {
        "https://labeling.example.org/labeling?expected_user=alice",
        "https://labeling.example.org/labeling?expected_user=bob",
    }
    assert {
        row["wrong_expected_user_personalized_work_url"]
        for row in browser_smoke_evidence["users"]
    } == {
        "https://labeling.example.org/my-work?expected_user=alice",
        "https://labeling.example.org/my-work?expected_user=bob",
    }
    assert {row["browser_only_runtime_verified"] for row in browser_smoke_evidence["users"]} == {False}
    assert {row["no_local_palette_install_verified"] for row in browser_smoke_evidence["users"]} == {False}
    assert {row["no_local_crimson_install_verified"] for row in browser_smoke_evidence["users"]} == {False}
    assert {
        row["no_local_conda_or_project_dependencies_verified"] for row in browser_smoke_evidence["users"]
    } == {False}
    assert {row["personalized_dataset_queue_verified"] for row in browser_smoke_evidence["users"]} == {False}
    assert {
        row["preferred_labeler_entry_url_matches_personal_dataset_queue"]
        for row in browser_smoke_evidence["users"]
    } == {False}
    assert {
        row["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        for row in browser_smoke_evidence["users"]
    } == {False}
    assert {row["personalized_work_dashboard_verified"] for row in browser_smoke_evidence["users"]} == {False}


def test_record_browser_smoke_evidence_reports_personalized_route_contract(tmp_path):
    evidence_path = tmp_path / "browser-smoke-evidence-template.json"
    template = _browser_smoke_evidence_template(
        base_url="https://labeling.example.org",
        users=["alice"],
    )
    evidence_path.write_text(json.dumps(template), encoding="utf-8")

    report = _record_browser_smoke_evidence(
        evidence_path=evidence_path,
        expected_user="alice",
        resolved_user="alice",
        operator="ops",
        checks={str(field): True for field in template["required_checks"]},
    )
    updated = json.loads(evidence_path.read_text())
    status = _operator_evidence_template_status(
        gate_id="browser_smoke",
        template_path=str(evidence_path),
        template=updated,
        present=True,
        valid=True,
    )

    assert report["ok"] is True
    assert report["personalized_route_smoke_contract"] == template[
        "personalized_route_smoke_contract"
    ]
    assert report["actual_personalized_route_smoke_contract"] == template[
        "personalized_route_smoke_contract"
    ]
    assert report["expected_personalized_route_smoke_contract"] == (
        _browser_smoke_personalized_route_contract()
    )
    assert report["personalized_route_smoke_contract_ready"] is True
    assert report["personalized_route_smoke_contract_missing_fields"] == []
    assert report["personalized_route_smoke_contract_operator_action"] == ""
    assert status["ready"] is True
    assert status["personalized_route_smoke_contract"] == template[
        "personalized_route_smoke_contract"
    ]
    assert status["actual_personalized_route_smoke_contract"] == template[
        "personalized_route_smoke_contract"
    ]
    assert status["expected_personalized_route_smoke_contract"] == (
        _browser_smoke_personalized_route_contract()
    )
    assert status["personalized_route_smoke_contract_ready"] is True
    assert status["personalized_route_smoke_contract_missing_fields"] == []
    assert status["personalized_route_smoke_contract_operator_action"] == ""
    assert status["user_statuses"][0]["personalized_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert status["user_statuses"][0]["personalized_work_url"] == (
        "https://labeling.example.org/my-work?expected_user=alice"
    )
    assert status["user_statuses"][0][
        "wrong_expected_user_personalized_dataset_queue_url"
    ] == ""
    assert status["user_statuses"][0]["wrong_expected_user_personalized_work_url"] == ""

    stale_updated = dict(updated)
    stale_updated.pop("personalized_route_smoke_contract", None)
    stale_status = _operator_evidence_template_status(
        gate_id="browser_smoke",
        template_path=str(evidence_path),
        template=stale_updated,
        present=True,
        valid=True,
    )

    assert stale_status["ready"] is False
    assert stale_status["actual_personalized_route_smoke_contract"] == {}
    assert stale_status["expected_personalized_route_smoke_contract"] == (
        _browser_smoke_personalized_route_contract()
    )
    assert stale_status["personalized_route_smoke_contract_ready"] is False
    assert "schema" in stale_status["personalized_route_smoke_contract_missing_fields"]
    assert "Regenerate browser-smoke-evidence-template.json" in stale_status[
        "personalized_route_smoke_contract_operator_action"
    ]
    stale_failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist={
            "operator_evidence_template_statuses": {
                "browser_smoke": stale_status,
            },
            "passed_gates_with_unapproved_evidence_templates": ["browser_smoke"],
        },
    )
    stale_failure_action_text = "\n".join(stale_failure_actions)
    assert "Browser smoke personalized route contract is stale" in stale_failure_action_text
    assert "Regenerate browser-smoke-evidence-template.json" in stale_failure_action_text
    assert "/labeling" in stale_failure_action_text
    assert (
        "--preferred-labeler-entry-url-matches-personal-dataset-queue"
        in stale_failure_action_text
    )
    assert (
        "--personalized-labeler-entry-url-matches-personal-dataset-queue"
        in stale_failure_action_text
    )
    assert "--personalized-work-dashboard-verified" in stale_failure_action_text
    stale_repair_command_rows = {
        row["id"]: row for row in _inspection_operator_repair_commands(stale_failure_actions)
    }
    assert stale_repair_command_rows["record_browser_smoke_evidence"]["gate_ids"] == [
        "browser_smoke"
    ]
    assert stale_repair_command_rows["record_browser_smoke_evidence"]["reason_ids"] == [
        "browser_smoke_personalized_route_contract_stale"
    ]
    assert "--personalized-work-dashboard-verified" in stale_repair_command_rows[
        "record_browser_smoke_evidence"
    ]["command"]
    assert "--preferred-labeler-entry-url-matches-personal-dataset-queue" in stale_repair_command_rows[
        "record_browser_smoke_evidence"
    ]["command"]
    assert "--personalized-labeler-entry-url-matches-personal-dataset-queue" in stale_repair_command_rows[
        "record_browser_smoke_evidence"
    ]["command"]
    assert stale_repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == [
        "browser_smoke"
    ]

    stale_evidence_path = tmp_path / "stale-browser-smoke-evidence-template.json"
    stale_template = dict(template)
    stale_template.pop("personalized_route_smoke_contract", None)
    stale_evidence_path.write_text(json.dumps(stale_template), encoding="utf-8")
    stale_report = _record_browser_smoke_evidence(
        evidence_path=stale_evidence_path,
        expected_user="alice",
        resolved_user="alice",
        operator="ops",
        checks={str(field): True for field in template["required_checks"]},
    )

    assert stale_report["ok"] is False
    assert stale_report["actual_personalized_route_smoke_contract"] == {}
    assert stale_report["expected_personalized_route_smoke_contract"] == (
        _browser_smoke_personalized_route_contract()
    )
    assert stale_report["personalized_route_smoke_contract_ready"] is False
    assert "schema" in stale_report["personalized_route_smoke_contract_missing_fields"]
    assert "Regenerate browser-smoke-evidence-template.json" in stale_report[
        "personalized_route_smoke_contract_operator_action"
    ]
    stale_error = next(
        error
        for error in stale_report["errors"]
        if error["error"] == "personalized_route_smoke_contract_stale"
    )
    assert stale_error["expected_personalized_route_smoke_contract"] == (
        _browser_smoke_personalized_route_contract()
    )
    assert stale_error["actual_personalized_route_smoke_contract"] == {}
    assert "Regenerate browser-smoke-evidence-template.json" in stale_error[
        "operator_action"
    ]
def test_inspect_handoff_validation_checklist_flags_passed_gate_without_approved_template(tmp_path):
    evidence_path = tmp_path / "identity-source-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "identity_personal_queue_evidence_status": "ready",
                "identity_source_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_identity_source_evidence_template.v1",
                "users": [
                    {
                        "expected_user": "alice",
                        "identity_matches_expected_user": False,
                        "operator_approved_at_utc": "",
                    }
                ],
            }
        )
    )

    inspection = _inspect_handoff_validation_checklist(tmp_path)

    status = inspection["operator_evidence_template_statuses"]["identity_probe_verification"]
    assert status["present"] is True
    assert status["ready"] is False
    assert status["approval_status"] == "pending_operator_confirmation"
    assert inspection["identity_personal_queue_evidence_status"] == "incomplete"
    assert inspection["identity_personal_queue_evidence_ready_count"] == 0
    assert inspection["identity_personal_queue_evidence_missing_count"] == 1
    assert inspection["identity_personal_queue_evidence_ready_users"] == []
    assert inspection["identity_personal_queue_evidence_missing_users"] == ["alice"]
    assert inspection["identity_all_users_have_personal_queue_evidence"] is False
    assert "identity_probe_verification" in inspection["operator_evidence_template_pending_gate_ids"]
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == ["identity_probe_verification"]
    assert "record_identity_source_evidence" in inspection[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert inspection["operator_validation_command_templates"]["commands_by_gate_id"][
        "identity_probe_verification"
    ] == [
        "record_identity_source_evidence",
        "apply_operator_evidence_templates",
    ]
    repair_command_rows = {
        row["id"]: row
        for row in _inspection_operator_repair_commands(
            [],
            validation_checklist=inspection,
        )
    }
    assert repair_command_rows["record_identity_source_evidence"]["category"] == "operator_evidence"
    assert repair_command_rows["record_identity_source_evidence"]["gate_ids"] == [
        "identity_probe_verification"
    ]
    assert repair_command_rows["record_identity_source_evidence"]["reason_ids"] == [
        "personal_dataset_queue_link_evidence_incomplete"
    ]
    assert repair_command_rows["apply_operator_evidence_templates"]["category"] == (
        "validation_checklist"
    )
    assert repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == [
        "browser_response_security_headers",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "mutable_zarr_backup_confirmation",
        "identity_probe_verification",
    ]


def test_inspect_handoff_validation_checklist_accepts_operator_approved_template(tmp_path):
    evidence_path = tmp_path / "identity-source-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "identity_source_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_identity_source_evidence_template.v1",
                "users": [
                    {
                        "expected_user": "alice",
                        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
                        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "preferred_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "personalized_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entry_url_matches_dataset_queue": True,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": True,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": True,
                        "resolved_user": "alice",
                        "identity_matches_expected_user": True,
                        "operator": "ops",
                        "operator_approved_at_utc": "2026-06-23T00:00:00+00:00",
                    }
                ],
            }
        )
    )

    inspection = _inspect_handoff_validation_checklist(tmp_path)

    status = inspection["operator_evidence_template_statuses"]["identity_probe_verification"]
    assert status["present"] is True
    assert status["ready"] is True
    assert status["approval_status"] == "operator_approved"
    assert status["user_statuses"][0]["resolved_user_matches_expected_user"] is True
    assert inspection["identity_personal_queue_evidence_ready_count"] == 1
    assert inspection["identity_personal_queue_evidence_missing_count"] == 0
    assert inspection["identity_personal_queue_evidence_ready_users"] == ["alice"]
    assert inspection["identity_personal_queue_evidence_missing_users"] == []
    assert inspection["identity_all_users_have_personal_queue_evidence"] is True
    assert "identity_probe_verification" in inspection["operator_evidence_template_approved_gate_ids"]
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == []


def test_record_identity_source_evidence_cli_updates_template_and_inspection(tmp_path, capsys):
    evidence_path = tmp_path / "identity-source-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "identity_source_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_identity_source_evidence_template.v1",
                "counts": {
                    "users": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "users": [
                    {
                        "expected_user": "alice",
                        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
                        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "preferred_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "personalized_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entry_url_matches_dataset_queue": False,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": False,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": False,
                        "resolved_user": "",
                        "identity_matches_expected_user": False,
                        "captured_at_utc": "",
                        "authenticated_session_context": "",
                        "operator": "",
                        "operator_approved_at_utc": "",
                        "notes": "",
                    }
                ],
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-identity-source-evidence",
            "--evidence",
            str(evidence_path),
            "--expected-user",
            "alice",
            "--resolved-user",
            "alice",
            "--operator",
            "ops",
            "--authenticated-session-context",
            "deployed proxy identity probe",
            "--notes",
            "matched in browser",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["identity_matches_expected_user"] is True
    assert payload["personal_queue_evidence_ready"] is True
    assert payload["preferred_labeler_entry_url_matches_dataset_queue"] is True
    assert payload["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert payload["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert payload["counts"]["operator_approved"] == 1
    assert updated["counts"]["pending_operator_confirmation"] == 0
    assert updated["users"][0]["resolved_user"] == "alice"
    assert updated["users"][0]["identity_matches_expected_user"] is True
    assert updated["users"][0]["preferred_labeler_entry_url_matches_dataset_queue"] is True
    assert updated["users"][0]["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert updated["users"][0]["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert updated["users"][0]["operator"] == "ops"
    assert updated["users"][0]["operator_approved_at_utc"]
    assert updated["users"][0]["authenticated_session_context"] == "deployed proxy identity probe"
    assert updated["users"][0]["notes"] == "matched in browser"
    status = inspection["operator_evidence_template_statuses"]["identity_probe_verification"]
    assert status["ready"] is True
    assert status["approval_status"] == "operator_approved"
    assert status["user_statuses"][0]["personal_queue_evidence_ready"] is True
    assert status["user_statuses"][0]["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert status["user_statuses"][0]["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert status["personal_queue_evidence_ready_count"] == 1
    assert status["personal_queue_evidence_missing_count"] == 0
    assert status["personal_queue_evidence_ready_users"] == ["alice"]
    assert status["personal_queue_evidence_missing_users"] == []
    assert status["personal_queue_evidence_missing_fields_by_user"] == {}
    assert status["all_users_have_personal_queue_evidence"] is True
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == []


def test_record_identity_source_evidence_cli_records_mismatch_without_approval(tmp_path, capsys):
    evidence_path = tmp_path / "identity-source-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "identity_source_evidence_template": str(evidence_path),
            }
        )
        + "\n"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_identity_source_evidence_template.v1",
                "counts": {
                    "users": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "users": [
                    {
                        "expected_user": "alice",
                        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
                        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "preferred_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "personalized_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entry_url_matches_dataset_queue": False,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": False,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": False,
                        "resolved_user": "",
                        "identity_matches_expected_user": False,
                        "captured_at_utc": "",
                        "authenticated_session_context": "",
                        "operator": "",
                        "operator_approved_at_utc": "",
                        "notes": "",
                    }
                ],
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-identity-source-evidence",
            "--evidence",
            str(evidence_path),
            "--expected-user",
            "alice",
            "--resolved-user",
            "bob",
            "--operator",
            "ops",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 2
    assert payload["ok"] is False
    assert payload["identity_matches_expected_user"] is False
    assert payload["personal_queue_evidence_ready"] is True
    assert payload["errors"][0]["error"] == "identity_mismatch"
    assert updated["counts"]["operator_approved"] == 0
    assert updated["counts"]["pending_operator_confirmation"] == 1
    assert updated["users"][0]["resolved_user"] == "bob"
    assert updated["users"][0]["identity_matches_expected_user"] is False
    assert updated["users"][0]["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert updated["users"][0]["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert updated["users"][0]["operator_approved_at_utc"] == ""
    status = inspection["operator_evidence_template_statuses"]["identity_probe_verification"]
    assert status["ready"] is False
    assert status["approval_status"] == "pending_operator_confirmation"
    assert status["user_statuses"][0]["expected_user"] == "alice"
    assert status["user_statuses"][0]["resolved_user"] == "bob"
    assert "identity_matches_expected_user" in status["user_statuses"][0]["missing_fields"]
    assert "operator_approved_at_utc" in status["user_statuses"][0]["missing_fields"]
    assert status["users_missing_required_fields"][0]["expected_user"] == "alice"
    failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist=inspection,
    )
    failure_action_text = "\n".join(failure_actions)
    assert "Identity-source evidence incomplete for alice" in failure_action_text
    assert "resolved_user=bob" in failure_action_text
    assert "record-identity-source-evidence --expected-user alice" in failure_action_text
    repair_command_rows = {row["id"]: row for row in _inspection_operator_repair_commands(failure_actions)}
    assert repair_command_rows["apply_operator_evidence_templates"]["category"] == "validation_checklist"
    assert repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == [
        "identity_probe_verification"
    ]


def test_record_identity_source_evidence_cli_blocks_missing_personal_queue_evidence(
    tmp_path, capsys
):
    evidence_path = tmp_path / "identity-source-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "identity_source_evidence_template": str(evidence_path),
            }
        )
        + "\n"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_identity_source_evidence_template.v1",
                "counts": {
                    "users": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "users": [
                    {
                        "expected_user": "alice",
                        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
                        "resolved_user": "",
                        "identity_matches_expected_user": False,
                        "captured_at_utc": "",
                        "authenticated_session_context": "",
                        "operator": "",
                        "operator_approved_at_utc": "",
                        "notes": "",
                    }
                ],
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-identity-source-evidence",
            "--evidence",
            str(evidence_path),
            "--expected-user",
            "alice",
            "--resolved-user",
            "alice",
            "--operator",
            "ops",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 2
    assert payload["ok"] is False
    assert payload["identity_matches_expected_user"] is True
    assert payload["personal_queue_evidence_ready"] is False
    assert payload["errors"][0]["error"] == "personal_dataset_queue_link_evidence_incomplete"
    assert "expected_user_personal_dataset_queue_url" in payload["errors"][0]["missing_fields"]
    assert "preferred_labeler_entry_url_matches_personal_dataset_queue" in payload[
        "errors"
    ][0]["missing_fields"]
    assert updated["counts"]["operator_approved"] == 0
    assert updated["counts"]["pending_operator_confirmation"] == 1
    assert updated["users"][0]["resolved_user"] == "alice"
    assert updated["users"][0]["identity_matches_expected_user"] is True
    assert updated["users"][0]["operator_approved_at_utc"] == ""
    status = inspection["operator_evidence_template_statuses"]["identity_probe_verification"]
    assert status["ready"] is False
    assert status["user_statuses"][0]["personal_queue_evidence_ready"] is False
    assert status["personal_queue_evidence_ready_count"] == 0
    assert status["personal_queue_evidence_missing_count"] == 1
    assert status["personal_queue_evidence_ready_users"] == []
    assert status["personal_queue_evidence_missing_users"] == ["alice"]
    assert status["all_users_have_personal_queue_evidence"] is False
    assert status["personal_queue_evidence_missing_fields_by_user"]["alice"]
    assert inspection["identity_personal_queue_evidence_status"] == "incomplete"
    assert inspection["identity_personal_queue_evidence_ready_count"] == 0
    assert inspection["identity_personal_queue_evidence_missing_count"] == 1
    assert inspection["identity_personal_queue_evidence_ready_users"] == []
    assert inspection["identity_personal_queue_evidence_missing_users"] == ["alice"]
    assert inspection["identity_all_users_have_personal_queue_evidence"] is False
    assert inspection["identity_personal_queue_evidence_missing_fields_by_user"]["alice"]
    assert "expected_user_personal_dataset_queue_url" in status["user_statuses"][0][
        "missing_fields"
    ]
    assert "preferred_labeler_entry_url_matches_personal_dataset_queue" in status[
        "user_statuses"
    ][0]["missing_fields"]
    failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist=inspection,
    )
    failure_action_text = "\n".join(failure_actions)
    assert "preferred_labeler_entry_url plus personalized_labeler_entry_url" in failure_action_text
    assert "/my-datasets?expected_user=<user>" in failure_action_text
    repair_command_rows = {
        row["id"]: row for row in _inspection_operator_repair_commands(failure_actions)
    }
    assert repair_command_rows["record_identity_source_evidence"]["reason_ids"] == [
        "personal_dataset_queue_link_evidence_incomplete"
    ]
    assert repair_command_rows["record_identity_source_evidence"]["gate_ids"] == [
        "identity_probe_verification"
    ]
    merged_repair_command_rows = {
        row["id"]: row
        for row in _inspection_operator_repair_commands(
            failure_actions,
            validation_checklist=inspection,
        )
    }
    assert merged_repair_command_rows["record_identity_source_evidence"]["reason_ids"] == [
        "personal_dataset_queue_link_evidence_incomplete"
    ]
    assert merged_repair_command_rows["record_identity_source_evidence"]["gate_ids"] == [
        "identity_probe_verification"
    ]


def test_record_browser_response_security_evidence_cli_updates_template_and_inspection(tmp_path, capsys):
    evidence_path = tmp_path / "browser-response-security-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "browser_response_security_headers",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "browser_response_security_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_browser_response_security_evidence_template.v1",
                "expected_headers": {
                    "Cache-Control": "no-store",
                    "X-Frame-Options": "DENY",
                },
                "expected_user_capture_query_required": True,
                "preferred_capture_url": "https://labeling.example.org/datasets?expected_user=alice",
                "captured_headers": {
                    "Cache-Control": "",
                    "X-Frame-Options": "",
                },
                "capture": {
                    "url": "",
                    "authenticated_test_user": "",
                    "captured_at_utc": "",
                    "capture_command_or_browser_note": "",
                    "proxy_or_deployment": "",
                },
                "checks": {
                    "cache_control_preserved": False,
                    "x_frame_options_preserved": False,
                    "proxy_strips_or_weakens_no_headers": False,
                },
                "operator_approval": {
                    "status": "pending_operator_confirmation",
                    "operator": "",
                    "approved_at_utc": "",
                    "notes": "",
                },
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-browser-response-security-evidence",
            "--evidence",
            str(evidence_path),
            "--header",
            "cache-control=no-store",
            "--header",
            "x-frame-options=DENY",
            "--operator",
            "ops",
            "--capture-url",
            "https://labeling.example.org/datasets?expected_user=alice",
            "--authenticated-test-user",
            "alice",
            "--capture-note",
            "captured with browser devtools",
            "--proxy-or-deployment",
            "labeling proxy",
            "--notes",
            "all expected headers preserved",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["matched_header_count"] == 2
    assert payload["expected_header_count"] == 2
    assert payload["operator_approval_status"] == "operator_approved"
    assert updated["captured_headers"] == {
        "Cache-Control": "no-store",
        "X-Frame-Options": "DENY",
    }
    assert updated["checks"]["cache_control_preserved"] is True
    assert updated["checks"]["x_frame_options_preserved"] is True
    assert updated["checks"]["proxy_strips_or_weakens_no_headers"] is True
    assert updated["capture"]["url"] == "https://labeling.example.org/datasets?expected_user=alice"
    assert updated["capture"]["authenticated_test_user"] == "alice"
    assert updated["capture"]["proxy_or_deployment"] == "labeling proxy"
    assert updated["operator_approval"]["status"] == "operator_approved"
    assert updated["operator_approval"]["operator"] == "ops"
    assert updated["operator_approval"]["approved_at_utc"]
    status = inspection["operator_evidence_template_statuses"]["browser_response_security_headers"]
    assert status["ready"] is True
    assert status["approval_status"] == "operator_approved"
    assert status["missing_headers"] == []
    assert status["missing_checks"] == []
    assert status["operator_approval_missing"] is False
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == []


def test_inspection_failure_actions_report_missing_browser_response_security_evidence(tmp_path):
    evidence_path = tmp_path / "browser-response-security-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "browser_response_security_headers",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "browser_response_security_evidence_template": str(evidence_path),
            }
        )
        + "\n"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_browser_response_security_evidence_template.v1",
                "expected_headers": {
                    "Cache-Control": "no-store",
                    "X-Frame-Options": "DENY",
                },
                "captured_headers": {
                    "Cache-Control": "no-store",
                    "X-Frame-Options": "",
                },
                "checks": {
                    "cache_control_preserved": True,
                    "x_frame_options_preserved": False,
                },
                "operator_approval": {
                    "status": "pending_operator_confirmation",
                },
            }
        )
        + "\n"
    )

    inspection = _inspect_handoff_validation_checklist(tmp_path)

    status = inspection["operator_evidence_template_statuses"]["browser_response_security_headers"]
    assert status["ready"] is False
    assert status["missing_headers"] == ["X-Frame-Options"]
    assert status["missing_checks"] == ["x_frame_options_preserved"]
    assert status["operator_approval_missing"] is True
    failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist=inspection,
    )
    failure_action_text = "\n".join(failure_actions)
    assert "Browser response-security evidence incomplete" in failure_action_text
    assert "missing headers X-Frame-Options" in failure_action_text
    assert "missing checks x_frame_options_preserved" in failure_action_text
    assert "--header 'X-Frame-Options=VALUE'" in failure_action_text
    assert "record-browser-response-security-evidence" in failure_action_text
    repair_command_rows = {row["id"]: row for row in _inspection_operator_repair_commands(failure_actions)}
    assert repair_command_rows["apply_operator_evidence_templates"]["category"] == "validation_checklist"
    assert repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == [
        "browser_response_security_headers"
    ]
    assert "DEPLOYED_MY_DATASETS_URL_WITH_EXPECTED_USER" in repair_command_rows[
        "record_browser_response_security_evidence"
    ]["command"]
    assert "SAME_USER_AS_EXPECTED_USER" in repair_command_rows[
        "record_browser_response_security_evidence"
    ]["command"]


def test_record_browser_smoke_evidence_cli_updates_template_and_inspection(tmp_path, capsys):
    evidence_path = tmp_path / "browser-smoke-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "browser_smoke",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "browser_smoke_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_browser_smoke_evidence_template.v1",
                "personalized_route_smoke_contract": _browser_smoke_personalized_route_contract(),
                "counts": {
                    "candidate_users": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "users": [
                    {
                        "expected_user": "alice",
                        "run_status": "pending_operator_confirmation",
                        "resolved_user": "",
                        "identity_matches_expected_user": False,
                        "browser_only_runtime_verified": False,
                        "no_local_palette_install_verified": False,
                        "no_local_crimson_install_verified": False,
                        "no_local_conda_or_project_dependencies_verified": False,
                        "personalized_dataset_queue_verified": False,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": False,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": False,
                        "personalized_work_dashboard_verified": False,
                        "labeler_sees_only_assigned_work": False,
                        "support_text_redacted": False,
                        "expected_user_mismatch_rejected": False,
                        "task_opened": False,
                        "induced_failure_support_detail_redacted": False,
                        "completion_verified": False,
                        "completed_task_read_only_verified": False,
                        "stale_tab_save_rejected": False,
                        "operator_reopen_verified": False,
                        "captured_at_utc": "",
                        "operator": "",
                        "operator_approved_at_utc": "",
                        "notes": "",
                    }
                ],
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-browser-smoke-evidence",
            "--evidence",
            str(evidence_path),
            "--expected-user",
            "alice",
            "--resolved-user",
            "alice",
            "--operator",
            "ops",
            "--browser-only-runtime-verified",
            "--no-local-palette-install-verified",
            "--no-local-crimson-install-verified",
            "--no-local-conda-or-project-dependencies-verified",
            "--personalized-dataset-queue-verified",
            "--preferred-labeler-entry-url-matches-personal-dataset-queue",
            "--personalized-labeler-entry-url-matches-personal-dataset-queue",
            "--personalized-work-dashboard-verified",
            "--labeler-sees-only-assigned-work",
            "--support-text-redacted",
            "--expected-user-mismatch-rejected",
            "--task-opened",
            "--induced-failure-support-detail-redacted",
            "--completion-verified",
            "--completed-task-read-only-verified",
            "--stale-tab-save-rejected",
            "--operator-reopen-verified",
            "--notes",
            "queue-first smoke passed",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["counts"]["operator_approved"] == 1
    assert updated["counts"]["pending_operator_confirmation"] == 0
    assert updated["users"][0]["run_status"] == "operator_approved"
    assert updated["users"][0]["resolved_user"] == "alice"
    assert updated["users"][0]["identity_matches_expected_user"] is True
    assert updated["users"][0]["browser_only_runtime_verified"] is True
    assert updated["users"][0]["no_local_palette_install_verified"] is True
    assert updated["users"][0]["no_local_crimson_install_verified"] is True
    assert updated["users"][0]["no_local_conda_or_project_dependencies_verified"] is True
    assert updated["users"][0]["personalized_dataset_queue_verified"] is True
    assert updated["users"][0]["preferred_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert updated["users"][0]["personalized_labeler_entry_url_matches_personal_dataset_queue"] is True
    assert updated["users"][0]["personalized_work_dashboard_verified"] is True
    assert updated["users"][0]["operator"] == "ops"
    assert updated["users"][0]["operator_approved_at_utc"]
    assert updated["users"][0]["notes"] == "queue-first smoke passed"
    for field in (
        "browser_only_runtime_verified",
        "no_local_palette_install_verified",
        "no_local_crimson_install_verified",
        "no_local_conda_or_project_dependencies_verified",
        "personalized_dataset_queue_verified",
        "preferred_labeler_entry_url_matches_personal_dataset_queue",
        "personalized_labeler_entry_url_matches_personal_dataset_queue",
        "personalized_work_dashboard_verified",
        "labeler_sees_only_assigned_work",
        "support_text_redacted",
        "expected_user_mismatch_rejected",
        "task_opened",
        "induced_failure_support_detail_redacted",
        "completion_verified",
        "completed_task_read_only_verified",
        "stale_tab_save_rejected",
        "operator_reopen_verified",
    ):
        assert updated["users"][0][field] is True
    status = inspection["operator_evidence_template_statuses"]["browser_smoke"]
    assert status["ready"] is True
    assert status["approval_status"] == "operator_approved"
    assert status["user_statuses"][0]["resolved_user"] == "alice"
    assert status["user_statuses"][0]["resolved_user_matches_expected_user"] is True
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == []


def test_record_browser_smoke_evidence_cli_records_incomplete_run_without_approval(tmp_path, capsys):
    evidence_path = tmp_path / "browser-smoke-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "browser_smoke",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "browser_smoke_evidence_template": str(evidence_path),
            }
        )
        + "\n"
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_browser_smoke_evidence_template.v1",
                "personalized_route_smoke_contract": _browser_smoke_personalized_route_contract(),
                "counts": {
                    "candidate_users": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "users": [
                    {
                        "expected_user": "alice",
                        "run_status": "pending_operator_confirmation",
                        "resolved_user": "",
                        "identity_matches_expected_user": False,
                        "browser_only_runtime_verified": False,
                        "no_local_palette_install_verified": False,
                        "no_local_crimson_install_verified": False,
                        "no_local_conda_or_project_dependencies_verified": False,
                        "personalized_dataset_queue_verified": False,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": False,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": False,
                        "personalized_work_dashboard_verified": False,
                        "labeler_sees_only_assigned_work": False,
                        "support_text_redacted": False,
                        "expected_user_mismatch_rejected": False,
                        "task_opened": False,
                        "induced_failure_support_detail_redacted": False,
                        "completion_verified": False,
                        "completed_task_read_only_verified": False,
                        "stale_tab_save_rejected": False,
                        "operator_reopen_verified": False,
                        "captured_at_utc": "",
                        "operator": "",
                        "operator_approved_at_utc": "",
                        "notes": "",
                    }
                ],
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-browser-smoke-evidence",
            "--evidence",
            str(evidence_path),
            "--expected-user",
            "alice",
            "--resolved-user",
            "alice",
            "--operator",
            "ops",
            "--labeler-sees-only-assigned-work",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 2
    assert payload["ok"] is False
    assert payload["errors"][0]["error"] == "browser_smoke_checks_incomplete"
    assert "browser_only_runtime_verified" in payload["errors"][0]["missing_fields"]
    assert "no_local_palette_install_verified" in payload["errors"][0]["missing_fields"]
    assert "no_local_crimson_install_verified" in payload["errors"][0]["missing_fields"]
    assert "no_local_conda_or_project_dependencies_verified" in payload["errors"][0]["missing_fields"]
    assert "personalized_dataset_queue_verified" in payload["errors"][0]["missing_fields"]
    assert (
        "preferred_labeler_entry_url_matches_personal_dataset_queue"
        in payload["errors"][0]["missing_fields"]
    )
    assert (
        "personalized_labeler_entry_url_matches_personal_dataset_queue"
        in payload["errors"][0]["missing_fields"]
    )
    assert "personalized_work_dashboard_verified" in payload["errors"][0]["missing_fields"]
    assert "stale_tab_save_rejected" in payload["errors"][0]["missing_fields"]
    assert updated["counts"]["operator_approved"] == 0
    assert updated["counts"]["pending_operator_confirmation"] == 1
    assert updated["users"][0]["run_status"] == "pending_operator_confirmation"
    assert updated["users"][0]["resolved_user"] == "alice"
    assert updated["users"][0]["identity_matches_expected_user"] is True
    assert updated["users"][0]["labeler_sees_only_assigned_work"] is True
    assert updated["users"][0]["operator_approved_at_utc"] == ""
    status = inspection["operator_evidence_template_statuses"]["browser_smoke"]
    assert status["ready"] is False
    assert status["approval_status"] == "pending_operator_confirmation"
    assert status["user_statuses"][0]["expected_user"] == "alice"
    assert status["user_statuses"][0]["resolved_user_matches_expected_user"] is True
    assert "browser_only_runtime_verified" in status["user_statuses"][0]["missing_fields"]
    assert "no_local_palette_install_verified" in status["user_statuses"][0]["missing_fields"]
    assert (
        "preferred_labeler_entry_url_matches_personal_dataset_queue"
        in status["user_statuses"][0]["missing_fields"]
    )
    assert (
        "personalized_labeler_entry_url_matches_personal_dataset_queue"
        in status["user_statuses"][0]["missing_fields"]
    )
    assert "operator_approved_at_utc" in status["user_statuses"][0]["missing_fields"]
    assert status["users_missing_required_fields"][0]["expected_user"] == "alice"
    failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist=inspection,
    )
    failure_action_text = "\n".join(failure_actions)
    assert "Browser smoke evidence incomplete for alice" in failure_action_text
    assert "--browser-only-runtime-verified" in failure_action_text
    assert "--no-local-palette-install-verified" in failure_action_text
    assert "--no-local-crimson-install-verified" in failure_action_text
    assert "--no-local-conda-or-project-dependencies-verified" in failure_action_text
    assert "--labeler-sees-only-assigned-work" in failure_action_text
    assert "--stale-tab-save-rejected" in failure_action_text
    repair_command_rows = {row["id"]: row for row in _inspection_operator_repair_commands(failure_actions)}
    assert repair_command_rows["apply_operator_evidence_templates"]["category"] == "validation_checklist"
    assert repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == ["browser_smoke"]


def test_record_disposable_zarr_mutation_smoke_evidence_cli_updates_template_and_inspection(tmp_path, capsys):
    evidence_path = tmp_path / "disposable-zarr-mutation-smoke-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "disposable_zarr_mutation_smoke",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "disposable_zarr_mutation_smoke_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_disposable_zarr_mutation_smoke_evidence_template.v1",
                "counts": {
                    "workflow_kinds": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "workflows": [
                    {
                        "workflow_kind": "keypoints",
                        "status": "pending_operator_confirmation",
                        "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
                        "primary_mutation_target_kind": "task_scoped_training_zarr",
                        "source_mutation_target_kind": "",
                        "promotion_mutation_target_kind": "",
                        "training_zarr_mutation_target_kind": "task_scoped_training_zarr",
                        "browser_label_write_target": "training_zarr",
                        "training_zarr_write_mode": "direct",
                        "csv_handoff_artifact_role": "metadata_only_control_plane",
                        "csv_handoff_artifacts_are_label_write_targets": False,
                        "handoff_csv_artifacts_are_label_write_targets": False,
                        "intermediate_csv_artifacts_are_label_write_targets": False,
                        "handoff_artifacts_are_metadata_only": True,
                        "browser_writes_csv_or_handoff_files": False,
                        "browser_writes_handoff_csv": False,
                        "browser_writes_intermediate_csv": False,
                        "browser_receives_zarr_write_authority": False,
                        "browser_has_direct_zarr_write_authority": False,
                        "disposable_recording_id": "",
                        "disposable_task_id": "",
                        "labeler_user": "",
                        "disposable_zarr_or_known_good_source": "",
                        "backup_or_regeneration_verified": False,
                        "server_write_scope_verified": False,
                        "task_scoped_training_zarr_write_verified": False,
                        "browser_no_direct_zarr_write_authority_verified": False,
                        "handoff_artifacts_metadata_only_verified": False,
                        "browser_no_csv_or_handoff_write_verified": False,
                        "client_target_selector_rejection_verified": False,
                        "mutation_event_ids": [],
                        "audit_event_verified": False,
                        "operator_event_lookup_verified": False,
                        "operator_event_lookup_report_paths": [],
                        "operator_event_lookup_event_ids": [],
                        "registry_refresh_event_ids": [],
                        "completion_verified": False,
                        "stale_tab_save_rejected": False,
                        "bad_mutation_recovery_verified": False,
                        "bad_mutation_recovery_mode": "",
                        "bad_mutation_recovery_report": "",
                        "restored_or_discarded": False,
                        "operator": "",
                        "operator_approved_at_utc": "",
                        "notes": "",
                    }
                ],
            }
        )
        + "\n"
    )
    lookup_report_path = tmp_path / "event-1-lookup.json"
    lookup_report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "schema": "palette.web_labeling_audit_event_lookup.v1",
                "event_id": "event-1",
                "event": {
                    "event_id": "event-1",
                    "task_id": "task-smoke",
                    "recording_id": "rec-smoke",
                    "user": "alice",
                    "event_type": "save_keypoints",
                    "workflow_kind": "keypoints",
                    "target": {"roi_idx": 7},
                    "after": {"outcome": "saved"},
                },
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-disposable-zarr-mutation-smoke-evidence",
            "--evidence",
            str(evidence_path),
            "--workflow-kind",
            "keypoints",
            "--operator",
            "ops",
            "--mutation-event-id",
            "event-1",
            "--registry-refresh-event-id",
            "registry-1",
            "--disposable-recording-id",
            "rec-smoke",
            "--disposable-task-id",
            "task-smoke",
            "--labeler-user",
            "alice",
            "--disposable-zarr-or-known-good-source",
            "/operator/disposable/rec-smoke.zarr",
            "--backup-or-regeneration-verified",
            "--server-write-scope-verified",
            "--task-scoped-training-zarr-write-verified",
            "--browser-no-direct-zarr-write-authority-verified",
            "--handoff-artifacts-metadata-only-verified",
            "--browser-no-csv-or-handoff-write-verified",
            "--client-target-selector-rejection-verified",
            "--audit-event-verified",
            "--event-lookup-report",
            str(lookup_report_path),
            "--completion-verified",
            "--stale-tab-save-rejected",
            "--bad-mutation-recovery-verified",
            "--bad-mutation-recovery-mode",
            "discard_disposable",
            "--bad-mutation-recovery-report",
            "discarded disposable smoke zarr",
            "--restored-or-discarded",
            "--notes",
            "keypoint disposable mutation smoke passed",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 0
    assert payload["ok"] is True
    assert payload["counts"]["operator_approved"] == 1
    assert payload["event_lookup_report_paths"] == [str(lookup_report_path)]
    assert payload["event_lookup_event_ids"] == ["event-1"]
    assert payload["operator_event_lookup_verified"] is True
    assert payload["labeler_user"] == "alice"
    assert payload["bad_mutation_recovery_mode"] == "discard_disposable"
    assert payload["bad_mutation_recovery_report"] == "discarded disposable smoke zarr"
    assert updated["counts"]["pending_operator_confirmation"] == 0
    row = updated["workflows"][0]
    assert row["status"] == "operator_approved"
    assert row["mutation_event_ids"] == ["event-1"]
    assert row["registry_refresh_event_ids"] == ["registry-1"]
    assert row["operator_event_lookup_report_paths"] == [str(lookup_report_path)]
    assert row["operator_event_lookup_event_ids"] == ["event-1"]
    assert row["bad_mutation_recovery_mode"] == "discard_disposable"
    assert row["bad_mutation_recovery_report"] == "discarded disposable smoke zarr"
    assert row["disposable_recording_id"] == "rec-smoke"
    assert row["disposable_task_id"] == "task-smoke"
    assert row["labeler_user"] == "alice"
    assert row["disposable_zarr_or_known_good_source"] == "/operator/disposable/rec-smoke.zarr"
    assert row["operator"] == "ops"
    assert row["operator_approved_at_utc"]
    assert row["notes"] == "keypoint disposable mutation smoke passed"
    for field in (
        "backup_or_regeneration_verified",
        "server_write_scope_verified",
        "task_scoped_training_zarr_write_verified",
        "browser_no_direct_zarr_write_authority_verified",
        "handoff_artifacts_metadata_only_verified",
        "browser_no_csv_or_handoff_write_verified",
        "client_target_selector_rejection_verified",
        "audit_event_verified",
        "operator_event_lookup_verified",
        "completion_verified",
        "stale_tab_save_rejected",
        "bad_mutation_recovery_verified",
        "restored_or_discarded",
    ):
        assert row[field] is True
    status = inspection["operator_evidence_template_statuses"]["disposable_zarr_mutation_smoke"]
    assert status["ready"] is True
    assert status["approval_status"] == "operator_approved"
    assert len(status["workflow_statuses"]) == 1
    workflow_status = status["workflow_statuses"][0]
    assert workflow_status["workflow_kind"] == "keypoints"
    assert workflow_status["status"] == "operator_approved"
    assert workflow_status["ready"] is True
    assert workflow_status["missing_fields"] == []
    assert workflow_status["workflow_contract_ready"] is True
    assert workflow_status["workflow_contract_missing_fields"] == []
    assert workflow_status["data_plane_write_target"] == "server_owned_assigned_task_zarr_scope"
    assert workflow_status["primary_mutation_target_kind"] == "task_scoped_training_zarr"
    assert workflow_status["training_zarr_mutation_target_kind"] == "task_scoped_training_zarr"
    assert workflow_status["training_zarr_write_mode"] == "direct"
    assert workflow_status["csv_handoff_artifact_role"] == "metadata_only_control_plane"
    assert workflow_status["mutation_event_ids_present"] is True
    assert status["workflows_missing_required_fields"] == []
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == []


def test_record_disposable_zarr_mutation_smoke_evidence_cli_records_incomplete_run_without_approval(tmp_path, capsys):
    evidence_path = tmp_path / "disposable-zarr-mutation-smoke-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "disposable_zarr_mutation_smoke",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "disposable_zarr_mutation_smoke_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_disposable_zarr_mutation_smoke_evidence_template.v1",
                "counts": {
                    "workflow_kinds": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "workflows": [
                    {
                        "workflow_kind": "keypoints",
                        "status": "pending_operator_confirmation",
                        "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
                        "primary_mutation_target_kind": "task_scoped_training_zarr",
                        "source_mutation_target_kind": "",
                        "promotion_mutation_target_kind": "",
                        "training_zarr_mutation_target_kind": "task_scoped_training_zarr",
                        "browser_label_write_target": "training_zarr",
                        "training_zarr_write_mode": "direct",
                        "csv_handoff_artifact_role": "metadata_only_control_plane",
                        "csv_handoff_artifacts_are_label_write_targets": False,
                        "handoff_csv_artifacts_are_label_write_targets": False,
                        "intermediate_csv_artifacts_are_label_write_targets": False,
                        "handoff_artifacts_are_metadata_only": True,
                        "browser_writes_csv_or_handoff_files": False,
                        "browser_writes_handoff_csv": False,
                        "browser_writes_intermediate_csv": False,
                        "browser_receives_zarr_write_authority": False,
                        "browser_has_direct_zarr_write_authority": False,
                        "backup_or_regeneration_verified": False,
                        "server_write_scope_verified": False,
                        "task_scoped_training_zarr_write_verified": False,
                        "browser_no_direct_zarr_write_authority_verified": False,
                        "handoff_artifacts_metadata_only_verified": False,
                        "browser_no_csv_or_handoff_write_verified": False,
                        "client_target_selector_rejection_verified": False,
                        "mutation_event_ids": [],
                        "audit_event_verified": False,
                        "operator_event_lookup_verified": False,
                        "completion_verified": False,
                        "stale_tab_save_rejected": False,
                        "bad_mutation_recovery_verified": False,
                        "restored_or_discarded": False,
                        "operator": "",
                        "operator_approved_at_utc": "",
                    }
                ],
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-disposable-zarr-mutation-smoke-evidence",
            "--evidence",
            str(evidence_path),
            "--workflow-kind",
            "keypoints",
            "--operator",
            "ops",
            "--mutation-event-id",
            "event-1",
            "--server-write-scope-verified",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())
    inspection = _inspect_handoff_validation_checklist(tmp_path)

    assert rc == 2
    assert payload["ok"] is False
    assert payload["errors"][0]["error"] == "disposable_zarr_mutation_smoke_checks_incomplete"
    assert "task_scoped_training_zarr_write_verified" in payload["errors"][0]["missing_fields"]
    assert "browser_no_direct_zarr_write_authority_verified" in payload["errors"][0]["missing_fields"]
    assert "handoff_artifacts_metadata_only_verified" in payload["errors"][0]["missing_fields"]
    assert "browser_no_csv_or_handoff_write_verified" in payload["errors"][0]["missing_fields"]
    assert "client_target_selector_rejection_verified" in payload["errors"][0]["missing_fields"]
    assert "stale_tab_save_rejected" in payload["errors"][0]["missing_fields"]
    assert "operator_event_lookup_verified" in payload["errors"][0]["missing_fields"]
    assert "bad_mutation_recovery_verified" in payload["errors"][0]["missing_fields"]
    row = updated["workflows"][0]
    assert row["status"] == "pending_operator_confirmation"
    assert row["mutation_event_ids"] == ["event-1"]
    assert row["server_write_scope_verified"] is True
    assert row["operator_approved_at_utc"] == ""
    assert updated["counts"]["operator_approved"] == 0
    assert updated["counts"]["pending_operator_confirmation"] == 1
    status = inspection["operator_evidence_template_statuses"]["disposable_zarr_mutation_smoke"]
    assert status["ready"] is False
    assert status["approval_status"] == "pending_operator_confirmation"
    assert status["workflow_statuses"][0]["workflow_kind"] == "keypoints"
    assert status["workflow_statuses"][0]["ready"] is False
    assert "operator_event_lookup_verified" in status["workflow_statuses"][0]["missing_fields"]
    assert "bad_mutation_recovery_verified" in status["workflow_statuses"][0]["missing_fields"]
    assert "mutation_event_ids" not in status["workflow_statuses"][0]["missing_fields"]
    assert status["workflow_statuses"][0]["mutation_event_ids_present"] is True
    assert status["workflows_missing_required_fields"][0]["workflow_kind"] == "keypoints"
    failure_actions = _inspection_failure_actions(
        ["validation_evidence_template_unapproved"],
        validation_checklist=inspection,
    )
    failure_action_text = "\n".join(failure_actions)
    assert "Disposable-Zarr mutation smoke evidence incomplete for keypoints" in failure_action_text
    assert "operator_event_lookup_verified" in failure_action_text
    assert "bad_mutation_recovery_verified" in failure_action_text
    assert "--task-scoped-training-zarr-write-verified" in failure_action_text
    assert "--browser-no-direct-zarr-write-authority-verified" in failure_action_text
    assert "--handoff-artifacts-metadata-only-verified" in failure_action_text
    assert "--browser-no-csv-or-handoff-write-verified" in failure_action_text
    assert "--client-target-selector-rejection-verified" in failure_action_text
    assert "--operator-event-lookup-verified" in failure_action_text
    assert "--bad-mutation-recovery-verified" in failure_action_text
    repair_command_rows = {row["id"]: row for row in _inspection_operator_repair_commands(failure_actions)}
    assert repair_command_rows["apply_operator_evidence_templates"]["category"] == "validation_checklist"
    assert repair_command_rows["apply_operator_evidence_templates"]["gate_ids"] == [
        "disposable_zarr_mutation_smoke"
    ]


def test_record_disposable_zarr_mutation_smoke_evidence_rejects_lookup_context_mismatch(tmp_path, capsys):
    evidence_path = tmp_path / "disposable-zarr-mutation-smoke-evidence-template.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_disposable_zarr_mutation_smoke_evidence_template.v1",
                "counts": {
                    "workflow_kinds": 1,
                    "pending_operator_confirmation": 1,
                    "operator_approved": 0,
                },
                "workflows": [
                    {
                        "workflow_kind": "keypoints",
                        "status": "pending_operator_confirmation",
                        "data_plane_write_target": "server_owned_assigned_task_zarr_scope",
                        "primary_mutation_target_kind": "task_scoped_training_zarr",
                        "source_mutation_target_kind": "",
                        "promotion_mutation_target_kind": "",
                        "training_zarr_mutation_target_kind": "task_scoped_training_zarr",
                        "browser_label_write_target": "training_zarr",
                        "training_zarr_write_mode": "direct",
                        "csv_handoff_artifact_role": "metadata_only_control_plane",
                        "csv_handoff_artifacts_are_label_write_targets": False,
                        "handoff_csv_artifacts_are_label_write_targets": False,
                        "intermediate_csv_artifacts_are_label_write_targets": False,
                        "handoff_artifacts_are_metadata_only": True,
                        "browser_writes_csv_or_handoff_files": False,
                        "browser_writes_handoff_csv": False,
                        "browser_writes_intermediate_csv": False,
                        "browser_receives_zarr_write_authority": False,
                        "browser_has_direct_zarr_write_authority": False,
                        "backup_or_regeneration_verified": False,
                        "server_write_scope_verified": False,
                        "task_scoped_training_zarr_write_verified": False,
                        "browser_no_direct_zarr_write_authority_verified": False,
                        "handoff_artifacts_metadata_only_verified": False,
                        "browser_no_csv_or_handoff_write_verified": False,
                        "client_target_selector_rejection_verified": False,
                        "mutation_event_ids": [],
                        "audit_event_verified": False,
                        "operator_event_lookup_verified": False,
                        "completion_verified": False,
                        "stale_tab_save_rejected": False,
                        "bad_mutation_recovery_verified": False,
                        "restored_or_discarded": False,
                        "operator": "",
                        "operator_approved_at_utc": "",
                    }
                ],
            }
        )
        + "\n"
    )
    lookup_report_path = tmp_path / "event-1-lookup.json"
    lookup_report_path.write_text(
        json.dumps(
            {
                "ok": True,
                "schema": "palette.web_labeling_audit_event_lookup.v1",
                "event_id": "event-1",
                "event": {
                    "event_id": "event-1",
                    "task_id": "task-smoke",
                    "recording_id": "rec-smoke",
                    "user": "bob",
                    "workflow_kind": "keypoints",
                },
            }
        )
        + "\n"
    )

    rc = labeling_work.main(
        [
            "record-disposable-zarr-mutation-smoke-evidence",
            "--evidence",
            str(evidence_path),
            "--workflow-kind",
            "keypoints",
            "--operator",
            "ops",
            "--mutation-event-id",
            "event-1",
            "--event-lookup-report",
            str(lookup_report_path),
            "--disposable-recording-id",
            "rec-smoke",
            "--disposable-task-id",
            "task-smoke",
            "--labeler-user",
            "alice",
            "--backup-or-regeneration-verified",
            "--server-write-scope-verified",
            "--task-scoped-training-zarr-write-verified",
            "--browser-no-direct-zarr-write-authority-verified",
            "--handoff-artifacts-metadata-only-verified",
            "--browser-no-csv-or-handoff-write-verified",
            "--client-target-selector-rejection-verified",
            "--audit-event-verified",
            "--completion-verified",
            "--stale-tab-save-rejected",
            "--bad-mutation-recovery-verified",
            "--restored-or-discarded",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(evidence_path.read_text())

    assert rc == 2
    assert payload["ok"] is False
    assert payload["errors"][0]["error"] == "event_lookup_report_context_mismatch"
    assert payload["errors"][0]["mismatches"] == [
        {"field": "user", "expected": "alice", "actual": "bob"}
    ]
    assert payload["operator_event_lookup_verified"] is False
    assert updated["workflows"][0]["status"] == "pending_operator_confirmation"


def test_apply_operator_evidence_templates_cli_marks_only_approved_template_gates(tmp_path, capsys):
    checklist_path = tmp_path / "validation-checklist.json"
    validation_log_path = tmp_path / "validation-log-template.md"
    identity_evidence_path = tmp_path / "identity-source-evidence-template.json"
    browser_smoke_evidence_path = tmp_path / "browser-smoke-evidence-template.json"
    checklist_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "identity_probe_verification",
                        "status": "pending_operator_evidence",
                        "required": True,
                    },
                    {
                        "id": "browser_smoke",
                        "status": "pending_operator_evidence",
                        "required": True,
                    },
                ],
                "identity_source_evidence_template": str(identity_evidence_path),
                "browser_smoke_evidence_template": str(browser_smoke_evidence_path),
            }
        )
        + "\n"
    )
    identity_evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_identity_source_evidence_template.v1",
                "users": [
                    {
                        "expected_user": "alice",
                        "expected_user_dataset_queue_url": "https://labeling.example.org/datasets?expected_user=alice",
                        "expected_user_personal_dataset_queue_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "preferred_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
                        "personalized_labeler_entry_url": "https://labeling.example.org/my-datasets?expected_user=alice",
                        "preferred_labeler_entry_url_matches_dataset_queue": True,
                        "preferred_labeler_entry_url_matches_personal_dataset_queue": True,
                        "personalized_labeler_entry_url_matches_personal_dataset_queue": True,
                        "resolved_user": "alice",
                        "identity_matches_expected_user": True,
                        "operator": "ops",
                        "operator_approved_at_utc": "2026-06-23T00:00:00+00:00",
                    }
                ],
            }
        )
        + "\n"
    )
    browser_smoke_evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_browser_smoke_evidence_template.v1",
                "personalized_route_smoke_contract": _browser_smoke_personalized_route_contract(),
                "users": [
                    {
                        "expected_user": "alice",
                        "run_status": "pending_operator_confirmation",
                        "identity_matches_expected_user": True,
                    }
                ],
            }
        )
        + "\n"
    )
    validation_log_path.write_text("# Web Labeling Validation Log\n")

    rc = labeling_work.main(
        [
            "apply-operator-evidence-templates",
            "--path",
            str(checklist_path),
            "--operator",
            "ops",
            "--append-log",
            str(validation_log_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    updated = json.loads(checklist_path.read_text())
    gate_statuses = {gate["id"]: gate["status"] for gate in updated["gates"]}
    identity_gate = next(gate for gate in updated["gates"] if gate["id"] == "identity_probe_verification")
    smoke_gate = next(gate for gate in updated["gates"] if gate["id"] == "browser_smoke")
    validation_log = validation_log_path.read_text()

    assert rc == 0
    assert payload["ok"] is True
    assert payload["applied_count"] == 1
    assert payload["applied"][0]["gate_id"] == "identity_probe_verification"
    assert gate_statuses["identity_probe_verification"] == "passed"
    assert gate_statuses["browser_smoke"] == "pending_operator_evidence"
    assert identity_gate["evidence_recorded_by"] == "ops"
    assert str(identity_evidence_path) in identity_gate["evidence_files"]
    assert "Approved operator evidence template applied" in identity_gate["evidence_notes"][0]
    assert smoke_gate.get("evidence_files") in (None, [])
    assert "Validation Evidence: identity_probe_verification" in validation_log
    assert "identity-source-evidence-template.json" in validation_log
    assert payload["operator_evidence_pending_gate_ids"] == ["browser_smoke"]
    assert payload["generated_contract_failed_gate_ids"] == []
    assert "browser_smoke" in payload["validation_summary"]["required_pending_gate_ids"]
    assert payload["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
    assert payload["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert payload["safe_share_ready_to_send_is_sufficient"] is False
    assert payload["safe_share_checklist_gate_evidence_complete"] is False
    assert "identity_probe_verification" in payload["safe_share_launch_blocking_satisfied_gate_ids"]
    assert "browser_smoke" in payload["safe_share_launch_blocking_pending_gate_ids"]
    assert updated["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert updated["safe_share_checklist_gate_evidence_complete"] is False
    assert payload["validation_summary"]["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert payload["validation_summary"]["safe_share_ready_to_send_is_sufficient"] is False
    assert payload["validation_summary"]["safe_share_checklist_gate_evidence_complete"] is False
    assert payload["handoff_refresh_enabled"] is True
    assert payload["handoff_refresh"]["manifest_count"] == 0
    assert payload["handoff_refresh_refreshed_manifest_count"] == 0
    assert payload["handoff_refresh_refreshed_file_count"] == 0
    assert payload["handoff_refresh_refreshed_files"] == []
    assert payload["handoff_refresh_refreshed_visible_json_file_count"] == 0
    assert payload["handoff_refresh_refreshed_visible_json_files"] == []
    assert payload["handoff_refresh_skipped_count"] == 0
    assert payload["handoff_refresh_skipped"] == []
    assert payload["handoff_refresh"]["skipped_count"] == 0
    assert payload["handoff_refresh"]["skipped"] == []
    assert payload["checksum_refresh_required"] is False
    assert payload["checksum_refresh_command"] == ""


def test_refresh_handoff_checksums_cli_records_auditable_refresh(tmp_path, capsys):
    package_dir = tmp_path / "launch-bundle"
    package_dir.mkdir()
    evidence_path = package_dir / "identity-source-evidence-template.json"
    evidence_path.write_text('{"status":"pending"}\n')
    (package_dir / "validation-checklist.json").write_text('{"schema":"palette.web_labeling_validation_checklist.v1"}\n')
    _write_directory_checksums(package_dir, package_dir / "checksums.json")
    previous_checksum_bytes = (package_dir / "checksums.json").read_bytes()
    previous_checksum_sha256 = hashlib.sha256(previous_checksum_bytes).hexdigest()

    evidence_path.write_text('{"status":"operator_approved"}\n')
    rc = labeling_work.main(
        [
            "refresh-handoff-checksums",
            "--path",
            str(package_dir),
            "--operator",
            "ops",
            "--reason",
            "operator evidence update",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    checksums = json.loads((package_dir / "checksums.json").read_text())
    rows = {row["path"]: row for row in checksums["files"]}
    refresh_log_entries = [
        json.loads(line)
        for line in (package_dir / "checksums-refresh-log.jsonl").read_text().splitlines()
    ]

    assert rc == 0
    assert payload["ok"] is True
    assert payload["refresh"]["operator"] == "ops"
    assert payload["refresh"]["reason"] == "operator evidence update"
    assert payload["refresh"]["previous_checksums_sha256"] == previous_checksum_sha256
    assert rows["identity-source-evidence-template.json"]["sha256"] == hashlib.sha256(
        b'{"status":"operator_approved"}\n'
    ).hexdigest()
    assert "checksums-refresh-log.jsonl" in rows
    assert checksums["refresh"]["previous_checksums_sha256"] == previous_checksum_sha256
    assert refresh_log_entries[-1]["operator"] == "ops"


def test_inspect_handoff_validation_checklist_accepts_backup_execution_manifest_evidence(tmp_path):
    evidence_path = tmp_path / "zarr-backup-evidence-template.json"
    (tmp_path / "validation-checklist.json").write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_validation_checklist.v1",
                "gates": [
                    {
                        "id": "mutable_zarr_backup_confirmation",
                        "status": "passed",
                        "required": True,
                    }
                ],
                "zarr_backup_evidence_template": str(evidence_path),
            }
        )
    )
    evidence_path.write_text(
        json.dumps(
            {
                "schema": "palette.web_labeling_zarr_backup_evidence_template.v1",
                "targets": [
                    {
                        "status": "operator_approved",
                        "backup_execution_manifest_path": "/operator/zarr-backups/manifest.json",
                        "backup_verified_at_utc": "2026-06-23T00:00:00+00:00",
                        "restore_test_result": "passed",
                        "operator_approved_at_utc": "2026-06-23T00:05:00+00:00",
                    }
                ],
            }
        )
    )

    inspection = _inspect_handoff_validation_checklist(tmp_path)

    status = inspection["operator_evidence_template_statuses"]["mutable_zarr_backup_confirmation"]
    assert status["present"] is True
    assert status["ready"] is True
    assert status["approval_status"] == "operator_approved"
    assert "mutable_zarr_backup_confirmation" in inspection["operator_evidence_template_approved_gate_ids"]
    assert inspection["passed_gates_with_unapproved_evidence_templates"] == []


def test_export_launch_bundle_dry_run_reports_plan_without_secret_or_writes(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "launch-bundle"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice")
        store.assign_recording(recording_id="rec-b", assignee_user="bob")
        store.upsert_task(task_id="task-a", recording_id="rec-a", workflow_kind="keypoints")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-launch-bundle",
            "--dry-run",
            "--base-url",
            "https://labeling.example.org",
            "--output-dir",
            str(output_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["labeler_landing_page_path"] == "/"
    assert payload["labeler_landing_url"] == "https://labeling.example.org"
    assert payload["dashboard_path"] == "/work"
    assert payload["dashboard_url"] == "https://labeling.example.org/work"
    assert payload["labeler_safety"]["dashboard_identity_check_required"] is True
    assert payload["counts"]["users"] == 2
    assert payload["counts"]["zarr_backup_targets"] == 0
    assert payload["counts"]["zarr_backup_required_targets"] == 0
    assert payload["counts"]["zarr_backup_targets_by_role"] == {}
    assert payload["counts"]["zarr_backup_required_targets_by_role"] == {}
    assert payload["counts"]["zarr_backup_missing_path_tasks"] == 1
    assert payload["zarr_backup_plan_summary"]["schema"] == "palette.web_labeling_zarr_backup_plan.v1"
    assert payload["zarr_backup_policy"]["validation_gate"] == "mutable_zarr_backup_confirmation"
    assert payload["zarr_backup_policy"]["copy_before_labeling"] is True
    assert payload["mutation_audit_policy"]["event_store"] == "labeling_task_events"
    assert payload["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert payload["browser_mutation_write_policy"]["mutable_label_data_plane"] == "task_scoped_training_zarr"
    assert payload["browser_mutation_write_policy"]["training_zarr_mutations_are_server_owned"] is True
    assert payload["browser_mutation_write_policy"]["handoff_artifacts_are_metadata_only"] is True
    assert payload["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert payload["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert payload["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert payload["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert payload["browser_mutation_write_checklist"]["ready"] is True
    assert payload["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert payload["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert payload["labeler_route_authorization_policy"]["expected_user_must_match_resolved_user"] is True
    assert payload["labeler_route_authorization_policy"]["task_open_requires_active_assignment"] is True
    assert payload["labeler_route_authorization_policy"]["task_open_requires_startable_task_state"] is True
    assert payload["labeler_route_authorization_policy"]["startable_task_states"] == [
        "pending",
        "in_progress",
    ]
    assert payload["labeler_route_authorization_policy"]["signed_links_are_entry_hints_not_authorization"] is True
    assert payload["mutation_audit_policy"]["validation_gate"] == "disposable_zarr_mutation_smoke"
    assert payload["session_guard_policy"]["requires_current_session"] is True
    assert payload["session_guard_policy"]["target_token_required_for_mutation"] is True
    assert payload["session_guard_policy"]["labeler_promotion_retry_requires_current_session"] is True
    assert payload["session_guard_policy"]["closure_event_field"] == "session_closure_event"
    assert payload["zarr_backup_plan_summary"]["counts"]["tasks_missing_zarr_path"] == 1
    assert payload["warning_count"] >= 1
    assert payload["handoff_users"] == [
        {"user": "alice", "output_dir": str(output_dir / "handoffs" / "alice")},
        {"user": "bob", "output_dir": str(output_dir / "handoffs" / "bob")},
    ]
    assert payload["planned_files"]["manifest"] == str(output_dir / "manifest.json")
    assert payload["planned_files"]["zarr_backup_plan"] == str(output_dir / "zarr-backup-plan.json")
    assert payload["planned_files"]["zarr_backup_evidence_template"] == str(
        output_dir / "zarr-backup-evidence-template.json"
    )
    assert payload["planned_files"]["browser_response_security_evidence_template"] == str(
        output_dir / "browser-response-security-evidence-template.json"
    )
    assert payload["planned_files"]["identity_source_evidence_template"] == str(
        output_dir / "identity-source-evidence-template.json"
    )
    assert payload["planned_files"]["browser_smoke_evidence_template"] == str(
        output_dir / "browser-smoke-evidence-template.json"
    )
    assert payload["planned_files"]["disposable_zarr_mutation_smoke_evidence_template"] == str(
        output_dir / "disposable-zarr-mutation-smoke-evidence-template.json"
    )
    assert payload["planned_files"]["operator_evidence_commands"] == str(
        output_dir / "operator-evidence-commands.txt"
    )
    assert payload["planned_files"]["launch_evidence_execution_checklist"] == str(
        output_dir / "launch-evidence-execution-checklist.txt"
    )
    assert payload["planned_files"]["handoffs_html_index"] == str(output_dir / "handoffs" / "index.html")
    assert payload["planned_files"]["handoffs_roster"] == str(output_dir / "handoffs" / "labeler-roster.csv")
    assert payload["labeling_home_page_path"] == "/labeling"
    assert payload["labeling_home_url"] == "https://labeling.example.org/labeling"
    assert payload["implementation_status_artifact_schema"] == (
        "palette.web_labeling_implementation_status_artifact.v1"
    )
    assert payload["implementation_status_artifact"]["schema"] == (
        "palette.web_labeling_implementation_status_artifact.v1"
    )
    assert payload["implementation_status_artifact"]["checklist_declared_path"] == str(
        output_dir / "implementation-status.txt"
    )
    assert payload["implementation_status_artifact"]["file"] == "implementation-status.txt"
    assert payload["implementation_status_artifact"]["is_launch_approval"] is False
    assert payload["implementation_status_artifact"][
        "operator_evidence_required_before_share"
    ] is True
    expected_dry_run_implementation_status_artifact_required_fields = [
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
    assert payload["implementation_status_artifact_required_fields"] == (
        expected_dry_run_implementation_status_artifact_required_fields
    )
    assert payload["implementation_status_artifact_required_field_count"] == len(
        expected_dry_run_implementation_status_artifact_required_fields
    )
    expected_dry_run_implementation_status_flat_fields = [
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
    assert payload["implementation_status_flat_fields"] == (
        expected_dry_run_implementation_status_flat_fields
    )
    assert payload["implementation_status_flat_field_count"] == len(
        expected_dry_run_implementation_status_flat_fields
    )
    assert payload["implementation_status_file"] == "implementation-status.txt"
    assert payload["implementation_status_role"] == (
        "bundle_local_implementation_evidence_status_summary"
    )
    assert payload["implementation_status_is_launch_approval"] is False
    assert payload["implementation_status_operator_evidence_required_before_share"] is True
    assert payload["implementation_status_safe_share_gate"] == "labeler_links_safe_to_share"
    assert payload["implementation_status_safe_share_required_inspection_field"] == (
        "labeler_links_safe_to_share"
    )
    assert payload["implementation_status_safe_share_required_inspection_value"] is True
    assert payload["implementation_status_require_shareable_inspection_before_share"] is True
    dry_run_gate_statuses = {gate["id"]: gate["status"] for gate in payload["validation_checklist"]["gates"]}
    assert payload["validation_checklist"]["schema"] == "palette.web_labeling_validation_checklist.v1"
    assert payload["validation_checklist"]["bundle_label"] == "launch bundle dry-run"
    assert payload["validation_checklist"]["dry_run"] is True
    assert payload["validation_checklist"]["implementation_status"] == str(
        output_dir / "implementation-status.txt"
    )
    assert payload["validation_checklist"]["implementation_status_artifact"]["schema"] == (
        "palette.web_labeling_implementation_status_artifact.v1"
    )
    assert payload["validation_checklist"]["implementation_status_artifact"][
        "is_launch_approval"
    ] is False
    assert payload["validation_checklist"]["implementation_status_artifact_required_fields"] == (
        expected_dry_run_implementation_status_artifact_required_fields
    )
    assert payload["validation_checklist"][
        "implementation_status_artifact_required_field_count"
    ] == len(expected_dry_run_implementation_status_artifact_required_fields)
    assert payload["validation_checklist"]["implementation_status_flat_fields"] == (
        expected_dry_run_implementation_status_flat_fields
    )
    assert payload["validation_checklist"]["implementation_status_flat_field_count"] == len(
        expected_dry_run_implementation_status_flat_fields
    )
    assert payload["validation_checklist"]["implementation_status_file"] == (
        "implementation-status.txt"
    )
    assert payload["validation_checklist"]["implementation_status_is_launch_approval"] is False
    assert payload["validation_checklist"][
        "implementation_status_operator_evidence_required_before_share"
    ] is True
    assert payload["validation_checklist"]["implementation_status_safe_share_gate"] == (
        "labeler_links_safe_to_share"
    )
    assert payload["validation_checklist"][
        "implementation_status_safe_share_required_inspection_field"
    ] == "labeler_links_safe_to_share"
    assert payload["validation_checklist"][
        "implementation_status_safe_share_required_inspection_value"
    ] is True
    assert payload["validation_checklist"][
        "implementation_status_require_shareable_inspection_before_share"
    ] is True
    assert payload["validation_checklist"]["counts"]["implementation_status_present"] == 1
    assert payload["validation_checklist"]["validation_log"] == str(output_dir / "validation-log-template.md")
    assert payload["validation_checklist"]["labeler_landing_page_path"] == "/"
    assert payload["validation_checklist"]["labeler_landing_url"] == "https://labeling.example.org"
    assert payload["validation_checklist"]["labeling_home_page_path"] == "/labeling"
    assert payload["validation_checklist"]["labeling_home_url"] == "https://labeling.example.org/labeling"
    assert payload["validation_checklist"]["dataset_queue_page_path"] == "/datasets"
    assert payload["validation_checklist"]["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert payload["validation_checklist"]["queue_first_entry_contract"]["ready"] is True
    assert payload["validation_checklist"]["queue_first_entry_contract"]["labeling_home_ready"] is True
    assert payload["validation_checklist"]["queue_first_entry_contract"]["labeling_home_url"] == (
        "https://labeling.example.org/labeling"
    )
    assert payload["validation_checklist"]["queue_first_entry_contract"]["dashboard_ready"] is True
    assert payload["validation_checklist"]["queue_first_entry_contract"]["labeler_landing_page_kind"] == (
        "datasets_waiting_queue"
    )
    assert (
        payload["validation_checklist"]["queue_first_entry_contract"][
            "landing_serves_datasets_waiting_queue"
        ]
        is True
    )
    assert payload["validation_checklist"]["queue_first_entry_contract"]["datasets_waiting_aliases_ready"] is True
    assert payload["validation_checklist"]["identity_probe_link_contract"]["ready"] is True
    assert payload["validation_checklist"]["identity_probe_link_contract"]["identity_check_required"] is True
    assert payload["validation_checklist"]["identity_probe_link_contract"]["expected_user_identity_probe_url_present"] is False
    assert payload["validation_checklist"]["identity_probe_link_contract"]["batch_identity_probe_evidence_present"] is True
    assert payload["validation_checklist"]["counts"]["identity_probe_link_contract_ready"] == 1
    assert payload["validation_checklist"]["single_owner_policy"]["one_active_owner"] is True
    assert payload["validation_checklist"]["single_owner_policy"][
        "multiple_labelers_per_recording_allowed"
    ] is False
    assert payload["validation_checklist"]["operator_authorization_contract"]["ready"] is True
    assert payload["validation_checklist"]["operator_authorization_contract"]["labelers_are_not_operators_by_default"] is True
    assert payload["validation_checklist"]["operator_recovery_contract"]["ready"] is True
    assert payload["validation_checklist"]["operator_recovery_contract"]["task_reopen_operator_only"] is True
    assert payload["validation_checklist"]["operator_recovery_contract"]["audit_event_lookup_route"] == (
        "/api/admin/events/{event_id}"
    )
    assert payload["validation_checklist"]["operator_recovery_contract"]["operator_repair_records_audit_event"] is True
    assert payload["validation_checklist"]["operator_recovery_contract"]["bad_disposable_mutation_recovery_ready"] is True
    assert payload["validation_checklist"]["operator_recovery_contract"][
        "disposable_mutation_smoke_requires_recovery_path_verification"
    ] is True
    assert payload["validation_checklist"]["operator_recovery_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert payload["validation_checklist"]["operator_recovery_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert payload["validation_checklist"]["counts"]["operator_recovery_contract_ready"] == 1
    assert payload["validation_checklist"]["browser_payload_redaction_contract"]["ready"] is True
    assert payload["validation_checklist"]["browser_payload_redaction_contract"]["browser_receives_raw_zarr_paths"] is False
    assert payload["validation_checklist"]["assignment_ownership_contract"]["ready"] is True
    assert payload["validation_checklist"]["assignment_ownership_contract"][
        "assignment_scope"
    ] == "recording"
    assert payload["validation_checklist"]["assignment_ownership_contract"][
        "recording_assignment_key"
    ] == "recording_id"
    assert payload["validation_checklist"]["assignment_ownership_contract"][
        "multiple_labelers_per_recording_allowed"
    ] is False
    assert payload["validation_checklist"]["assignment_ownership_contract"][
        "browser_mutation_requires_current_assignment_owner"
    ] is True
    assert payload["validation_checklist"]["assignment_ownership_contract"]["duplicate_active_owner_count"] == 0
    assert payload["validation_checklist"]["operator_authorization_policy"]["admin_routes_require_operator"] is True
    assert payload["validation_checklist"]["labeler_route_authorization_policy"]["expected_user_must_match_resolved_user"] is True
    assert payload["validation_checklist"]["labeler_route_authorization_contract"]["ready"] is True
    assert payload["validation_checklist"]["signed_link_policy"]["authorization_grant"] is False
    assert payload["validation_checklist"]["signed_link_contract"]["ready"] is True
    assert payload["validation_checklist"]["signed_link_contract"]["runtime_operator_validation_start_gate_enforced"] is True
    assert payload["validation_checklist"]["signed_link_contract"]["operator_validation_start_gate_checked_before_session_create"] is True
    assert payload["validation_checklist"]["expected_user_guard_contract"]["ready"] is True
    assert payload["validation_checklist"]["expected_user_guard_contract"]["missing_or_mismatched_guards"] == []
    assert payload["validation_checklist"]["expected_user_guard_contract"][
        "promotion_retry_guarded_support_only"
    ] is True
    assert payload["validation_checklist"]["expected_user_guard_contract"][
        "promotion_retry_labeler_mutation_enabled"
    ] is False
    assert payload["validation_checklist"]["expected_user_guard_contract"][
        "promotion_retry_labeler_rejection_error"
    ] == "operator_support_required"
    assert payload["validation_checklist"]["mutation_audit_contract"]["ready"] is True
    assert payload["validation_checklist"]["mutation_audit_contract"]["event_store"] == "labeling_task_events"
    assert payload["validation_checklist"]["zarr_backup_contract"]["ready"] is True
    assert payload["validation_checklist"]["zarr_backup_contract"]["labelers_do_not_receive_backup_paths"] is True
    assert payload["validation_checklist"]["session_guard_contract"]["ready"] is True
    assert payload["validation_checklist"]["session_guard_contract"]["stale_tab_save_rejected"] is True
    assert payload["validation_checklist"]["session_guard_contract"]["non_startable_task_sessions_rejected"] is True
    assert payload["validation_checklist"]["task_state_contract"]["ready"] is True
    assert payload["validation_checklist"]["task_state_contract"]["startable_task_states"] == ["pending", "in_progress"]
    assert payload["validation_checklist"]["task_state_contract"]["completed_tasks_read_only"] is True
    assert payload["validation_checklist"]["task_state_contract"]["non_startable_task_open_requests"] == (
        "reject_task_not_startable"
    )
    assert payload["validation_checklist"]["task_state_contract"]["non_startable_task_save_requests"] == (
        "reject_task_not_startable"
    )
    assert payload["validation_checklist"]["task_state_contract"][
        "labeler_promotion_retry_mutation_enabled"
    ] is False
    assert payload["validation_checklist"]["task_state_contract"][
        "labeler_promotion_retry_rejection_error"
    ] == "operator_support_required"
    assert payload["validation_checklist"]["task_state_contract"][
        "ordinary_labeler_promotion_retry_mutation"
    ] == "operator_support_required"
    assert payload["validation_checklist"]["browser_workflow_scope_contract"]["ready"] is True
    assert payload["validation_checklist"]["browser_workflow_scope_contract"]["workflows_with_browser_filesystem_authority"] == []
    assert payload["validation_checklist"]["browser_response_security_policy"]["proxy_must_preserve_headers"] is True
    assert payload["validation_checklist"]["browser_response_security_contract"]["ready"] is True
    assert payload["validation_checklist"]["browser_response_security_contract"]["referrer_leakage_protection"] is True
    assert payload["validation_checklist"]["task_state_policy"]["browser_mutation_target_token"] == "required_current_target_token"
    assert payload["validation_checklist"]["browser_mutation_target_contract"]["ready"] is True
    assert payload["validation_checklist"]["browser_mutation_write_policy"]["authoritative_label_state"] == (
        "assigned_task_zarr_scope"
    )
    assert payload["validation_checklist"]["browser_mutation_write_policy"]["mutable_label_data_plane"] == (
        "task_scoped_training_zarr"
    )
    assert payload["validation_checklist"]["browser_mutation_write_policy"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["validation_checklist"]["browser_mutation_write_policy"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert (
        payload["validation_checklist"]["browser_mutation_write_policy"][
            "csv_handoff_artifacts_are_label_write_targets"
        ]
        is False
    )
    assert (
        payload["validation_checklist"]["browser_mutation_write_policy"][
            "training_zarr_mutations_are_server_owned"
        ]
        is True
    )
    assert payload["validation_checklist"]["browser_mutation_write_contract"]["ready"] is True
    assert payload["validation_checklist"]["browser_mutation_write_contract"]["mutable_label_data_plane"] == (
        "task_scoped_training_zarr"
    )
    assert payload["validation_checklist"]["browser_mutation_write_contract"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert payload["validation_checklist"]["browser_mutation_write_contract"]["handoff_artifacts_are_metadata_only"] is True
    assert payload["validation_checklist"]["browser_mutation_write_contract"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert (
        payload["validation_checklist"]["browser_mutation_write_contract"][
            "csv_handoff_artifacts_are_label_write_targets"
        ]
        is False
    )
    assert payload["validation_checklist"]["assignment_ownership_integrity"]["ok"] is True
    assert payload["validation_checklist"]["counts"]["assignment_ownership_duplicate_active_owners"] == 0
    assert dry_run_gate_statuses["static_readiness"] == "pending_operator_evidence"
    assert dry_run_gate_statuses["queue_first_entry_contract"] == "passed"
    assert dry_run_gate_statuses["identity_probe_link_contract"] == "passed"
    assert dry_run_gate_statuses["assignment_ownership_contract"] == "passed"
    assert dry_run_gate_statuses["browser_payload_redaction_contract"] == "passed"
    assert dry_run_gate_statuses["labeler_route_authorization"] == "passed"
    assert dry_run_gate_statuses["signed_link_contract"] == "passed"
    assert dry_run_gate_statuses["expected_user_guard_contract"] == "passed"
    assert dry_run_gate_statuses["session_guard_contract"] == "passed"
    assert dry_run_gate_statuses["task_state_contract"] == "passed"
    assert dry_run_gate_statuses["operator_authorization_contract"] == "passed"
    assert dry_run_gate_statuses["operator_recovery_contract"] == "passed"
    assert dry_run_gate_statuses["browser_response_security_contract"] == "passed"
    assert dry_run_gate_statuses["browser_workflow_scope_contract"] == "passed"
    assert dry_run_gate_statuses["mutation_audit_contract"] == "passed"
    assert dry_run_gate_statuses["zarr_backup_contract"] == "passed"
    assert dry_run_gate_statuses["browser_mutation_target_contract"] == "passed"
    assert dry_run_gate_statuses["browser_mutation_write_policy"] == "passed"
    assert "Dry-run preview only" in next(
        gate for gate in payload["validation_checklist"]["gates"] if gate["id"] == "static_readiness"
    )["details"]
    assert dry_run_gate_statuses["identity_probe_verification"] == "pending_operator_evidence"
    assert dry_run_gate_statuses["operator_authorization_boundary"] == "pending_operator_evidence"
    assert dry_run_gate_statuses["browser_response_security_headers"] == "pending_operator_evidence"
    assert dry_run_gate_statuses["dashboard_visibility"] == "pending_operator_evidence"
    assert dry_run_gate_statuses["mutable_zarr_backup_confirmation"] == "needs_review"
    assert not output_dir.exists()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-launch-bundle",
            "--dry-run",
            "--warnings-as-errors",
            "--base-url",
            "https://labeling.example.org",
            "--output-dir",
            str(output_dir),
        ]
    )

    strict_payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert strict_payload["ok"] is False
    assert strict_payload["warnings_as_errors"] is True
    assert strict_payload["blocking_warning_count"] >= 1
    assert not output_dir.exists()


def test_export_user_handoffs_cli_writes_batch_index_and_user_dirs(tmp_path, capsys):
    store_path = tmp_path / "labeling_work.sqlite"
    output_dir = tmp_path / "handoffs"
    zip_path = tmp_path / "handoffs.zip"
    store = LabelingStore(store_path)
    try:
        store.initialize()
        store.assign_recording(recording_id="rec-a", assignee_user="alice", notes="Alice instructions")
        store.assign_recording(recording_id="rec-empty", assignee_user="alice", notes="Waiting for task generation")
        store.assign_recording(recording_id="rec-b", assignee_user="bob", notes="Bob instructions")
        store.assign_recording(recording_id="rec-c", assignee_user="carol", status="inactive")
        store.upsert_task(
            task_id="task-a",
            recording_id="rec-a",
            workflow_kind="keypoints",
            title="Alice task",
            scope={"zarr_path": "/secret/alice-training.zarr"},
        )
        store.upsert_task(task_id="task-b", recording_id="rec-b", workflow_kind="detect_analysis", title="Bob task")
        store.upsert_task(task_id="task-c", recording_id="rec-c", workflow_kind="subject_mask_component", title="Carol inactive")
    finally:
        store.close()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "export-user-handoffs",
            "--link-secret",
            "test-secret",
            "--base-url",
            "https://labeling.example.org",
            "--output-dir",
            str(output_dir),
            "--zip-output",
            str(zip_path),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    index = json.loads((output_dir / "index.json").read_text())
    html_index = (output_dir / "index.html").read_text()
    readme = (output_dir / "handoff-readme.txt").read_text()
    validation_log = (output_dir / "validation-log-template.md").read_text()
    validation_checklist = json.loads((output_dir / "validation-checklist.json").read_text())
    roster_rows = list(csv.DictReader((output_dir / "labeler-roster.csv").open()))
    alice_manifest = json.loads((output_dir / "alice" / "manifest.json").read_text())
    bob_manifest = json.loads((output_dir / "bob" / "manifest.json").read_text())
    alice_dataset_queue = json.loads((output_dir / "alice" / "dataset-queue.json").read_text())
    bob_dataset_queue = json.loads((output_dir / "bob" / "dataset-queue.json").read_text())
    alice_html = (output_dir / "alice" / "index.html").read_text()
    alice_message = (output_dir / "alice" / "message.txt").read_text()
    alice_quickstart = (output_dir / "alice" / "labeler-quickstart.txt").read_text()
    alice_links = [json.loads(line) for line in (output_dir / "alice" / "signed-links.jsonl").read_text().splitlines()]
    bob_links = [json.loads(line) for line in (output_dir / "bob" / "signed-links.jsonl").read_text().splitlines()]

    assert rc == 2
    assert payload["ok"] is False
    assert payload["store_checks_ok"] is True
    assert payload["labeler_landing_page_path"] == "/"
    assert payload["labeler_landing_url"] == "https://labeling.example.org"
    assert payload["dashboard_path"] == "/work"
    assert payload["dashboard_url"] == "https://labeling.example.org/work"
    assert payload["dataset_queue_page_path"] == "/datasets"
    assert payload["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert payload["labeler_safety"]["dashboard_identity_check_required"] is True
    assert payload["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert payload["operator_validation_visibility_policy"]["operator_only_fields"] == [
        "operator_validation_checklist_path"
    ]
    assert index["operator_validation_visibility_policy"] == payload[
        "operator_validation_visibility_policy"
    ]
    assert payload["operator_validation_command_templates"]["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert payload["operator_validation_command_templates"]["commands_are_operator_only"] is True
    assert payload["operator_validation_command_templates"][
        "commands_are_labeler_instructions"
    ] is False
    assert payload["operator_validation_command_templates"]["labelers_must_not_run_commands"] is True
    assert "record_browser_smoke_evidence" in payload[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert "apply_operator_evidence_templates" in payload[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert index["operator_validation_command_templates"] == payload[
        "operator_validation_command_templates"
    ]
    assert payload["counts"]["users"] == 2
    assert payload["counts"]["ready_to_send"] == 0
    assert payload["counts"]["not_ready_to_send"] == 2
    assert payload["counts"]["waiting_datasets"] == 2
    assert payload["counts"]["dataset_open_tasks"] == 2
    assert payload["counts"]["recordings_without_open_tasks"] == 1
    assert payload["counts"]["recordings_without_open_tasks_by_reason"] == {"tasks_not_generated": 1}
    assert "Generate or import browser-labeling tasks" in payload["counts"][
        "recordings_without_open_tasks_actions"
    ][0]
    assert payload["counts"]["redacted_summary_fields"] == 2
    assert payload["progress_summary"]["waiting_recording_count"] == 2
    assert payload["progress_summary"]["blocked_recording_count"] == 1
    assert payload["progress_summary"]["blocked_recordings_by_reason"] == {"tasks_not_generated": 1}
    assert payload["counts"]["sendability_reasons"] == {"operator_validation_pending": 2}
    assert {tuple(warning["reasons"]) for warning in payload["sendability_warnings"]} == {
        ("operator_validation_pending",)
    }
    assert payload["generated_at_utc"]
    assert payload["files"]["html_index"] == str(output_dir / "index.html")
    assert payload["files"]["readme"] == str(output_dir / "handoff-readme.txt")
    assert payload["files"]["labeler_roster"] == str(output_dir / "labeler-roster.csv")
    assert payload["files"]["validation_log"] == str(output_dir / "validation-log-template.md")
    assert payload["files"]["validation_checklist"] == str(output_dir / "validation-checklist.json")
    assert payload["files"]["bundle_zip"] == str(zip_path)
    assert index["files"]["validation_log"] == payload["files"]["validation_log"]
    assert index["files"]["validation_checklist"] == payload["files"]["validation_checklist"]
    assert index["handoffs"] == payload["handoffs"]
    assert index["ok"] is False
    assert index["store_checks_ok"] is True
    assert index["labeler_landing_page_path"] == "/"
    assert index["labeler_landing_url"] == "https://labeling.example.org"
    assert index["dashboard_path"] == "/work"
    assert index["dashboard_url"] == "https://labeling.example.org/work"
    assert index["dataset_queue_page_path"] == "/datasets"
    assert index["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert index["personal_work_page_path"] == "/my-work"
    assert index["personal_work_url"] == "https://labeling.example.org/my-work"
    assert index["personal_dataset_queue_page_path"] == "/my-datasets"
    assert index["personal_dataset_queue_url"] == "https://labeling.example.org/my-datasets"
    assert payload["personal_work_page_path"] == "/my-work"
    assert payload["personal_work_url"] == "https://labeling.example.org/my-work"
    assert payload["personal_dataset_queue_page_path"] == "/my-datasets"
    assert payload["personal_dataset_queue_url"] == "https://labeling.example.org/my-datasets"
    assert index["labeler_safety"]["dashboard_identity_check_required"] is True
    assert index["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"] is True
    assert index["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert index["browser_mutation_write_policy"]["handoff_artifacts_are_metadata_only"] is True
    assert index["browser_mutation_write_policy"]["browser_label_write_target"] == "training_zarr"
    assert index["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert index["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert index["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert index["browser_mutation_write_checklist"]["ready"] is True
    assert index["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert index["browser_mutation_write_checklist"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert index["browser_mutation_write_checklist"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert index["counts"]["sendability_reasons"] == {"operator_validation_pending": 2}
    assert "Complete required operator validation evidence" in index["sendability_actions"][0]
    assert index["progress_summary"] == payload["progress_summary"]
    assert index["dataset_queue_summary"]["waiting_dataset_count"] == 2
    assert index["counts"]["dataset_queue_states"] == {"has_open_dataset_work": 2}
    assert index["counts"]["dataset_queue_blocked_start_users"] == []
    assert index["counts"]["waiting_datasets"] == 2
    assert index["counts"]["dataset_open_tasks"] == 2
    assert {handoff["user"] for handoff in index["handoffs"]} == {"alice", "bob"}
    assert {handoff["ready_to_send"] for handoff in index["handoffs"]} == {False}
    assert {handoff["operator_validation_required_before_invite"] for handoff in index["handoffs"]} == {True}
    assert {handoff["operator_validation_all_complete"] for handoff in index["handoffs"]} == {False}
    assert {handoff["operator_validation_status"] for handoff in index["handoffs"]} == {
        "pending_operator_evidence"
    }
    assert {
        handoff["operator_validation_command_template_schema"]
        for handoff in index["handoffs"]
    } == {"palette.web_labeling_operator_validation_command_templates.v1"}
    assert {
        handoff["operator_validation_command_template_command_count"]
        for handoff in index["handoffs"]
    } == {7}
    assert all(
        "record_browser_smoke_evidence"
        in handoff["operator_validation_command_template_command_ids"]
        for handoff in index["handoffs"]
    )
    assert all(
        "browser_response_security_headers" in handoff["operator_validation_pending_gate_ids"]
        for handoff in index["handoffs"]
    )
    assert {tuple(handoff["operator_validation_needs_review_gate_ids"]) for handoff in index["handoffs"]} == {()}
    assert {handoff["known_labeler"] for handoff in index["handoffs"]} == {True}
    assert {handoff["known_user_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["assignment_ownership_ok"] for handoff in index["handoffs"]} == {True}
    assert {handoff["assignment_duplicate_active_owner_count"] for handoff in index["handoffs"]} == {0}
    assert {handoff["assignment_ownership_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["single_owner_policy_assignment_scope"] for handoff in index["handoffs"]} == {
        "recording"
    }
    assert {
        handoff["single_owner_policy_recording_assignment_key"]
        for handoff in index["handoffs"]
    } == {"recording_id"}
    assert {
        handoff["single_owner_policy_multiple_labelers_per_recording_allowed"]
        for handoff in index["handoffs"]
    } == {False}
    assert {
        handoff["single_owner_policy_browser_mutation_requires_current_assignment_owner"]
        for handoff in index["handoffs"]
    } == {True}
    assert {handoff["guarded_links_ready"] for handoff in index["handoffs"]} == {True}
    assert {tuple(handoff["missing_guarded_links"]) for handoff in index["handoffs"]} == {()}
    assert {handoff["handoff_artifacts_ready"] for handoff in index["handoffs"]} == {True}
    assert {tuple(handoff["missing_handoff_artifacts"]) for handoff in index["handoffs"]} == {()}
    assert {handoff["handoff_entry_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["preferred_labeler_entrypoint"] for handoff in index["handoffs"]} == {
        "personal_datasets_waiting_queue"
    }
    assert {handoff["personal_dataset_queue_link_role"] for handoff in index["handoffs"]} == {
        "preferred_queue"
    }
    assert {handoff["dataset_queue_link_role"] for handoff in index["handoffs"]} == {
        "canonical_queue_fallback"
    }
    assert {handoff["canonical_dataset_queue_link_role"] for handoff in index["handoffs"]} == {
        "canonical_queue_fallback"
    }
    assert {handoff["dashboard_link_role"] for handoff in index["handoffs"]} == {"fallback_dashboard"}
    assert {handoff["identity_probe_link_role"] for handoff in index["handoffs"]} == {"identity_check"}
    assert {handoff["task_links_role"] for handoff in index["handoffs"]} == {"convenience_entry_hints"}
    assert {handoff["preferred_labeler_entry_url_matches_dataset_queue"] for handoff in index["handoffs"]} == {
        True
    }
    assert {
        handoff["preferred_labeler_entry_url_matches_personal_dataset_queue"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        handoff["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        for handoff in index["handoffs"]
    } == {True}
    assert {handoff["labeler_safety_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["labeler_safety_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["labeler_safety_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["labeler_safety_browser_receives_raw_zarr_paths"] for handoff in index["handoffs"]} == {False}
    assert {handoff["labeler_route_authorization_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["labeler_route_authorization_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["labeler_route_authorization_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["signed_link_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["signed_link_policy_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["signed_link_policy_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["signed_link_authorization_grant"] for handoff in index["handoffs"]} == {False}
    assert {handoff["signed_link_forwarded_links_recheck_identity"] for handoff in index["handoffs"]} == {True}
    assert {
        handoff["signed_link_runtime_operator_validation_start_gate_enforced"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        handoff["signed_link_operator_validation_start_gate_checked_before_session_create"]
        for handoff in index["handoffs"]
    } == {True}
    assert {handoff["session_guard_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["session_guard_policy_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["session_guard_policy_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["session_guard_stale_tab_save_rejected"] for handoff in index["handoffs"]} == {True}
    assert {handoff["session_guard_non_startable_task_sessions_rejected"] for handoff in index["handoffs"]} == {True}
    assert {handoff["task_state_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["task_state_policy_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["task_state_policy_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["task_state_startable_task_states"] for handoff in index["handoffs"]} == {
        '["pending", "in_progress"]'
    }
    assert {handoff["task_state_completed_tasks_read_only"] for handoff in index["handoffs"]} == {True}
    assert {handoff["task_state_non_startable_task_open_requests"] for handoff in index["handoffs"]} == {
        "reject_task_not_startable"
    }
    assert {handoff["task_state_non_startable_task_save_requests"] for handoff in index["handoffs"]} == {
        "reject_task_not_startable"
    }
    assert {handoff["task_state_requires_current_target_token"] for handoff in index["handoffs"]} == {True}
    assert {handoff["zarr_backup_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["zarr_backup_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["zarr_backup_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["zarr_backup_copy_before_labeling"] for handoff in index["handoffs"]} == {True}
    assert {handoff["zarr_backup_labelers_do_not_receive_backup_paths"] for handoff in index["handoffs"]} == {True}
    assert {handoff["mutation_audit_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["mutation_audit_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["mutation_audit_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["mutation_audit_event_store"] for handoff in index["handoffs"]} == {
        "labeling_task_events"
    }
    assert {handoff["mutation_audit_server_records_events"] for handoff in index["handoffs"]} == {True}
    assert {handoff["mutation_audit_browser_records_events_directly"] for handoff in index["handoffs"]} == {False}
    assert {handoff["browser_response_security_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_response_security_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_response_security_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["browser_response_security_no_store_cache"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_response_security_clickjacking_protection"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_mutation_write_policy_present"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_mutation_write_ready"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_mutation_write_readiness"] for handoff in index["handoffs"]} == {"passed"}
    assert {handoff["browser_mutation_target_contract_met"] for handoff in index["handoffs"]} == {True}
    assert {handoff["browser_mutation_target_mismatch_count"] for handoff in index["handoffs"]} == {0}
    assert {handoff["direct_browser_start_contract_met"] for handoff in index["handoffs"]} == {True}
    assert {handoff["direct_browser_start_mismatch_count"] for handoff in index["handoffs"]} == {0}
    assert {handoff["single_owner_policy_contract_met"] for handoff in index["handoffs"]} == {True}
    assert {handoff["dashboard_url"] for handoff in index["handoffs"]} == {"https://labeling.example.org/work"}
    assert {handoff["labeler_landing_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org"
    }
    assert {handoff["expected_user_labeler_landing_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org?expected_user=alice",
        "https://labeling.example.org?expected_user=bob",
    }
    assert {handoff["expected_user_dashboard_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org/work?expected_user=alice",
        "https://labeling.example.org/work?expected_user=bob",
    }
    assert {handoff["expected_user_dataset_queue_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org/datasets?expected_user=alice",
        "https://labeling.example.org/datasets?expected_user=bob",
    }
    assert {handoff["expected_user_personal_dataset_queue_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {handoff["expected_user_personal_work_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org/my-work?expected_user=alice",
        "https://labeling.example.org/my-work?expected_user=bob",
    }
    assert {handoff["personalized_labeler_entry_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {handoff["queue_first_entry_contract"]["ready"] for handoff in index["handoffs"]} == {
        True
    }
    assert {
        handoff["queue_first_entry_contract"][
            "personalized_labeler_entry_url_matches_personal_dataset_queue"
        ]
        for handoff in index["handoffs"]
    } == {True}
    assert {handoff["expected_user_identity_probe_url"] for handoff in index["handoffs"]} == {
        "https://labeling.example.org/identity?expected_user=alice",
        "https://labeling.example.org/identity?expected_user=bob",
    }
    assert {handoff["direct_browser_start_contract_summary"]["ready"] for handoff in index["handoffs"]} == {
        True
    }
    assert {handoff["direct_browser_start_contract_summary_schema"] for handoff in index["handoffs"]} == {
        "palette.web_labeling_direct_browser_start_contract_summary.v1"
    }
    assert {handoff["direct_browser_start_contract_summary_browser_label_write_target"] for handoff in index["handoffs"]} == {
        "training_zarr"
    }
    assert {handoff["direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority"] for handoff in index["handoffs"]} == {
        False
    }
    assert {
        handoff["runtime_operator_validation_gate_cli_policy"]["preferred_require_flag"]
        for handoff in index["handoffs"]
    } == {"--require-operator-validation-for-browser-work"}
    assert {
        handoff["runtime_operator_validation_gate_cli_policy_preferred_require_flag"]
        for handoff in index["handoffs"]
    } == {"--require-operator-validation-for-browser-work"}
    assert {
        handoff["runtime_operator_validation_gate_cli_policy_protects_browser_start_open"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        handoff["runtime_operator_validation_gate_cli_policy_protects_browser_mutations"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        handoff["runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        handoff["runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        handoff["runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation"]
        for handoff in index["handoffs"]
    } == {True}
    assert {
        "Complete required operator validation evidence" in handoff["sendability_actions"][0]
        for handoff in index["handoffs"]
    } == {True}
    for handoff in index["handoffs"]:
        for gate_id in (
            "mutable_zarr_backup_confirmation",
            "browser_response_security_headers",
            "identity_probe_verification",
            "browser_smoke",
            "disposable_zarr_mutation_smoke",
            "operator_recovery_contract",
        ):
            assert handoff[f"operator_validation_gate_{gate_id}_status"] == (
                "missing_evidence"
            )
            assert handoff[f"operator_validation_gate_{gate_id}_pending"] is True
            assert handoff[f"operator_validation_gate_{gate_id}_missing_evidence"] is True
            assert handoff[f"operator_validation_gate_{gate_id}_needs_review"] is False
            assert handoff[f"operator_validation_gate_{gate_id}_passed"] is False
    alice_handoff = next(handoff for handoff in index["handoffs"] if handoff["user"] == "alice")
    assert alice_handoff["progress_summary"]["waiting_recording_count"] == 1
    assert alice_handoff["progress_summary"]["blocked_recording_count"] == 1
    assert alice_handoff["dataset_queue_summary"]["waiting_dataset_count"] == 1
    assert alice_handoff["dataset_queue_state_code"] == "has_open_dataset_work"
    assert alice_handoff["labeler_work_completion_status"] == "waiting"
    assert alice_handoff["labeler_work_completion_has_waiting_work"] is True
    assert alice_handoff["labeler_work_completion_completed"] is False
    assert alice_handoff["dataset_queue_blocks_labeler_start"] is False
    assert alice_handoff["dataset_queue_start_ready"] is True
    assert alice_handoff["dataset_queue_start_status"] == "passed"
    assert alice_handoff["guarded_links_ready"] is True
    assert alice_handoff["handoff_artifacts_ready"] is True
    assert alice_handoff["handoff_entry_readiness"] == "passed"
    assert alice_handoff["dataset_queue_preview_url"] == "https://labeling.example.org/my-datasets?expected_user=alice"
    assert (
        alice_handoff["canonical_dataset_queue_preview_url"]
        == "https://labeling.example.org/datasets?expected_user=alice"
    )
    assert alice_handoff["files"]["dataset_queue"] == str(output_dir / "alice" / "dataset-queue.json")
    assert alice_handoff["operator_validation_visibility_policy"] == alice_manifest[
        "operator_validation_visibility_policy"
    ]
    assert alice_dataset_queue["operator_validation_visibility_policy"] == alice_manifest[
        "operator_validation_visibility_policy"
    ]
    assert {handoff["labeler_safety"]["dashboard_identity_check_required"] for handoff in index["handoffs"]} == {True}
    assert {
        handoff["labeler_safety"]["labeler_api_redaction"]["redacts_user_summary_path_like_string_values"]
        for handoff in index["handoffs"]
    } == {True}
    assert index["counts"]["recordings_without_open_tasks"] == 1
    assert index["counts"]["recordings_without_open_tasks_by_reason"] == {"tasks_not_generated": 1}
    assert "Generate or import browser-labeling tasks" in index["counts"][
        "recordings_without_open_tasks_actions"
    ][0]
    assert index["safe_share_gate"]["schema"] == "palette.web_labeling_safe_share_gate.v1"
    assert index["safe_share_gate"]["gate"] == "labeler_links_safe_to_share"
    assert index["safe_share_gate"]["ready_to_send_is_sufficient"] is False
    assert index["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert index["safe_share_requires_require_shareable_inspection"] is True
    assert index["safe_share_ready_to_send_is_sufficient"] is False
    assert index["safe_share_required_inspection_field"] == "labeler_links_safe_to_share"
    assert index["safe_share_required_inspection_value"] is True
    assert index["safe_share_launch_blocking_evidence_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ]
    assert index["counts"]["redacted_summary_fields"] == 2
    assert all(handoff["links_expire_at_utc"] for handoff in index["handoffs"])
    assert "Palette labeling handoffs" in html_index
    assert "ready" in html_index
    assert "Landing URL: https://labeling.example.org" in html_index
    assert "https://labeling.example.org?expected_user=alice" in html_index
    assert "https://labeling.example.org/work" in html_index
    assert "https://labeling.example.org/my-datasets?expected_user=alice" in html_index
    assert "https://labeling.example.org/my-datasets?expected_user=bob" in html_index
    assert "https://labeling.example.org/datasets?expected_user=alice" in html_index
    assert "https://labeling.example.org/datasets?expected_user=bob" in html_index
    assert "Dashboard URL: https://labeling.example.org/work" in html_index
    assert "Personalized dataset queue URL: https://labeling.example.org/my-datasets" in html_index
    assert "Personalized work URL: https://labeling.example.org/my-work" in html_index
    assert "Landing URL: https://labeling.example.org" in readme
    assert "Dataset queue URL: https://labeling.example.org/datasets" in html_index
    assert "Dashboard URL: https://labeling.example.org/work" in readme
    assert "Personalized dataset queue URL: https://labeling.example.org/my-datasets" in readme
    assert "Personalized work URL: https://labeling.example.org/my-work" in readme
    assert "Dataset queue URL: https://labeling.example.org/datasets" in readme
    assert "Generate or import browser-labeling tasks" in html_index
    assert "Generate or import browser-labeling tasks" in readme
    assert "No open reasons" in html_index
    assert "tasks_not_generated" in html_index
    assert "Store checks ok: True" in html_index
    assert "Safe-share gate:" in html_index
    assert "Do not share labeler links solely because per-user handoffs say ready_to_send" in html_index
    assert "labeler_links_safe_to_share=true" in html_index
    assert "Unapproved mutable-Zarr backup, browser response-security, identity-source, browser-smoke, disposable-Zarr mutation, or operator-recovery evidence gates are launch blockers" in html_index
    assert "Safe-share next actions: 6" in html_index
    assert "browser_smoke=missing_evidence" in html_index
    assert "Waiting recordings: 2" in html_index
    assert "Waiting datasets: 2" in html_index
    assert "Queue state" in html_index
    assert "Guarded links" in html_index
    assert "Entry readiness" in html_index
    assert "has_open_dataset_work" in html_index
    assert "Blocked/no-open recordings: 1" in html_index
    assert "Validation log" in html_index
    assert "validation-log-template.md" in html_index
    assert "Validation checklist" in html_index
    assert "validation-checklist.json" in html_index
    assert "alice/index.html" in html_index
    assert "alice/message.txt" in html_index
    assert "alice/labeler-quickstart.txt" in html_index
    assert "bob/index.html" in html_index
    assert "alice/manifest.json" in html_index
    assert "bob/manifest.json" in html_index
    assert "carol" not in html_index
    assert "Palette labeling handoff bundle" in readme
    assert "Validation log:" in readme
    assert "validation-log-template.md" in readme
    assert "validation-checklist.json" in readme
    assert "Generated at UTC:" in readme
    assert "Ready to send: 0" in readme
    assert "Not ready to send: 2" in readme
    assert "Waiting recordings: 2" in readme
    assert "Waiting datasets: 2" in readme
    assert "Blocked/no-open recordings: 1" in readme
    assert 'Blocked/no-open recordings by reason: {"tasks_not_generated": 1}' in readme
    assert "Store checks ok: True" in readme
    assert "Assigned recordings without startable tasks: 1" in readme
    assert 'Assigned recordings without startable tasks by reason: {"tasks_not_generated": 1}' in readme
    assert "Redacted user-summary fields: 2" in readme
    assert "Signed links are convenience entry points" in readme
    assert "Implementation status: browser labeling contracts are generated into this bundle" in readme
    assert "server-owned task/training Zarr writes" in readme
    assert "Launch evidence still requires approved mutable-Zarr backup" in readme
    assert "docs/web_labeling_implementation_status.md" in readme
    assert "Forwarded queue, dashboard, and task links are rechecked" in readme
    assert "Do not send links solely because per-user handoffs say ready_to_send" in readme
    assert "labeler_links_safe_to_share=true" in readme
    assert "Unapproved mutable-Zarr backup, browser response-security, identity-source, browser-smoke, disposable-Zarr mutation, or operator-recovery evidence gates are launch blockers" in readme
    assert "Safe-share next actions: 6" in readme
    assert "operator_recovery_contract=missing_evidence" in readme
    assert "guarded landing page or personalized dataset queue as the queue-first entry point" in readme
    assert "Handoff CSV, HTML, and JSON files are metadata only" in readme
    assert "guarded dataset queue page links and dashboard filter links" in readme
    assert "# Web Labeling Validation Log" in validation_log
    assert "multi-user handoff bundle" in validation_log
    assert "Queue-First Entry" in validation_log
    assert "Browser Payload Redaction" in validation_log
    assert "Assignment Ownership Contract" in validation_log
    assert "Labeler Route Authorization" in validation_log
    assert "Signed Link Contract" in validation_log
    assert "Operator Boundary Static Contract" in validation_log
    assert "Operator Recovery Static Contract" in validation_log
    assert "Operator Authorization Boundary" in validation_log
    assert "Browser Response Security Headers" in validation_log
    assert "Browser Response Security Static Contract" in validation_log
    assert "Mutation Audit and Provenance" in validation_log
    assert "Zarr Backup and Rollback Contract" in validation_log
    assert "Identity Probe Links" in validation_log
    assert "Dataset Queue Start Readiness" in validation_log
    assert "dataset_queue_blocked_start_users" in validation_log
    assert "Assignment Transition Evidence" in validation_log
    assert f"- Manifest: {output_dir / 'index.json'}" in validation_log
    assert f"- Handoff roster: {output_dir / 'labeler-roster.csv'}" in validation_log
    handoff_gate_statuses = {gate["id"]: gate["status"] for gate in validation_checklist["gates"]}
    assert validation_checklist["schema"] == "palette.web_labeling_validation_checklist.v1"
    assert validation_checklist["bundle_label"] == "multi-user handoff bundle"
    assert validation_checklist["operator_validation_visibility_policy"] == payload[
        "operator_validation_visibility_policy"
    ]
    assert validation_checklist["operator_validation_command_templates"]["schema"] == (
        "palette.web_labeling_operator_validation_command_templates.v1"
    )
    assert validation_checklist["safe_share_gate"]["schema"] == (
        "palette.web_labeling_safe_share_gate.v1"
    )
    assert validation_checklist["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert validation_checklist["safe_share_ready_to_send_is_sufficient"] is False
    assert validation_checklist["safe_share_required_inspection_field"] == (
        "labeler_links_safe_to_share"
    )
    assert "operator_recovery_contract" in validation_checklist[
        "safe_share_launch_blocking_evidence_gate_ids"
    ]
    assert "operator_validation_record_command_ids" in validation_checklist[
        "safe_share_launch_blocking_next_action_detail_fields"
    ]
    assert "operator_validation_evidence_template_path" in validation_checklist[
        "safe_share_launch_blocking_next_action_command_fields"
    ]
    assert "record_browser_smoke_evidence" in validation_checklist[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert "apply_operator_evidence_templates" in validation_checklist[
        "operator_validation_command_templates"
    ]["command_ids"]
    assert "multi_user_dry_run" in validation_checklist[
        "operator_validation_command_templates"
    ]["missing_command_gate_ids"]
    assert validation_checklist["validation_log"] == str(output_dir / "validation-log-template.md")
    assert validation_checklist["labeler_landing_page_path"] == "/"
    assert validation_checklist["labeler_landing_url"] == "https://labeling.example.org"
    assert validation_checklist["dataset_queue_page_path"] == "/datasets"
    assert validation_checklist["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert validation_checklist["personal_dataset_queue_page_path"] == "/my-datasets"
    assert validation_checklist["personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets"
    )
    assert validation_checklist["expected_user_personal_dataset_queue_url"] == ""
    assert validation_checklist["personal_work_page_path"] == "/my-work"
    assert validation_checklist["personal_work_url"] == "https://labeling.example.org/my-work"
    assert validation_checklist["expected_user_personal_work_url"] == ""
    assert validation_checklist["personalized_labeler_entrypoint"] == (
        "personal_datasets_waiting_queue"
    )
    assert validation_checklist["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets"
    )
    assert validation_checklist["queue_first_entry_contract"]["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets"
    )
    assert validation_checklist["queue_first_entry_contract"][
        "personalized_labeler_entry_url_matches_personal_dataset_queue"
    ] is True
    assert validation_checklist["dataset_queue_states"] == {"has_open_dataset_work": 2}
    assert validation_checklist["dataset_queue_blocked_start_users"] == []
    assert validation_checklist["queue_first_entry_contract"]["ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["dataset_queue_ready"] is True
    assert validation_checklist["queue_first_entry_contract"]["labeler_landing_page_kind"] == (
        "datasets_waiting_queue"
    )
    assert validation_checklist["queue_first_entry_contract"]["datasets_waiting_aliases_ready"] is True
    assert validation_checklist["identity_probe_link_contract"]["ready"] is True
    assert validation_checklist["identity_probe_link_contract"]["identity_check_required"] is True
    assert validation_checklist["identity_probe_link_contract"]["expected_user_identity_probe_url_present"] is False
    assert validation_checklist["identity_probe_link_contract"]["batch_identity_probe_evidence_present"] is True
    assert validation_checklist["counts"]["identity_probe_link_contract_ready"] == 1
    assert validation_checklist["browser_response_security_policy"]["headers"]["Cache-Control"] == (
        "no-store, no-cache, must-revalidate, max-age=0"
    )
    assert validation_checklist["operator_authorization_contract"]["ready"] is True
    assert validation_checklist["operator_authorization_contract"]["operator_boundary_required_for_launch"] is True
    assert validation_checklist["operator_recovery_policy"]["task_state_route"] == (
        "/api/admin/tasks/{task_id}/state"
    )
    assert validation_checklist["operator_recovery_policy"]["task_repair_route"] == (
        "/api/admin/tasks/{task_id}/repair"
    )
    assert validation_checklist["operator_recovery_policy"]["reassignment_session_repair_route"] == (
        "/api/admin/recordings/{recording_id}/repair-reassignment-sessions"
    )
    assert validation_checklist["operator_recovery_policy"]["audit_event_lookup_route"] == (
        "/api/admin/events/{event_id}"
    )
    assert validation_checklist["operator_recovery_policy"]["failed_promotion_retry_route"] == (
        "/api/admin/events/{event_id}/retry-promotion"
    )
    assert validation_checklist["operator_recovery_contract"]["ready"] is True
    assert validation_checklist["operator_recovery_contract"]["reassignment_closes_previous_owner_sessions"] is True
    assert validation_checklist["operator_recovery_contract"][
        "reassignment_closes_previous_owner_sessions_before_assignment_update"
    ] is True
    assert validation_checklist["operator_recovery_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert validation_checklist["operator_recovery_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert validation_checklist["operator_recovery_contract"]["task_reopen_operator_only"] is True
    assert validation_checklist["operator_recovery_contract"]["task_repair_route"] == (
        "/api/admin/tasks/{task_id}/repair"
    )
    assert validation_checklist["operator_recovery_contract"]["reassignment_session_repair_route"] == (
        "/api/admin/recordings/{recording_id}/repair-reassignment-sessions"
    )
    assert validation_checklist["operator_recovery_contract"]["audit_event_lookup_route"] == (
        "/api/admin/events/{event_id}"
    )
    assert validation_checklist["operator_recovery_contract"]["operator_repair_records_audit_event"] is True
    assert validation_checklist["operator_recovery_contract"]["failed_promotion_retry_operator_only"] is True
    assert validation_checklist["operator_recovery_contract"][
        "failed_promotion_retry_requires_failed_event"
    ] is True
    assert validation_checklist["operator_recovery_contract"][
        "failed_promotion_retry_claims_after_event_type_check"
    ] is True
    assert validation_checklist["operator_recovery_contract"][
        "failed_promotion_retry_claim_event_type"
    ] == "promotion_retry_started"
    assert validation_checklist["operator_recovery_contract"]["rollback_requires_backup_plan"] is True
    assert validation_checklist["operator_recovery_contract"]["bad_disposable_mutation_recovery_ready"] is True
    assert validation_checklist["operator_recovery_contract"][
        "disposable_mutation_smoke_requires_recovery_path_verification"
    ] is True
    assert validation_checklist["counts"]["operator_recovery_contract_ready"] == 1
    assert validation_checklist["browser_response_security_contract"]["ready"] is True
    assert validation_checklist["browser_response_security_contract"]["mime_sniffing_protection"] is True
    assert validation_checklist["browser_payload_redaction_contract"]["ready"] is True
    assert validation_checklist["browser_payload_redaction_contract"]["redacts_mutation_response_paths"] is True
    assert validation_checklist["browser_payload_redaction_contract"]["redacts_error_detail_paths"] is True
    assert validation_checklist["labeler_route_authorization_policy"]["known_assignment_store_user_required"] is True
    assert validation_checklist["labeler_route_authorization_policy"][
        "single_owner_store_proof_required_for_browser_work"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"]["ready"] is True
    assert validation_checklist["labeler_route_authorization_contract"]["copied_links_rechecked_server_side"] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_required_for_browser_work"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "single_owner_store_proof_requires_zero_duplicate_active_owners"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "personal_work_page_expected_user_guarded"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"][
        "personal_dataset_queue_page_expected_user_guarded"
    ] is True
    assert validation_checklist["labeler_route_authorization_contract"]["task_open_requires_startable_task_state"] is True
    assert validation_checklist["labeler_route_authorization_contract"]["startable_task_states"] == [
        "pending",
        "in_progress",
    ]
    assert validation_checklist["signed_link_policy"]["authorization_grant"] is False
    assert validation_checklist["signed_link_contract"]["ready"] is True
    assert validation_checklist["signed_link_contract"]["forwarded_signed_links_recheck_identity"] is True
    assert validation_checklist["signed_link_contract"]["runtime_operator_validation_start_gate_enforced"] is True
    assert validation_checklist["signed_link_contract"]["operator_validation_start_gate_checked_before_session_create"] is True
    assert validation_checklist["expected_user_guard_contract"]["ready"] is True
    assert validation_checklist["expected_user_guard_contract"]["configured_guards"]["dataset_queue_page"] == (
        "dashboard_user_mismatch"
    )
    assert validation_checklist["expected_user_guard_contract"]["configured_guards"]["personal_work_page"] == (
        "dashboard_user_mismatch"
    )
    assert validation_checklist["expected_user_guard_contract"]["configured_guards"][
        "personal_dataset_queue_page"
    ] == "dashboard_user_mismatch"
    assert validation_checklist["expected_user_guard_contract"]["promotion_retry_guarded_support_only"] is True
    assert validation_checklist["expected_user_guard_contract"]["promotion_retry_labeler_mutation_enabled"] is False
    assert validation_checklist["expected_user_guard_contract"]["promotion_retry_labeler_rejection_error"] == (
        "operator_support_required"
    )
    assert validation_checklist["session_guard_contract"]["ready"] is True
    assert validation_checklist["session_guard_contract"]["rejects_after_reassignment"] is True
    assert validation_checklist["session_guard_contract"]["rejects_after_completion_or_reopen"] is True
    assert validation_checklist["session_guard_contract"]["non_startable_task_sessions_rejected"] is True
    assert validation_checklist["task_state_policy"]["browser_mutation_target_token"] == "required_current_target_token"
    assert validation_checklist["task_state_contract"]["ready"] is True
    assert validation_checklist["task_state_contract"]["startable_task_states"] == ["pending", "in_progress"]
    assert validation_checklist["task_state_contract"]["non_startable_task_open_requests"] == "reject_task_not_startable"
    assert validation_checklist["task_state_contract"]["non_startable_task_save_requests"] == "reject_task_not_startable"
    assert validation_checklist["task_state_contract"]["labeler_promotion_retry_mutation_enabled"] is False
    assert validation_checklist["task_state_contract"]["labeler_promotion_retry_rejection_error"] == (
        "operator_support_required"
    )
    assert validation_checklist["task_state_contract"]["ordinary_labeler_promotion_retry_mutation"] == (
        "operator_support_required"
    )
    assert validation_checklist["task_state_contract"]["operator_reopen_required_before_more_labeling"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["ready"] is True
    assert validation_checklist["browser_workflow_scope_contract"]["target_indices_components_labels_frames_server_owned"] is True
    assert validation_checklist["browser_mutation_target_contract"]["ready"] is True
    assert validation_checklist["browser_mutation_write_policy"]["authoritative_label_state"] == "assigned_task_zarr_scope"
    assert validation_checklist["browser_mutation_write_policy"]["mutable_label_data_plane"] == (
        "task_scoped_training_zarr"
    )
    assert validation_checklist["browser_mutation_write_policy"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert validation_checklist["browser_mutation_write_policy"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert validation_checklist["browser_mutation_write_policy"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert validation_checklist["browser_mutation_write_policy"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert validation_checklist["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert validation_checklist["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert validation_checklist["browser_mutation_write_contract"]["ready"] is True
    assert validation_checklist["browser_mutation_write_contract"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert validation_checklist["browser_mutation_write_contract"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert validation_checklist["browser_mutation_write_contract"]["training_zarr_mutations_are_server_owned"] is True
    assert validation_checklist["browser_mutation_write_contract"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert validation_checklist["browser_mutation_write_contract"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert validation_checklist["browser_mutation_write_contract"]["browser_writes_handoff_csv"] is False
    assert validation_checklist["browser_mutation_write_contract"]["browser_writes_intermediate_csv"] is False
    assert validation_checklist["browser_mutation_write_contract"]["browser_writes_csv_or_handoff_files"] is False
    assert validation_checklist["mutation_audit_contract"]["ready"] is True
    assert validation_checklist["mutation_audit_contract"]["server_records_events"] is True
    assert validation_checklist["mutation_audit_contract"]["browser_records_events_directly"] is False
    assert validation_checklist["zarr_backup_contract"]["ready"] is True
    assert validation_checklist["zarr_backup_contract"]["operator_only"] is True
    assert validation_checklist["zarr_backup_contract"]["labelers_do_not_receive_backup_paths"] is True
    assert validation_checklist["assignment_ownership_contract"]["ready"] is True
    assert validation_checklist["assignment_ownership_contract"]["assignment_scope"] == "recording"
    assert validation_checklist["assignment_ownership_contract"]["recording_assignment_key"] == (
        "recording_id"
    )
    assert validation_checklist["assignment_ownership_contract"][
        "multiple_labelers_per_recording_allowed"
    ] is False
    assert validation_checklist["assignment_ownership_contract"]["one_active_owner"] is True
    assert validation_checklist["assignment_ownership_contract"][
        "stale_sessions_closed_before_assignment_update"
    ] is True
    assert validation_checklist["assignment_ownership_contract"][
        "reassignment_target_validated_before_session_closure"
    ] is True
    assert validation_checklist["assignment_ownership_contract"][
        "session_closure_and_assignment_update_atomic"
    ] is True
    assert validation_checklist["assignment_ownership_contract"]["duplicate_active_owner_count"] == 0
    assert validation_checklist["assignment_ownership_contract"][
        "browser_mutation_requires_current_assignment_owner"
    ] is True
    assert validation_checklist["ready_for_operator_validation"] is True
    assert validation_checklist["all_validation_complete"] is False
    assert handoff_gate_statuses["static_readiness"] == "passed"
    assert handoff_gate_statuses["queue_first_entry_contract"] == "passed"
    assert handoff_gate_statuses["identity_probe_link_contract"] == "passed"
    assert handoff_gate_statuses["assignment_ownership_contract"] == "passed"
    assert handoff_gate_statuses["browser_payload_redaction_contract"] == "passed"
    assert handoff_gate_statuses["labeler_route_authorization"] == "passed"
    assert handoff_gate_statuses["signed_link_contract"] == "passed"
    assert handoff_gate_statuses["expected_user_guard_contract"] == "passed"
    assert handoff_gate_statuses["session_guard_contract"] == "passed"
    assert handoff_gate_statuses["task_state_contract"] == "passed"
    assert handoff_gate_statuses["operator_authorization_contract"] == "passed"
    assert handoff_gate_statuses["operator_recovery_contract"] == "passed"
    assert handoff_gate_statuses["browser_response_security_contract"] == "passed"
    assert handoff_gate_statuses["browser_workflow_scope_contract"] == "passed"
    assert handoff_gate_statuses["dataset_queue_start_readiness"] == "passed"
    assert handoff_gate_statuses["browser_mutation_target_contract"] == "passed"
    assert handoff_gate_statuses["browser_mutation_write_policy"] == "passed"
    assert handoff_gate_statuses["mutation_audit_contract"] == "passed"
    assert handoff_gate_statuses["zarr_backup_contract"] == "passed"
    assert handoff_gate_statuses["identity_probe_verification"] == "pending_operator_evidence"
    assert handoff_gate_statuses["operator_authorization_boundary"] == "pending_operator_evidence"
    assert handoff_gate_statuses["browser_response_security_headers"] == "pending_operator_evidence"
    assert handoff_gate_statuses["dashboard_visibility"] == "pending_operator_evidence"
    assert handoff_gate_statuses["multi_user_dry_run"] == "pending_operator_evidence"
    assert handoff_gate_statuses["mutable_zarr_backup_confirmation"] == "pending_operator_evidence"
    browser_smoke_gate = next(gate for gate in validation_checklist["gates"] if gate["id"] == "browser_smoke")
    browser_smoke_evidence_text = "\n".join(browser_smoke_gate["operator_evidence"])
    assert "personalized /my-datasets queue entry" in browser_smoke_evidence_text
    assert "human-readable /labeling alias" in browser_smoke_evidence_text
    assert "personalized /my-work dashboard fallback" in browser_smoke_evidence_text
    backup_gate = next(gate for gate in validation_checklist["gates"] if gate["id"] == "mutable_zarr_backup_confirmation")
    assert backup_gate["required"] is True
    assert "does not include a top-level zarr backup plan" in backup_gate["details"]
    assert "Reference the launch bundle backup plan" in backup_gate["operator_evidence"][0]
    assert "Your Palette labeling work is ready." not in alice_message
    assert "Your Palette labeling handoff needs operator review before starting." in alice_message
    assert "Identity check:" in alice_message
    assert "https://labeling.example.org/identity?expected_user=alice" in alice_message
    assert "https://labeling.example.org/datasets?expected_user=alice" in alice_message
    assert "Human-readable labeling home alias:" in alice_message
    assert "https://labeling.example.org/labeling?expected_user=alice" in alice_message
    assert "Dataset queue state: has_open_dataset_work" in alice_message
    assert "Dataset queue start: allowed" in alice_message
    assert "Preview queue-first entry point:" in alice_message
    assert "Queue-first start page:" in alice_message
    assert "https://labeling.example.org/my-datasets?expected_user=alice" in alice_message
    assert "confirm the dashboard shows you as alice" in alice_message
    assert "https://labeling.example.org/work?expected_user=alice" in alice_message
    assert "assigned task/training Zarr scope" in alice_message
    assert "CSV, HTML, JSON, and handoff files are metadata only" in alice_message
    assert "Each recording has one active assigned owner" in alice_message
    assert "No Palette or Crimson installation is needed" in alice_quickstart
    assert "Wait for operator review before starting." in alice_quickstart
    assert "Dataset queue state: has_open_dataset_work" in alice_quickstart
    assert "Dataset queue start: allowed" in alice_quickstart
    assert "Open the identity check and confirm it reports you as alice" in alice_quickstart
    assert "https://labeling.example.org/identity?expected_user=alice" in alice_quickstart
    assert "Human-readable labeling home alias" in alice_quickstart
    assert "https://labeling.example.org/labeling?expected_user=alice" in alice_quickstart
    assert "https://labeling.example.org/my-datasets?expected_user=alice" in alice_quickstart
    assert "https://labeling.example.org/datasets?expected_user=alice" in alice_quickstart
    assert "Canonical dataset queue fallback" in alice_quickstart
    assert "preferred queue-first view" in alice_quickstart
    assert "Confirm the dashboard shows you as alice" in alice_quickstart
    assert "https://labeling.example.org/work?expected_user=alice" in alice_quickstart
    assert "assigned task/training Zarr scope" in alice_quickstart
    assert "not label write targets" in alice_quickstart
    assert "Each recording has one active assigned owner" in alice_quickstart
    assert "Human-readable labeling home alias" in alice_html
    assert "https://labeling.example.org/labeling?expected_user=alice" in alice_html
    assert alice_manifest["expected_user_dashboard_url"] == "https://labeling.example.org/work?expected_user=alice"
    assert bob_manifest["expected_user_dashboard_url"] == "https://labeling.example.org/work?expected_user=bob"
    assert alice_manifest["expected_user_labeler_landing_url"] == "https://labeling.example.org?expected_user=alice"
    assert bob_manifest["expected_user_labeler_landing_url"] == "https://labeling.example.org?expected_user=bob"
    assert alice_manifest["expected_user_labeling_home_url"] == "https://labeling.example.org/labeling?expected_user=alice"
    assert bob_manifest["expected_user_labeling_home_url"] == "https://labeling.example.org/labeling?expected_user=bob"
    assert alice_manifest["expected_user_dataset_queue_url"] == "https://labeling.example.org/datasets?expected_user=alice"
    assert bob_manifest["expected_user_dataset_queue_url"] == "https://labeling.example.org/datasets?expected_user=bob"
    assert alice_manifest["expected_user_identity_probe_url"] == "https://labeling.example.org/identity?expected_user=alice"
    assert bob_manifest["expected_user_identity_probe_url"] == "https://labeling.example.org/identity?expected_user=bob"
    assert alice_manifest["files"]["dataset_queue"] == str(output_dir / "alice" / "dataset-queue.json")
    assert bob_manifest["files"]["dataset_queue"] == str(output_dir / "bob" / "dataset-queue.json")
    for user_manifest, dataset_queue in (
        (alice_manifest, alice_dataset_queue),
        (bob_manifest, bob_dataset_queue),
    ):
        assert dataset_queue["browser_mutation_write_policy"] == user_manifest[
            "browser_mutation_write_policy"
        ]
        assert dataset_queue["dataset_queue_direct_start_policy"] == user_manifest[
            "dataset_queue_direct_start_policy"
        ]
        assert dataset_queue["runtime_operator_validation_gate_cli_policy"] == user_manifest[
            "runtime_operator_validation_gate_cli_policy"
        ]
        assert user_manifest["runtime_operator_validation_gate_cli_policy"][
            "preferred_require_flag"
        ] == "--require-operator-validation-for-browser-work"
        assert user_manifest["runtime_operator_validation_gate_cli_policy"][
            "protects_browser_start_open"
        ] is True
        assert user_manifest["runtime_operator_validation_gate_cli_policy"][
            "protects_browser_mutations"
        ] is True
        assert user_manifest["runtime_operator_validation_gate_cli_policy"][
            "blocks_before_target_token_check"
        ] is True
        assert user_manifest["runtime_operator_validation_gate_cli_policy"][
            "blocks_before_zarr_write"
        ] is True
        assert user_manifest["runtime_operator_validation_gate_cli_policy"][
            "blocks_before_audit_event_creation"
        ] is True
        assert dataset_queue["browser_mutation_write_checklist"]["ready"] is True
        assert dataset_queue["browser_mutation_write_checklist"][
            "label_mutation_target_kind"
        ] == "task_scoped_training_zarr"
        assert dataset_queue["browser_mutation_write_checklist"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert dataset_queue["browser_mutation_write_checklist"][
            "csv_handoff_artifact_role"
        ] == "metadata_only_control_plane"
        assert dataset_queue["browser_mutation_write_checklist"][
            "csv_handoff_artifacts_are_label_write_targets"
        ] is False
        assert dataset_queue["browser_mutation_write_checklist"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert dataset_queue["browser_mutation_write_checklist"][
            "browser_writes_handoff_csv"
        ] is False
        assert dataset_queue["browser_mutation_write_checklist"][
            "browser_writes_intermediate_csv"
        ] is False
        assert dataset_queue["browser_mutation_target_contract_met"] is True
        assert dataset_queue["browser_mutation_target_mismatch_count"] == 0
        assert dataset_queue["direct_browser_start_contract_met"] is True
        assert dataset_queue["direct_browser_start_mismatch_count"] == 0
        assert dataset_queue["single_owner_policy_contract_met"] is True
        assert user_manifest["browser_mutation_write_policy"][
            "label_mutation_target_kind"
        ] == "task_scoped_training_zarr"
        assert user_manifest["browser_mutation_write_policy"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert user_manifest["browser_mutation_write_policy"][
            "csv_handoff_artifact_role"
        ] == "metadata_only_control_plane"
        assert user_manifest["browser_mutation_write_policy"][
            "csv_handoff_artifacts_are_label_write_targets"
        ] is False
        assert user_manifest["browser_mutation_write_policy"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert user_manifest["browser_mutation_write_policy"][
            "browser_writes_handoff_csv"
        ] is False
        assert user_manifest["browser_mutation_write_policy"][
            "browser_writes_intermediate_csv"
        ] is False
        assert user_manifest["browser_mutation_target_contract_met"] is True
        assert user_manifest["direct_browser_start_contract_met"] is True
        assert user_manifest["single_owner_policy_contract_met"] is True
        assert user_manifest["assignment_ownership_integrity"][
            "duplicate_active_owner_count"
        ] == 0
        assert user_manifest["single_owner_policy"]["one_active_owner"] is True
        assert user_manifest["dataset_queue_direct_start_policy"][
            "label_mutation_target_kind"
        ] == "task_scoped_training_zarr"
        assert user_manifest["dataset_queue_direct_start_policy"][
            "browser_label_write_target"
        ] == "training_zarr"
        assert user_manifest["dataset_queue_direct_start_policy"][
            "csv_handoff_artifact_role"
        ] == "metadata_only_control_plane"
        assert user_manifest["dataset_queue_direct_start_policy"][
            "csv_handoff_artifacts_are_label_write_targets"
        ] is False
        assert user_manifest["dataset_queue_direct_start_policy"][
            "browser_writes_csv_or_handoff_files"
        ] is False
        assert user_manifest["dataset_queue_direct_start_policy"][
            "browser_writes_handoff_csv"
        ] is False
        assert user_manifest["dataset_queue_direct_start_policy"][
            "browser_writes_intermediate_csv"
        ] is False
    assert alice_dataset_queue["progress_summary"] == alice_manifest["progress_summary"]
    assert alice_dataset_queue["dataset_queue_summary"] == alice_manifest["dataset_queue_summary"]
    assert alice_dataset_queue["direct_browser_start_contract_summary"] == alice_manifest[
        "direct_browser_start_contract_summary"
    ]
    assert alice_dataset_queue["direct_browser_start_contract_summary"]["ready"] is True
    assert alice_dataset_queue["direct_browser_start_contract_summary"][
        "browser_label_write_target"
    ] == "training_zarr"
    assert alice_dataset_queue["direct_browser_start_contract_summary"][
        "browser_has_direct_zarr_write_authority"
    ] is False
    assert alice_dataset_queue["dataset_queue_state"]["code"] == "has_open_dataset_work"
    assert alice_dataset_queue["dataset_queue_state"] == alice_manifest["dataset_queue_state"]
    assert alice_dataset_queue["expected_user_dataset_queue_url"] == alice_manifest["expected_user_dataset_queue_url"]
    assert alice_dataset_queue["expected_user_labeling_home_url"] == (
        alice_manifest["expected_user_labeling_home_url"]
    )
    assert alice_dataset_queue["expected_user_labeling_home_url"] == (
        "https://labeling.example.org/labeling?expected_user=alice"
    )
    assert alice_dataset_queue["expected_user_personal_dataset_queue_url"] == (
        alice_manifest["expected_user_personal_dataset_queue_url"]
    )
    assert alice_dataset_queue["expected_user_personal_dataset_queue_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert alice_dataset_queue["expected_user_personal_work_url"] == (
        "https://labeling.example.org/my-work?expected_user=alice"
    )
    assert alice_dataset_queue["personalized_labeler_entrypoint"] == (
        "personal_datasets_waiting_queue"
    )
    assert alice_dataset_queue["personalized_labeler_entry_url"] == (
        "https://labeling.example.org/my-datasets?expected_user=alice"
    )
    assert alice_dataset_queue["queue_first_entry_contract"]["ready"] is True
    assert alice_dataset_queue["queue_first_entry_contract"]["labeling_home_ready"] is True
    assert alice_dataset_queue["queue_first_entry_contract"]["expected_user_labeling_home_url"] == (
        "https://labeling.example.org/labeling?expected_user=alice"
    )
    assert alice_dataset_queue["queue_first_entry_contract"][
        "personalized_labeler_entry_url"
    ] == "https://labeling.example.org/my-datasets?expected_user=alice"
    assert alice_dataset_queue["queue_first_entry_contract"][
        "personalized_labeler_entry_url_matches_personal_dataset_queue"
    ] is True
    assert alice_manifest["queue_first_entry_contract"] == alice_dataset_queue[
        "queue_first_entry_contract"
    ]
    assert alice_dataset_queue["dataset_queue_preview_url"] == alice_manifest["dataset_queue_preview_url"]
    assert alice_dataset_queue["browser_mutation_write_checklist"] == alice_manifest[
        "browser_mutation_write_checklist"
    ]
    assert alice_dataset_queue["browser_mutation_write_checklist"]["ready"] is True
    assert alice_dataset_queue["browser_mutation_write_checklist"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert alice_dataset_queue["browser_mutation_write_checklist"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert alice_dataset_queue["browser_mutation_write_checklist"]["browser_writes_handoff_csv"] is False
    assert alice_dataset_queue["browser_mutation_write_checklist"]["browser_writes_intermediate_csv"] is False
    assert alice_manifest["browser_mutation_write_policy"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert alice_manifest["browser_mutation_write_policy"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert alice_manifest["browser_mutation_write_policy"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert alice_manifest["browser_mutation_write_policy"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert alice_manifest["browser_mutation_write_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert alice_manifest["browser_mutation_write_policy"]["browser_writes_handoff_csv"] is False
    assert alice_manifest["browser_mutation_write_policy"]["browser_writes_intermediate_csv"] is False
    assert alice_manifest["dataset_queue_direct_start_policy"]["label_mutation_target_kind"] == (
        "task_scoped_training_zarr"
    )
    assert alice_manifest["dataset_queue_direct_start_policy"]["browser_label_write_target"] == (
        "training_zarr"
    )
    assert alice_manifest["dataset_queue_direct_start_policy"]["csv_handoff_artifact_role"] == (
        "metadata_only_control_plane"
    )
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "csv_handoff_artifacts_are_label_write_targets"
    ] is False
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "post_body_expected_user_required"
    ] is True
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "post_body_expected_user_field"
    ] == "expected_user"
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "denied_start_returns_task_open_authorization_contract"
    ] is True
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "denied_start_support_preserves_task_open_authorization_contract"
    ] is True
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "denied_start_support_includes_authorization_context"
    ] is True
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "denied_start_contract_reports_no_session_created"
    ] is True
    assert alice_manifest["dataset_queue_direct_start_policy"][
        "denied_start_contract_reports_server_authorizes_open_false"
    ] is True
    assert alice_manifest["dataset_queue_direct_start_policy"]["browser_writes_csv_or_handoff_files"] is False
    assert alice_manifest["dataset_queue_direct_start_policy"]["browser_writes_handoff_csv"] is False
    assert alice_manifest["dataset_queue_direct_start_policy"]["browser_writes_intermediate_csv"] is False
    assert alice_dataset_queue["browser_mutation_write_policy"] == alice_manifest[
        "browser_mutation_write_policy"
    ]
    assert alice_dataset_queue["dataset_queue_direct_start_policy"] == alice_manifest[
        "dataset_queue_direct_start_policy"
    ]
    assert alice_dataset_queue["runtime_operator_validation_gate_cli_policy"] == alice_manifest[
        "runtime_operator_validation_gate_cli_policy"
    ]
    assert alice_dataset_queue["dataset_queue"] == alice_dataset_queue["datasets"]
    assert "Preview your personalized dataset queue" in alice_html
    assert "Preview your datasets-waiting landing page" in alice_html
    assert "https://labeling.example.org?expected_user=alice" in alice_html
    assert "https://labeling.example.org/datasets?expected_user=alice" in alice_html
    assert "https://labeling.example.org/work?expected_user=alice" in alice_html
    assert "assigned task/training Zarr scope" in alice_html
    assert "not label write targets" in alice_html
    assert "Each recording has one active assigned owner" in alice_html
    assert "record-browser-smoke-evidence --evidence" not in alice_html
    assert "apply-operator-evidence-templates --path" not in alice_html
    assert "record-zarr-backup-evidence --evidence" not in alice_html
    assert "record-browser-smoke-evidence --evidence" not in alice_message
    assert "apply-operator-evidence-templates --path" not in alice_message
    assert "record-zarr-backup-evidence --evidence" not in alice_message
    assert "record-browser-smoke-evidence --evidence" not in alice_quickstart
    assert "apply-operator-evidence-templates --path" not in alice_quickstart
    assert "record-zarr-backup-evidence --evidence" not in alice_quickstart
    assert {row["user"] for row in roster_rows} == {"alice", "bob"}
    assert {row["ready_to_send"] for row in roster_rows} == {"False"}
    assert {row["operator_validation_required_before_invite"] for row in roster_rows} == {"True"}
    assert {row["operator_validation_all_complete"] for row in roster_rows} == {"False"}
    assert {row["operator_validation_status"] for row in roster_rows} == {"pending_operator_evidence"}
    assert {row["operator_validation_command_template_schema"] for row in roster_rows} == {
        "palette.web_labeling_operator_validation_command_templates.v1"
    }
    assert {row["operator_validation_command_template_commands_are_operator_only"] for row in roster_rows} == {
        "True"
    }
    assert {
        row["operator_validation_command_template_commands_are_labeler_instructions"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["operator_validation_command_template_labelers_must_not_run_commands"]
        for row in roster_rows
    } == {"True"}
    assert {row["operator_validation_command_template_command_count"] for row in roster_rows} == {"7"}
    assert all(
        "record_browser_smoke_evidence"
        in row["operator_validation_command_template_command_ids"]
        for row in roster_rows
    )
    assert all(
        "browser_smoke" in row["operator_validation_command_template_template_backed_gate_ids"]
        for row in roster_rows
    )
    assert all(
        "browser_smoke" in row["operator_validation_command_template_apply_required_gate_ids"]
        for row in roster_rows
    )
    assert all(
        "browser_smoke_evidence_template"
        in row[
            "operator_validation_command_template_evidence_template_fields_by_gate_id"
        ]
        for row in roster_rows
    )
    assert all(
        "browser-smoke-evidence-template.json"
        in row[
            "operator_validation_command_template_evidence_template_paths_by_gate_id"
        ]
        for row in roster_rows
    )
    assert {row["operator_validation_command_template_missing_command_gate_ids"] for row in roster_rows} == {
        "[]"
    }
    assert {
        row["operator_validation_command_template_launch_evidence_collection_plan_schema"]
        for row in roster_rows
    } == {"palette.web_labeling_launch_evidence_collection_plan.v1"}
    assert {
        row["operator_validation_command_template_launch_evidence_collection_step_count"]
        for row in roster_rows
    } == {"6"}
    assert {
        row["operator_validation_command_template_launch_evidence_collection_operator_only"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["operator_validation_command_template_launch_evidence_collection_required_final_field"]
        for row in roster_rows
    } == {"labeler_links_safe_to_share"}
    assert {
        row["operator_validation_command_template_launch_evidence_collection_required_final_value"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["operator_validation_command_template_launch_evidence_collection_final_inspection_command"]
        for row in roster_rows
    } == {"inspect-handoff --path PACKAGE --require-shareable"}
    assert all(
        "browser_smoke"
        in row[
            "operator_validation_command_template_launch_evidence_collection_gate_ids"
        ]
        for row in roster_rows
    )
    assert all(
        "record_browser_smoke_evidence"
        in row[
            "operator_validation_command_template_launch_evidence_collection_record_command_ids"
        ]
        for row in roster_rows
    )
    assert all(
        "browser_response_security_headers" in row["operator_validation_pending_gate_ids"]
        for row in roster_rows
    )
    assert {row["operator_validation_needs_review_gate_ids"] for row in roster_rows} == {"[]"}
    for gate_id in (
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "operator_recovery_contract",
    ):
        assert {
            row[f"operator_validation_gate_{gate_id}_status"] for row in roster_rows
        } == {"missing_evidence"}
        assert {
            row[f"operator_validation_gate_{gate_id}_pending"] for row in roster_rows
        } == {"True"}
        assert {
            row[f"operator_validation_gate_{gate_id}_missing_evidence"]
            for row in roster_rows
        } == {"True"}
        assert {
            row[f"operator_validation_gate_{gate_id}_needs_review"] for row in roster_rows
        } == {"False"}
        assert {
            row[f"operator_validation_gate_{gate_id}_passed"] for row in roster_rows
        } == {"False"}
    assert {
        "Complete required operator validation evidence" in row["operator_validation_operator_action"]
        for row in roster_rows
    } == {True}
    assert {row["known_labeler"] for row in roster_rows} == {"True"}
    assert {row["known_user_readiness"] for row in roster_rows} == {"passed"}
    assert {row["known_user_active_assignment_count"] for row in roster_rows} == {"1", "2"}
    assert {row["assignment_ownership_ok"] for row in roster_rows} == {"True"}
    assert {row["assignment_duplicate_active_owner_count"] for row in roster_rows} == {"0"}
    assert {row["assignment_ownership_readiness"] for row in roster_rows} == {"passed"}
    assert {row["assignment_ownership_contract_ready"] for row in roster_rows} == {"True"}
    assert {
        row["assignment_ownership_contract_assignment_scope"] for row in roster_rows
    } == {"recording"}
    assert {
        row["assignment_ownership_contract_recording_assignment_key"] for row in roster_rows
    } == {"recording_id"}
    assert {
        row["assignment_ownership_contract_primary_key_columns"] for row in roster_rows
    } == {'["recording_id"]'}
    assert {
        row["assignment_ownership_contract_one_active_owner"] for row in roster_rows
    } == {"True"}
    assert {
        row["assignment_ownership_contract_multiple_labelers_per_recording_allowed"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["assignment_ownership_contract_browser_mutation_requires_current_assignment_owner"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["assignment_ownership_contract_duplicate_active_owner_count"]
        for row in roster_rows
    } == {"0"}
    assert {row["single_owner_policy_assignment_scope"] for row in roster_rows} == {
        "recording"
    }
    assert {row["single_owner_policy_recording_assignment_key"] for row in roster_rows} == {
        "recording_id"
    }
    assert {
        row["single_owner_policy_multiple_labelers_per_recording_allowed"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["single_owner_policy_browser_mutation_requires_current_assignment_owner"]
        for row in roster_rows
    } == {"True"}
    assert {row["guarded_links_ready"] for row in roster_rows} == {"True"}
    assert {row["missing_guarded_links"] for row in roster_rows} == {"[]"}
    assert {row["handoff_artifacts_ready"] for row in roster_rows} == {"True"}
    assert {row["missing_handoff_artifacts"] for row in roster_rows} == {"[]"}
    assert {row["handoff_entry_readiness"] for row in roster_rows} == {"passed"}
    assert {row["labeler_safety_policy_present"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_ready"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_readiness"] for row in roster_rows} == {"passed"}
    assert {row["labeler_safety_labeler_runtime_surface"] for row in roster_rows} == {"browser"}
    assert {row["labeler_safety_requires_local_palette_installation"] for row in roster_rows} == {"False"}
    assert {row["labeler_safety_requires_local_crimson_installation"] for row in roster_rows} == {"False"}
    assert {row["labeler_safety_requires_local_conda_environment"] for row in roster_rows} == {"False"}
    assert {row["labeler_safety_requires_local_project_dependencies"] for row in roster_rows} == {"False"}
    assert {row["labeler_safety_no_local_install_required"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_identity_probe_expected_user_guard_required"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_identity_probe_diagnostic_only"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_identity_probe_does_not_authorize_work"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_identity_probe_unknown_user_blocks_work_surfaces"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_identity_probe_success_launch_ctas_rendered"] for row in roster_rows} == {"True"}
    assert {row["labeler_safety_identity_probe_failed_launch_ctas_suppressed"] for row in roster_rows} == {"True"}
    assert {
        row["labeler_safety_identity_probe_failed_support_urls_diagnostic_only"]
        for row in roster_rows
    } == {"True"}
    assert {row["labeler_safety_browser_receives_raw_zarr_paths"] for row in roster_rows} == {"False"}
    assert {row["labeler_route_authorization_policy_present"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_ready"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_readiness"] for row in roster_rows} == {"passed"}
    assert {row["labeler_route_authorization_expected_user_match_required"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_known_user_required"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_active_assignment_required"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_task_open_requires_active_assignment"] for row in roster_rows} == {
        "True"
    }
    assert {row["labeler_route_authorization_task_open_requires_startable_task_state"] for row in roster_rows} == {
        "True"
    }
    assert {row["labeler_route_authorization_startable_task_states"] for row in roster_rows} == {
        '["pending", "in_progress"]'
    }
    assert {row["labeler_route_authorization_signed_links_are_entry_hints"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_signed_links_recheck_runtime_start_gate"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_forwarded_links_rechecked"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_runtime_checklist_present"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_runtime_checklist_ready"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_single_owner_store_proof_ready"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_assignment_ownership_integrity_ok"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_duplicate_active_owner_count"] for row in roster_rows} == {"0"}
    assert {row["labeler_route_authorization_browser_mutation_target_resolved_server_side"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_labelers_mutate_assigned_training_zarrs"] for row in roster_rows} == {"True"}
    assert {row["labeler_route_authorization_labelers_mutate_intermediate_csvs"] for row in roster_rows} == {"False"}
    assert {
        row["labeler_route_authorization_single_owner_store_proof_required_for_browser_work"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["labeler_route_authorization_single_owner_store_proof_requires_integrity_ok"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["labeler_route_authorization_single_owner_store_proof_requires_zero_duplicate_active_owners"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["labeler_route_authorization_single_owner_store_proof_requires_training_zarr_target"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["labeler_route_authorization_single_owner_store_proof_rejects_intermediate_csv_mutation"]
        for row in roster_rows
    } == {"True"}
    assert {row["signed_link_policy_present"] for row in roster_rows} == {"True"}
    assert {row["signed_link_policy_ready"] for row in roster_rows} == {"True"}
    assert {row["signed_link_policy_readiness"] for row in roster_rows} == {"passed"}
    assert {row["signed_link_authorization_grant"] for row in roster_rows} == {"False"}
    assert {row["signed_link_forwarded_links_recheck_identity"] for row in roster_rows} == {"True"}
    assert {
        row["signed_link_runtime_operator_validation_start_gate_enforced"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["signed_link_operator_validation_start_gate_checked_before_session_create"]
        for row in roster_rows
    } == {"True"}
    assert {row["session_guard_policy_present"] for row in roster_rows} == {"True"}
    assert {row["session_guard_policy_ready"] for row in roster_rows} == {"True"}
    assert {row["session_guard_policy_readiness"] for row in roster_rows} == {"passed"}
    assert {row["session_guard_stale_tab_save_rejected"] for row in roster_rows} == {"True"}
    assert {row["session_guard_non_startable_task_sessions_rejected"] for row in roster_rows} == {"True"}
    assert {row["task_state_policy_present"] for row in roster_rows} == {"True"}
    assert {row["task_state_policy_ready"] for row in roster_rows} == {"True"}
    assert {row["task_state_policy_readiness"] for row in roster_rows} == {"passed"}
    assert {row["task_state_startable_task_states"] for row in roster_rows} == {'["pending", "in_progress"]'}
    assert {row["task_state_completed_tasks_read_only"] for row in roster_rows} == {"True"}
    assert {row["task_state_non_startable_task_open_requests"] for row in roster_rows} == {
        "reject_task_not_startable"
    }
    assert {row["task_state_non_startable_task_save_requests"] for row in roster_rows} == {
        "reject_task_not_startable"
    }
    assert {row["task_state_requires_current_target_token"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_policy_present"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_ready"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_readiness"] for row in roster_rows} == {"passed"}
    assert {
        row["operator_recovery_reassignment_closes_previous_owner_sessions_before_assignment_update"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["operator_recovery_reassignment_target_validated_before_session_closure"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["operator_recovery_session_closure_and_assignment_update_atomic"]
        for row in roster_rows
    } == {"True"}
    assert {row["operator_recovery_task_reopen_operator_only"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_failed_promotion_retry_operator_only"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_rollback_requires_backup_plan"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_bad_disposable_mutation_recovery_ready"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_disposable_mutation_smoke_requires_recovery_path_verification"] for row in roster_rows} == {"True"}
    assert {row["operator_recovery_audit_event_lookup_route"] for row in roster_rows} == {
        "/api/admin/events/{event_id}"
    }
    assert {row["operator_recovery_task_repair_route"] for row in roster_rows} == {
        "/api/admin/tasks/{task_id}/repair"
    }
    assert {row["operator_recovery_failed_promotion_retry_route"] for row in roster_rows} == {
        "/api/admin/events/{event_id}/retry-promotion"
    }
    assert {row["operator_recovery_reassignment_session_repair_route"] for row in roster_rows} == {
        "/api/admin/recordings/{recording_id}/repair-reassignment-sessions"
    }
    assert {row["zarr_backup_policy_present"] for row in roster_rows} == {"True"}
    assert {row["zarr_backup_ready"] for row in roster_rows} == {"True"}
    assert {row["zarr_backup_readiness"] for row in roster_rows} == {"passed"}
    assert {row["zarr_backup_copy_before_labeling"] for row in roster_rows} == {"True"}
    assert {row["zarr_backup_labelers_do_not_receive_backup_paths"] for row in roster_rows} == {"True"}
    assert {row["mutation_audit_policy_present"] for row in roster_rows} == {"True"}
    assert {row["mutation_audit_ready"] for row in roster_rows} == {"True"}
    assert {row["mutation_audit_readiness"] for row in roster_rows} == {"passed"}
    assert {row["mutation_audit_event_store"] for row in roster_rows} == {"labeling_task_events"}
    assert {row["mutation_audit_server_records_events"] for row in roster_rows} == {"True"}
    assert {row["mutation_audit_browser_records_events_directly"] for row in roster_rows} == {"False"}
    assert {row["browser_response_security_policy_present"] for row in roster_rows} == {"True"}
    assert {row["browser_response_security_ready"] for row in roster_rows} == {"True"}
    assert {row["browser_response_security_readiness"] for row in roster_rows} == {"passed"}
    assert {row["browser_response_security_no_store_cache"] for row in roster_rows} == {"True"}
    assert {row["browser_response_security_clickjacking_protection"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_write_policy_present"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_write_ready"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_write_readiness"] for row in roster_rows} == {"passed"}
    assert {row["browser_mutation_target_contract_met"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_target_mismatch_count"] for row in roster_rows} == {"0"}
    assert {row["direct_browser_start_contract_met"] for row in roster_rows} == {"True"}
    assert {row["direct_browser_start_mismatch_count"] for row in roster_rows} == {"0"}
    assert {row["single_owner_policy_contract_met"] for row in roster_rows} == {"True"}
    assert {row["dashboard_url"] for row in roster_rows} == {"https://labeling.example.org/work"}
    assert {row["expected_user_labeler_landing_url"] for row in roster_rows} == {
        "https://labeling.example.org?expected_user=alice",
        "https://labeling.example.org?expected_user=bob",
    }
    assert {row["expected_user_labeling_home_url"] for row in roster_rows} == {
        "https://labeling.example.org/labeling?expected_user=alice",
        "https://labeling.example.org/labeling?expected_user=bob",
    }
    assert {row["expected_user_dashboard_url"] for row in roster_rows} == {
        "https://labeling.example.org/work?expected_user=alice",
        "https://labeling.example.org/work?expected_user=bob",
    }
    assert {row["expected_user_dataset_queue_url"] for row in roster_rows} == {
        "https://labeling.example.org/datasets?expected_user=alice",
        "https://labeling.example.org/datasets?expected_user=bob",
    }
    assert {row["expected_user_personal_work_url"] for row in roster_rows} == {
        "https://labeling.example.org/my-work?expected_user=alice",
        "https://labeling.example.org/my-work?expected_user=bob",
    }
    assert {row["expected_user_personal_dataset_queue_url"] for row in roster_rows} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["preferred_labeler_entrypoint"] for row in roster_rows} == {
        "personal_datasets_waiting_queue"
    }
    assert {row["preferred_labeler_entry_url"] for row in roster_rows} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["personalized_labeler_entrypoint"] for row in roster_rows} == {
        "personal_datasets_waiting_queue"
    }
    assert {row["personalized_labeler_entry_url"] for row in roster_rows} == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["queue_first_entry_contract_schema"] for row in roster_rows} == {
        alice_manifest["queue_first_entry_contract"]["schema"]
    }
    assert {row["queue_first_entry_contract_ready"] for row in roster_rows} == {"True"}
    assert {
        row["queue_first_entry_contract_preferred_labeler_entrypoint"]
        for row in roster_rows
    } == {"personal_datasets_waiting_queue"}
    assert {
        row["queue_first_entry_contract_preferred_labeler_entry_url"]
        for row in roster_rows
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {
        row["queue_first_entry_contract_personalized_labeler_entrypoint"]
        for row in roster_rows
    } == {"personal_datasets_waiting_queue"}
    assert {
        row["queue_first_entry_contract_personalized_labeler_entry_url"]
        for row in roster_rows
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {
        row["queue_first_entry_contract_personalized_entry_required"]
        for row in roster_rows
    } == {"True"}
    assert {
        row[
            "queue_first_entry_contract_personalized_labeler_entry_url_matches_personal_dataset_queue"
        ]
        for row in roster_rows
    } == {"True"}
    assert {
        row[
            "queue_first_entry_contract_preferred_labeler_entry_url_matches_personal_dataset_queue"
        ]
        for row in roster_rows
    } == {"True"}
    assert {
        row[
            "queue_first_entry_contract_preferred_labeler_entry_url_is_expected_user_guarded"
        ]
        for row in roster_rows
    } == {"True"}
    assert {
        row[
            "queue_first_entry_contract_personalized_labeler_entry_url_is_expected_user_guarded"
        ]
        for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_landing_ready"] for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_labeling_home_ready"] for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_dataset_queue_ready"] for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_personal_dataset_queue_ready"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_personal_work_ready"] for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_queue_first_paths_ready"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_datasets_waiting_aliases_ready"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_expected_user_landing_guard"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_expected_user_queue_guard"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["queue_first_entry_contract_expected_user_dashboard_guard"]
        for row in roster_rows
    } == {"True"}
    assert {row["personal_dataset_queue_link_role"] for row in roster_rows} == {
        "preferred_queue"
    }
    assert {row["dataset_queue_link_role"] for row in roster_rows} == {
        "canonical_queue_fallback"
    }
    assert {row["canonical_dataset_queue_link_role"] for row in roster_rows} == {
        "canonical_queue_fallback"
    }
    assert {row["dashboard_link_role"] for row in roster_rows} == {"fallback_dashboard"}
    assert {row["identity_probe_link_role"] for row in roster_rows} == {"identity_check"}
    assert {row["task_links_role"] for row in roster_rows} == {"convenience_entry_hints"}
    assert {row["preferred_labeler_entry_url_matches_dataset_queue"] for row in roster_rows} == {"True"}
    assert {
        row["preferred_labeler_entry_url_matches_personal_dataset_queue"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["personalized_labeler_entry_url_matches_personal_dataset_queue"]
        for row in roster_rows
    } == {"True"}
    assert {row["expected_user_identity_probe_url"] for row in roster_rows} == {
        "https://labeling.example.org/identity?expected_user=alice",
        "https://labeling.example.org/identity?expected_user=bob",
    }
    assert {row["dataset_queue_direct_start_label_mutation_target_kind"] for row in roster_rows} == {
        "task_scoped_training_zarr"
    }
    assert {row["dataset_queue_direct_start_browser_label_write_target"] for row in roster_rows} == {
        "training_zarr"
    }
    assert {
        row["dataset_queue_direct_start_post_body_expected_user_required"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["dataset_queue_direct_start_post_body_expected_user_field"]
        for row in roster_rows
    } == {"expected_user"}
    assert {
        row["dataset_queue_direct_start_denied_start_returns_task_open_authorization_contract"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["dataset_queue_direct_start_denied_start_support_preserves_task_open_authorization_contract"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["dataset_queue_direct_start_denied_start_support_includes_authorization_context"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["dataset_queue_direct_start_denied_start_contract_reports_no_session_created"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["dataset_queue_direct_start_denied_start_contract_reports_server_authorizes_open_false"]
        for row in roster_rows
    } == {"True"}
    assert {row["dataset_queue_direct_start_csv_handoff_artifact_role"] for row in roster_rows} == {
        "metadata_only_control_plane"
    }
    assert {
        row["dataset_queue_direct_start_csv_handoff_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["dataset_queue_direct_start_handoff_csv_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["dataset_queue_direct_start_intermediate_csv_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {row["dataset_queue_direct_start_browser_writes_csv_or_handoff_files"] for row in roster_rows} == {
        "False"
    }
    assert {row["dataset_queue_direct_start_browser_writes_handoff_csv"] for row in roster_rows} == {
        "False"
    }
    assert {row["dataset_queue_direct_start_browser_writes_intermediate_csv"] for row in roster_rows} == {
        "False"
    }
    assert {row["dataset_queue_direct_start_browser_receives_zarr_write_authority"] for row in roster_rows} == {
        "False"
    }
    assert {row["dataset_queue_direct_start_browser_has_direct_zarr_write_authority"] for row in roster_rows} == {
        "False"
    }
    assert {
        row["runtime_operator_validation_gate_cli_policy_preferred_require_flag"]
        for row in roster_rows
    } == {"--require-operator-validation-for-browser-work"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_validation_checklist_flag"]
        for row in roster_rows
    } == {"--validation-checklist"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_protects_browser_start_open"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_protects_browser_mutations"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_target_token_check"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_zarr_write"]
        for row in roster_rows
    } == {"True"}
    assert {
        row["runtime_operator_validation_gate_cli_policy_blocks_before_audit_event_creation"]
        for row in roster_rows
    } == {"True"}
    assert {row["safe_share_gate_schema"] for row in roster_rows} == {
        "palette.web_labeling_safe_share_gate.v1"
    }
    assert {row["safe_share_gate_id"] for row in roster_rows} == {"labeler_links_safe_to_share"}
    assert {row["safe_share_requires_require_shareable_inspection"] for row in roster_rows} == {
        "True"
    }
    assert {row["safe_share_ready_to_send_is_sufficient"] for row in roster_rows} == {"False"}
    assert {row["safe_share_required_inspection_field"] for row in roster_rows} == {
        "labeler_links_safe_to_share"
    }
    assert {row["safe_share_required_inspection_value"] for row in roster_rows} == {"True"}
    assert all(
        "disposable_zarr_mutation_smoke" in row["safe_share_launch_blocking_evidence_gate_ids"]
        for row in roster_rows
    )
    assert {row["safe_share_launch_blocking_gate_count"] for row in roster_rows} == {"6"}
    assert {row["safe_share_launch_blocking_missing_evidence_gate_count"] for row in roster_rows} == {
        "6"
    }
    assert {row["safe_share_launch_blocking_unsatisfied_gate_count"] for row in roster_rows} == {
        "6"
    }
    assert {row["safe_share_checklist_gate_evidence_complete"] for row in roster_rows} == {
        "False"
    }
    assert {row["safe_share_launch_blocking_next_action_count"] for row in roster_rows} == {
        "6"
    }
    assert all(
        row["safe_share_next_action_summary"].startswith("Safe-share next actions: 6;")
        for row in roster_rows
    )
    assert all(
        "browser_smoke" in row["safe_share_launch_blocking_next_actions"]
        for row in roster_rows
    )
    assert all(
        "browser_smoke" in row["safe_share_launch_blocking_missing_evidence_gate_ids"]
        for row in roster_rows
    )
    assert all(
        "operator_recovery_contract" in row["safe_share_launch_blocking_unsatisfied_gate_ids"]
        for row in roster_rows
    )
    assert all(
        "labeler_links_safe_to_share is true" in row["safe_share_operator_action"]
        for row in roster_rows
    )
    assert {row["personalized_launch_readiness_schema"] for row in roster_rows} == {
        "palette.web_labeling_personalized_launch_readiness.v1"
    }
    assert all(
        "browser_label_write_target" in row["personalized_launch_readiness_fields"]
        for row in roster_rows
    )
    assert {
        row["personalized_launch_readiness_personalized_labeler_entry_url"]
        for row in roster_rows
    } == {
        "https://labeling.example.org/my-datasets?expected_user=alice",
        "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert {row["personalized_launch_readiness_browser_label_write_target"] for row in roster_rows} == {
        "training_zarr"
    }
    assert {
        row["personalized_launch_readiness_browser_writes_csv_or_handoff_files"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["personalized_launch_readiness_browser_has_direct_zarr_write_authority"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["personalized_launch_readiness_external_launch_evidence_gap_action_required"]
        for row in roster_rows
    } == {"True"}
    assert all(
        "browser_smoke"
        in row["personalized_launch_readiness_external_launch_evidence_gap_gate_ids"]
        for row in roster_rows
    )
    assert all(
        "record_browser_smoke_evidence"
        in row["personalized_launch_readiness_external_launch_evidence_gap_todos"]
        for row in roster_rows
    )
    assert {
        row["personalized_launch_readiness_external_launch_evidence_gap_todo_count"]
        for row in roster_rows
    } == {"6"}
    assert all(
        "operator_validation_evidence_template_path"
        in row[
            "personalized_launch_readiness_external_launch_evidence_gap_todo_fields"
        ]
        for row in roster_rows
    )
    assert all(
        "browser-smoke-evidence-template.json"
        in row[
            "personalized_launch_readiness_external_launch_evidence_gap_template_paths_by_gate_id"
        ]
        for row in roster_rows
    )
    assert all(
        "record_browser_smoke_evidence"
        in row[
            "personalized_launch_readiness_external_launch_evidence_gap_record_command_ids_by_gate_id"
        ]
        for row in roster_rows
    )
    assert all(
        "record_browser_smoke_evidence"
        in row["safe_share_external_launch_evidence_gap_todos"]
        for row in roster_rows
    )
    assert {row["safe_share_external_launch_evidence_gap_todo_count"] for row in roster_rows} == {
        "6"
    }
    assert {row["direct_browser_start_contract_summary_schema"] for row in roster_rows} == {
        "palette.web_labeling_direct_browser_start_contract_summary.v1"
    }
    assert {row["direct_browser_start_contract_summary_ready"] for row in roster_rows} == {"True"}
    assert {row["direct_browser_start_contract_summary_expected_user_guard_enforced_by_api"] for row in roster_rows} == {
        "True"
    }
    assert {row["direct_browser_start_contract_summary_server_rechecks_on_post"] for row in roster_rows} == {
        "True"
    }
    assert {row["direct_browser_start_contract_summary_label_mutation_target_kind"] for row in roster_rows} == {
        "task_scoped_training_zarr"
    }
    assert {row["direct_browser_start_contract_summary_browser_label_write_target"] for row in roster_rows} == {
        "training_zarr"
    }
    assert {row["direct_browser_start_contract_summary_csv_handoff_artifact_role"] for row in roster_rows} == {
        "metadata_only_control_plane"
    }
    assert {
        row["direct_browser_start_contract_summary_csv_handoff_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["direct_browser_start_contract_summary_handoff_csv_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["direct_browser_start_contract_summary_intermediate_csv_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {row["direct_browser_start_contract_summary_browser_writes_csv_or_handoff_files"] for row in roster_rows} == {
        "False"
    }
    assert {row["direct_browser_start_contract_summary_browser_has_direct_zarr_write_authority"] for row in roster_rows} == {
        "False"
    }
    assert {row["task_state_labeler_promotion_retry_mutation_enabled"] for row in roster_rows} == {
        "False"
    }
    assert {row["task_state_labeler_promotion_retry_rejection_error"] for row in roster_rows} == {
        "operator_support_required"
    }
    assert {row["task_state_ordinary_labeler_promotion_retry_mutation"] for row in roster_rows} == {
        "operator_support_required"
    }
    assert {row["browser_mutation_authoritative_label_state"] for row in roster_rows} == {
        "assigned_task_zarr_scope"
    }
    assert {row["browser_mutation_data_plane_write_target"] for row in roster_rows} == {
        "server_owned_assigned_task_zarr_scope"
    }
    assert {row["browser_mutation_mutable_label_data_plane"] for row in roster_rows} == {
        "task_scoped_training_zarr"
    }
    assert {row["browser_mutation_label_mutation_target_kind"] for row in roster_rows} == {
        "task_scoped_training_zarr"
    }
    assert {row["browser_mutation_browser_label_write_target"] for row in roster_rows} == {
        "training_zarr"
    }
    assert {row["browser_mutation_server_mutates_task_scoped_zarr_targets"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_training_zarr_mutations_are_server_owned"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_promotion_training_zarr_requires_task_scope"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_handoff_artifacts_are_metadata_only"] for row in roster_rows} == {"True"}
    assert {row["browser_mutation_csv_handoff_artifact_role"] for row in roster_rows} == {
        "metadata_only_control_plane"
    }
    assert {row["browser_mutation_csv_handoff_artifacts_are_label_write_targets"] for row in roster_rows} == {
        "False"
    }
    assert {
        row["browser_mutation_handoff_csv_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {
        row["browser_mutation_intermediate_csv_artifacts_are_label_write_targets"]
        for row in roster_rows
    } == {"False"}
    assert {row["browser_mutation_browser_writes_csv_or_handoff_files"] for row in roster_rows} == {"False"}
    assert {row["browser_mutation_browser_writes_handoff_csv"] for row in roster_rows} == {"False"}
    assert {row["browser_mutation_browser_writes_intermediate_csv"] for row in roster_rows} == {
        "False"
    }
    assert {row["identity_personal_queue_evidence_status"] for row in roster_rows} == {"missing"}
    assert {row["identity_personal_queue_evidence_ready_count"] for row in roster_rows} == {"0"}
    assert {row["identity_personal_queue_evidence_missing_count"] for row in roster_rows} == {"0"}
    assert {row["identity_personal_queue_evidence_ready_users"] for row in roster_rows} == {"[]"}
    assert {row["identity_personal_queue_evidence_missing_users"] for row in roster_rows} == {"[]"}
    assert {row["identity_personal_queue_evidence_missing_fields_by_user"] for row in roster_rows} == {
        "{}"
    }
    assert {row["identity_all_users_have_personal_queue_evidence"] for row in roster_rows} == {
        "False"
    }
    assert {row["operator_validation_external_evidence_required"] for row in roster_rows} == {
        "True"
    }
    assert {row["operator_validation_external_evidence_required_gate_count"] for row in roster_rows} == {
        "5"
    }
    assert all(
        "browser_smoke" in row["operator_validation_external_evidence_required_gate_ids"]
        for row in roster_rows
    )
    assert all(
        "browser_smoke_evidence_template"
        in row[
            "operator_validation_external_evidence_template_fields_by_gate_id"
        ]
        for row in roster_rows
    )
    assert all(
        "browser-smoke-evidence-template.json"
        in row[
            "operator_validation_external_evidence_template_paths_by_gate_id"
        ]
        for row in roster_rows
    )
    assert {row["operator_validation_checklist_only_required_gate_ids"] for row in roster_rows} == {
        "[\"dashboard_visibility\", \"final_signoff\", \"multi_user_dry_run\", \"one_labeler_dry_run\", \"operator_authorization_boundary\"]"
    }
    assert {row["operator_validation_checklist_only_required_gate_count"] for row in roster_rows} == {
        "5"
    }
    assert {row["browser_mutation_browser_receives_zarr_write_authority"] for row in roster_rows} == {"False"}
    assert {row["browser_mutation_browser_has_direct_zarr_write_authority"] for row in roster_rows} == {"False"}
    alice_roster = next(row for row in roster_rows if row["user"] == "alice")
    bob_roster = next(row for row in roster_rows if row["user"] == "bob")
    assert alice_roster["recordings_without_open_tasks"] == "1"
    assert alice_roster["recordings_without_open_tasks_by_reason"] == '{"tasks_not_generated": 1}'
    assert "Generate or import browser-labeling tasks" in alice_roster["recordings_without_open_tasks_actions"]
    assert alice_roster["redacted_summary_fields"] == "1"
    assert alice_roster["waiting_datasets"] == "1"
    assert alice_roster["dataset_open_tasks"] == "1"
    assert alice_roster["dataset_queue_state_code"] == "has_open_dataset_work"
    assert alice_roster["labeler_work_completion_status"] == "waiting"
    assert alice_roster["labeler_work_completion_has_waiting_work"] == "True"
    assert alice_roster["labeler_work_completion_completed"] == "False"
    assert alice_roster["dataset_queue_blocks_labeler_start"] == "False"
    assert alice_roster["dataset_queue_start_ready"] == "True"
    assert alice_roster["dataset_queue_start_status"] == "passed"
    assert alice_roster["dataset_queue"] == str(output_dir / "alice" / "dataset-queue.json")
    assert alice_roster["dataset_queue_preview_url"] == "https://labeling.example.org/my-datasets?expected_user=alice"
    assert (
        alice_roster["canonical_dataset_queue_preview_url"]
        == "https://labeling.example.org/datasets?expected_user=alice"
    )
    assert alice_roster["operator_validation_operator_only_fields"] == (
        "[\"operator_validation_checklist_path\"]"
    )
    assert (
        "identity_personal_queue_evidence_status"
        in alice_roster["operator_validation_public_fields"]
    )
    assert (
        "identity_personal_queue_evidence_ready_count"
        in alice_roster["operator_validation_public_fields"]
    )
    assert (
        "identity_all_users_have_personal_queue_evidence"
        in alice_roster["operator_validation_public_fields"]
    )
    assert (
        "operator_validation_external_evidence_required"
        in alice_roster["operator_validation_public_fields"]
    )
    assert (
        "operator_validation_external_evidence_template_paths_by_gate_id"
        in alice_roster["operator_validation_public_fields"]
    )
    assert alice_roster[
        "operator_validation_identity_personal_queue_evidence_status_values"
    ] == "[\"missing\", \"incomplete\", \"ready\"]"
    assert alice_roster["operator_validation_gate_status_values"] == (
        "[\"unknown\", \"pending\", \"missing_evidence\", \"needs_review\", \"passed\"]"
    )
    assert alice_roster["operator_validation_gate_ids"] == (
        "[\"mutable_zarr_backup_confirmation\", \"browser_response_security_headers\", "
        "\"identity_probe_verification\", \"browser_smoke\", "
        "\"disposable_zarr_mutation_smoke\", \"operator_recovery_contract\"]"
    )
    assert alice_roster["operator_validation_gate_flat_field_suffixes"] == (
        "[\"status\", \"pending\", \"missing_evidence\", \"needs_review\", \"passed\"]"
    )
    assert alice_roster["operator_validation_labeler_visible_payloads_include_operator_only_fields"] == "False"
    assert alice_roster["operator_validation_per_user_payloads_use_public_fields_only"] == "True"
    assert bob_roster["recordings_without_open_tasks"] == "0"
    assert bob_roster["redacted_summary_fields"] == "1"
    assert bob_roster["waiting_datasets"] == "1"
    assert bob_roster["dataset_queue_state_code"] == "has_open_dataset_work"
    assert bob_roster["labeler_work_completion_status"] == "waiting"
    assert bob_roster["labeler_work_completion_has_waiting_work"] == "True"
    assert bob_roster["dataset_queue_blocks_labeler_start"] == "False"
    assert bob_roster["dataset_queue_start_ready"] == "True"
    assert bob_roster["dataset_queue_start_status"] == "passed"
    assert all(row["links_expire_at_utc"] for row in roster_rows)
    assert {row["message"] for row in roster_rows} == {
        str(output_dir / "alice" / "message.txt"),
        str(output_dir / "bob" / "message.txt"),
    }
    assert {row["quickstart"] for row in roster_rows} == {
        str(output_dir / "alice" / "labeler-quickstart.txt"),
        str(output_dir / "bob" / "labeler-quickstart.txt"),
    }
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "handoffs/index.html" in names
        assert "handoffs/handoff-readme.txt" in names
        assert "handoffs/labeler-roster.csv" in names
        assert "handoffs/validation-log-template.md" in names
        assert "handoffs/validation-checklist.json" in names
        assert "handoffs/alice/index.html" in names
        assert "handoffs/alice/dataset-queue.json" in names
        assert "handoffs/bob/message.txt" in names
        assert "handoffs/bob/labeler-quickstart.txt" in names
    assert alice_manifest["counts"]["signed_links"] == 1
    assert bob_manifest["counts"]["signed_links"] == 1
    assert {link["expected_user"] for link in alice_links} == {"alice"}
    assert {link["expected_user"] for link in bob_links} == {"bob"}
    assert alice_links[0]["task_id"] == "task-a"
    assert bob_links[0]["task_id"] == "task-b"
    assert not (output_dir / "carol" / "manifest.json").exists()

    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(zip_path),
            "--output",
            str(tmp_path / "handoff-inspection.json"),
        ]
    )

    inspection = json.loads(capsys.readouterr().out)
    archived_inspection = json.loads((tmp_path / "handoff-inspection.json").read_text())
    assert rc == 2
    assert inspection["ok"] is False
    assert inspection["status"] == "needs_review"
    assert inspection["failure_reasons"] == [
        "handoff_not_ready",
        "validation_evidence_pending",
    ]
    assert "implementation_status_artifact_incomplete" not in inspection["failure_reasons"]
    failure_actions = "\n".join(inspection["failure_actions"])
    assert "update-validation-checklist" in failure_actions
    assert "identity_probe_verification" in failure_actions
    assert "browser_response_security_headers" in failure_actions
    repair_commands = {row["id"]: row["command"] for row in inspection["operator_repair_commands"]}
    assert "update_validation_checklist" in repair_commands
    assert "refresh_handoff_checksums" in repair_commands
    assert "regenerate_package_with_implementation_status_artifact" not in repair_commands
    repair_command_rows = {row["id"]: row for row in inspection["operator_repair_commands"]}
    assert repair_command_rows["update_validation_checklist"]["category"] == "validation_checklist"
    assert set(repair_command_rows["update_validation_checklist"]["gate_ids"]) == {
        "browser_response_security_headers",
        "browser_smoke",
        "dashboard_visibility",
        "disposable_zarr_mutation_smoke",
        "final_signoff",
        "identity_probe_verification",
        "multi_user_dry_run",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "operator_authorization_boundary",
    }
    assert repair_command_rows["refresh_handoff_checksums"]["category"] == "checksum_refresh"
    assert inspection["operator_repair_command_count"] == len(inspection["operator_repair_commands"])
    assert inspection["operator_repair_command_categories"]["validation_checklist"] == 2
    assert inspection["operator_repair_command_categories"]["checksum_refresh"] == 1
    assert inspection["operator_repair_command_gate_ids"] == [
        "browser_response_security_headers",
        "browser_smoke",
        "dashboard_visibility",
        "disposable_zarr_mutation_smoke",
        "final_signoff",
        "identity_probe_verification",
        "multi_user_dry_run",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "operator_authorization_boundary",
    ]
    assert inspection["operator_repair_command_reason_ids"] == []
    assert inspection["operator_repair_commands_requiring_checksum_refresh"] == 7
    assert inspection["labeler_links_safe_to_share"] is False
    assert inspection["shareability_status"] == "needs_operator_review"
    assert inspection["operator_action_required_before_share"] is True
    assert inspection["browser_mutation_target_contract_met"] is True
    assert inspection["browser_mutation_target_mismatch_count"] == 0
    assert inspection["browser_mutation_target_mismatch_users"] == []
    assert inspection["browser_mutation_target_contract"]["required_values"][
        "browser_mutation_browser_label_write_target"
    ] == "training_zarr"
    assert inspection["browser_mutation_target_required_values"][
        "browser_mutation_browser_writes_handoff_csv"
    ] is False
    assert inspection["browser_mutation_target_required_values"][
        "browser_mutation_browser_writes_intermediate_csv"
    ] is False
    assert inspection["direct_browser_start_contract_met"] is True
    assert inspection["direct_browser_start_mismatch_count"] == 0
    assert inspection["direct_browser_start_mismatch_users"] == []
    assert inspection["direct_browser_start_required_values"][
        "dataset_queue_direct_start_policy_present"
    ] is True
    assert inspection["direct_browser_start_required_values"][
        "dataset_queue_direct_start_post_body_expected_user_required"
    ] is True
    assert inspection["direct_browser_start_required_values"][
        "dataset_queue_direct_start_browser_label_write_target"
    ] == "training_zarr"
    assert inspection["single_owner_package_contract_met"] is True
    assert inspection["single_owner_package_mismatch_count"] == 0
    assert inspection["single_owner_package_mismatch_recording_ids"] == []
    assert inspection["single_owner_package_contract"]["multiple_labelers_per_recording_allowed"] is False
    assert inspection["labeler_route_authorization_runtime_checklist_gate_met"] is True
    assert inspection["labeler_route_authorization_runtime_checklist_mismatch_count"] == 0
    assert inspection["labeler_route_authorization_runtime_checklist_mismatch_users"] == []
    assert inspection["labeler_route_authorization_runtime_checklist_mismatches"] == []
    assert inspection["labeler_route_authorization_runtime_checklist_required_values"][
        "labeler_route_authorization_runtime_checklist_ready"
    ] is True
    assert inspection["labeler_route_authorization_runtime_checklist_required_values"][
        "labeler_route_authorization_assignment_ownership_integrity_ok"
    ] is True
    assert inspection["labeler_route_authorization_runtime_checklist_required_values"][
        "labeler_route_authorization_duplicate_active_owner_count"
    ] == 0
    assert inspection["labeler_route_authorization_runtime_checklist_required_values"][
        "labeler_route_authorization_browser_mutation_target_resolved_server_side"
    ] is True
    assert inspection["labeler_route_authorization_runtime_checklist_required_values"][
        "labeler_route_authorization_labelers_mutate_assigned_training_zarrs"
    ] is True
    assert inspection["labeler_route_authorization_runtime_checklist_required_values"][
        "labeler_route_authorization_labelers_mutate_intermediate_csvs"
    ] is False
    assert inspection["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert inspection["safe_share_ready_to_send_is_sufficient"] is False
    assert inspection["safe_share_checklist_gate_evidence_complete"] is False
    assert inspection["safe_share_launch_blocking_pending_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert inspection["safe_share_launch_blocking_unsatisfied_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert inspection["safe_share_launch_blocking_next_action_count"] == 5
    assert inspection["safe_share_launch_blocking_next_action_detail_fields"] == (
        inspection["shareability"]["safe_share_launch_blocking_next_action_detail_fields"]
    )
    assert "operator_validation_record_command_ids" in inspection[
        "safe_share_launch_blocking_next_action_detail_fields"
    ]
    assert "operator_validation_evidence_template_path" in inspection[
        "safe_share_launch_blocking_next_action_command_fields"
    ]
    assert inspection["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 5;"
    )
    assert "browser_response_security_headers=pending_operator_evidence" in inspection[
        "safe_share_next_action_summary"
    ]
    assert [
        action["gate_id"]
        for action in inspection["safe_share_launch_blocking_next_actions"]
    ] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert all(
        action["blocks_share"] is True
        for action in inspection["safe_share_launch_blocking_next_actions"]
    )
    response_security_action = next(
        action
        for action in inspection["safe_share_launch_blocking_next_actions"]
        if action["gate_id"] == "browser_response_security_headers"
    )
    assert response_security_action["operator_validation_record_command_ids"] == [
        "record_browser_response_security_evidence"
    ]
    assert response_security_action["operator_validation_apply_command_id"] == (
        "apply_operator_evidence_templates"
    )
    assert response_security_action["operator_validation_evidence_template_path"] == (
        "browser-response-security-evidence-template.json"
    )
    assert inspection["shareability_blocking_gate_ids"] == [
        "browser_response_security_headers",
        "browser_smoke",
        "dashboard_visibility",
        "disposable_zarr_mutation_smoke",
        "final_signoff",
        "identity_probe_verification",
        "multi_user_dry_run",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "operator_authorization_boundary",
    ]
    assert inspection["safe_share_external_launch_evidence_gap_action_required"] is True
    assert inspection["safe_share_external_launch_evidence_gap_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert inspection["safe_share_external_launch_evidence_gap_statuses"] == {
        "mutable_zarr_backup_confirmation": "pending_operator_evidence",
        "browser_response_security_headers": "pending_operator_evidence",
        "identity_probe_verification": "pending_operator_evidence",
        "browser_smoke": "pending_operator_evidence",
        "disposable_zarr_mutation_smoke": "pending_operator_evidence",
    }
    assert inspection[
        "safe_share_external_launch_evidence_gap_template_paths_by_gate_id"
    ] == {
        "mutable_zarr_backup_confirmation": "zarr-backup-evidence-template.json",
        "browser_response_security_headers": "browser-response-security-evidence-template.json",
        "identity_probe_verification": "identity-source-evidence-template.json",
        "browser_smoke": "browser-smoke-evidence-template.json",
        "disposable_zarr_mutation_smoke": "disposable-zarr-mutation-smoke-evidence-template.json",
    }
    assert inspection[
        "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id"
    ] == {
        "mutable_zarr_backup_confirmation": ["record_zarr_backup_evidence"],
        "browser_response_security_headers": [
            "record_browser_response_security_evidence"
        ],
        "identity_probe_verification": ["record_identity_source_evidence"],
        "browser_smoke": ["record_browser_smoke_evidence"],
        "disposable_zarr_mutation_smoke": [
            "record_disposable_zarr_mutation_smoke_evidence"
        ],
    }
    assert inspection["safe_share_external_launch_evidence_gap_todo_count"] == 5
    assert [
        todo["gate_id"]
        for todo in inspection["safe_share_external_launch_evidence_gap_todos"]
    ] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    external_gap_todos_by_gate = {
        todo["gate_id"]: todo
        for todo in inspection["safe_share_external_launch_evidence_gap_todos"]
    }
    assert external_gap_todos_by_gate["browser_response_security_headers"][
        "operator_validation_evidence_template_path"
    ] == "browser-response-security-evidence-template.json"
    assert external_gap_todos_by_gate["identity_probe_verification"][
        "operator_validation_record_command_ids"
    ] == ["record_identity_source_evidence"]
    assert "operator_review_required" in inspection["shareability_blocking_reason_ids"]
    assert "operator_repair_commands_required" in inspection["shareability_blocking_reason_ids"]
    assert "validation_gates_blocking_share" in inspection["shareability_blocking_reason_ids"]
    assert "implementation_status_artifact_incomplete" not in inspection[
        "shareability_blocking_reason_ids"
    ]
    assert inspection["shareability"]["safe_to_share"] is False
    assert inspection["shareability"]["schema"] == "palette.web_labeling_handoff_shareability.v1"
    assert inspection["shareability"]["decision_source"] == "inspect_handoff_package"
    assert inspection["shareability"]["safe_share_gate_id"] == "labeler_links_safe_to_share"
    assert inspection["shareability"]["safe_share_ready_to_send_is_sufficient"] is False
    assert inspection["shareability"]["safe_share_checklist_gate_evidence_complete"] is False
    assert inspection["shareability"]["safe_share_launch_blocking_pending_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert inspection["shareability"]["safe_share_launch_blocking_unsatisfied_gate_ids"] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert inspection["shareability"]["safe_share_launch_blocking_next_action_count"] == 5
    assert "operator_validation_record_command_ids" in inspection["shareability"][
        "safe_share_launch_blocking_next_action_detail_fields"
    ]
    assert "operator_validation_evidence_template_path" in inspection["shareability"][
        "safe_share_launch_blocking_next_action_command_fields"
    ]
    assert inspection["shareability"]["safe_share_next_action_summary"].startswith(
        "Safe-share next actions: 5;"
    )
    assert "identity_probe_verification=pending_operator_evidence" in inspection[
        "shareability"
    ]["safe_share_next_action_summary"]
    assert [
        action["gate_id"]
        for action in inspection["shareability"]["safe_share_launch_blocking_next_actions"]
    ] == [
        "mutable_zarr_backup_confirmation",
        "browser_response_security_headers",
        "identity_probe_verification",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
    ]
    assert all(
        action["operator_only"] is True
        for action in inspection["shareability"]["safe_share_launch_blocking_next_actions"]
    )
    identity_action = next(
        action
        for action in inspection["shareability"]["safe_share_launch_blocking_next_actions"]
        if action["gate_id"] == "identity_probe_verification"
    )
    assert identity_action["operator_validation_record_command_ids"] == [
        "record_identity_source_evidence"
    ]
    assert identity_action["operator_validation_apply_command_id"] == (
        "apply_operator_evidence_templates"
    )
    assert identity_action["operator_validation_evidence_template_path"] == (
        "identity-source-evidence-template.json"
    )
    assert inspection["shareability"]["status"] == "needs_operator_review"
    assert inspection["shareability"]["operator_action_required"] is True
    assert inspection["shareability"]["browser_mutation_target_contract_met"] is True
    assert inspection["shareability"]["browser_mutation_target_mismatch_count"] == 0
    assert "implementation_status_checklist_artifact_complete" in inspection["shareability"][
        "safe_to_share_requires"
    ]
    assert "implementation_status_checklist_artifact_complete_matches_required_value" in inspection["shareability"][
        "safe_to_share_requires"
    ]
    assert "browser_mutation_target_contract_met" in inspection["shareability"][
        "safe_to_share_requires"
    ]
    assert inspection["shareability"]["direct_browser_start_contract_met"] is True
    assert inspection["shareability"]["direct_browser_start_mismatch_count"] == 0
    assert "direct_browser_start_contract_met" in inspection["shareability"][
        "safe_to_share_requires"
    ]
    assert inspection["shareability"]["single_owner_package_contract_met"] is True
    assert inspection["shareability"]["single_owner_package_mismatch_count"] == 0
    assert "single_owner_package_contract_met" in inspection["shareability"][
        "safe_to_share_requires"
    ]
    assert inspection["shareability"][
        "labeler_route_authorization_runtime_checklist_gate_met"
    ] is True
    assert inspection["shareability"][
        "labeler_route_authorization_runtime_checklist_mismatch_count"
    ] == 0
    assert "labeler_route_authorization_runtime_checklist_gate_met" in inspection[
        "shareability"
    ]["safe_to_share_requires"]
    assert inspection["shareability"]["blocking_gate_ids"] == [
        "browser_response_security_headers",
        "browser_smoke",
        "dashboard_visibility",
        "disposable_zarr_mutation_smoke",
        "final_signoff",
        "identity_probe_verification",
        "multi_user_dry_run",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "operator_authorization_boundary",
    ]
    assert inspection["shareability"][
        "safe_share_external_launch_evidence_gap_gate_ids"
    ] == inspection["safe_share_external_launch_evidence_gap_gate_ids"]
    assert inspection["shareability"][
        "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id"
    ] == inspection[
        "safe_share_external_launch_evidence_gap_record_command_ids_by_gate_id"
    ]
    assert inspection["shareability"][
        "safe_share_external_launch_evidence_gap_todos"
    ] == inspection["safe_share_external_launch_evidence_gap_todos"]
    assert inspection["shareability"]["blocking_reason_ids"] == inspection[
        "shareability_blocking_reason_ids"
    ]
    assert inspection["shareability"]["repair_command_count"] == inspection[
        "operator_repair_command_count"
    ]
    assert inspection["shareability"]["repair_command_ids"] == [
        row["id"] for row in inspection["operator_repair_commands"]
    ]
    assert inspection["shareability"]["repair_command_categories_required"] == [
        "checksum_refresh",
        "operator_evidence",
        "validation_checklist",
    ]
    assert inspection["shareability"]["repair_command_gate_ids"] == [
        "browser_response_security_headers",
        "browser_smoke",
        "dashboard_visibility",
        "disposable_zarr_mutation_smoke",
        "final_signoff",
        "identity_probe_verification",
        "multi_user_dry_run",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "operator_authorization_boundary",
    ]
    assert inspection["shareability"]["repair_command_reason_ids"] == inspection[
        "operator_repair_command_reason_ids"
    ]
    assert inspection["shareability"]["repair_commands_require_checksum_refresh"] is True
    assert inspection["shareability"]["identity_personal_queue_evidence"] == {
        "status": "missing",
        "ready_count": 0,
        "missing_count": 0,
        "ready_users": [],
        "missing_users": [],
        "missing_fields_by_user": {},
        "all_users_have_personal_queue_evidence": False,
    }
    assert inspection["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert inspection["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert inspection["operator_validation_gate_flat_field_suffixes"] == [
        "status",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]
    assert inspection["shareability"]["operator_validation_gate_status_values"] == list(
        labeling_web_module.OPERATOR_VALIDATION_GATE_STATUS_VALUES
    )
    assert inspection["shareability"]["operator_validation_gate_ids"] == list(
        labeling_web_module.DEFAULT_OPERATOR_VALIDATION_GATE_IDS
    )
    assert inspection["shareability"]["operator_validation_gate_flat_field_suffixes"] == [
        "status",
        "pending",
        "missing_evidence",
        "needs_review",
        "passed",
    ]
    assert inspection["operator_validation_command_template_summary"] == inspection[
        "shareability"
    ]["operator_validation_command_template_summary"]
    assert inspection["operator_validation_command_template_summary"][
        "commands_are_operator_only"
    ] is True
    assert inspection["operator_validation_command_template_summary"][
        "commands_are_labeler_instructions"
    ] is False
    assert inspection["operator_validation_command_template_summary"][
        "labelers_must_not_run_commands"
    ] is True
    assert inspection["operator_validation_command_template_summary"][
        "launch_evidence_collection_plan_schema"
    ] == "palette.web_labeling_launch_evidence_collection_plan.v1"
    assert inspection["operator_validation_command_template_summary"][
        "launch_evidence_collection_required_final_field"
    ] == "labeler_links_safe_to_share"
    assert inspection["operator_validation_command_template_command_ids"] == inspection[
        "shareability"
    ]["operator_validation_command_template_command_ids"]
    assert "record_identity_source_evidence" in inspection[
        "operator_validation_command_template_command_ids"
    ]
    assert "record_browser_response_security_evidence" in inspection[
        "operator_validation_command_template_command_ids"
    ]
    assert inspection["operator_validation_command_template_template_backed_gate_ids"] == [
        "identity_probe_verification",
        "browser_response_security_headers",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "mutable_zarr_backup_confirmation",
    ]
    assert inspection["shareability"][
        "operator_validation_command_template_template_backed_gate_ids"
    ] == inspection["operator_validation_command_template_template_backed_gate_ids"]
    assert inspection["operator_validation_command_template_apply_required_gate_ids"] == [
        "identity_probe_verification",
        "browser_response_security_headers",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "mutable_zarr_backup_confirmation",
    ]
    assert inspection[
        "operator_validation_command_template_launch_evidence_collection_gate_ids"
    ] == [
        "identity_probe_verification",
        "operator_authorization_boundary",
        "browser_response_security_headers",
        "dashboard_visibility",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "multi_user_dry_run",
        "final_signoff",
    ]
    assert "record_identity_source_evidence" in inspection[
        "operator_validation_command_template_launch_evidence_collection_record_command_ids"
    ]
    assert inspection[
        "operator_validation_command_template_launch_evidence_collection_required_final_field"
    ] == "labeler_links_safe_to_share"
    assert inspection[
        "operator_validation_command_template_evidence_template_fields_by_gate_id"
    ]["browser_response_security_headers"] == (
        "browser_response_security_evidence_template"
    )
    assert inspection[
        "operator_validation_command_template_evidence_template_paths_by_gate_id"
    ]["identity_probe_verification"] == "identity-source-evidence-template.json"
    assert inspection["operator_validation_external_evidence_required"] is False
    assert inspection["operator_validation_external_evidence_required_gate_ids"] == []
    assert inspection["shareability"][
        "operator_validation_external_evidence_required_gate_ids"
    ] == inspection["operator_validation_external_evidence_required_gate_ids"]
    assert inspection[
        "operator_validation_external_evidence_template_paths_by_gate_id"
    ] == {}
    assert inspection["operator_validation_checklist_only_required_gate_ids"] == []
    assert inspection["operator_validation_command_template_gate_ids"] == [
        "identity_probe_verification",
        "operator_authorization_boundary",
        "browser_response_security_headers",
        "dashboard_visibility",
        "browser_smoke",
        "disposable_zarr_mutation_smoke",
        "mutable_zarr_backup_confirmation",
        "one_labeler_dry_run",
        "multi_user_dry_run",
        "final_signoff",
    ]
    assert inspection["operator_validation_command_template_missing_command_gate_ids"] == [
        "operator_authorization_boundary",
        "dashboard_visibility",
        "one_labeler_dry_run",
        "multi_user_dry_run",
        "final_signoff",
    ]
    assert inspection["kind"] == "batch_zip"
    assert inspection["counts"]["users"] == 2
    assert inspection["counts"]["signed_links"] == 2
    assert inspection["counts"]["waiting_datasets"] == 2
    assert inspection["counts"]["dataset_open_tasks"] == 2
    assert inspection["counts"]["dataset_queue_states"] == {"has_open_dataset_work": 2}
    assert inspection["counts"]["dataset_queue_blocked_start_users"] == []
    assert inspection["validation_log"]["present"] is True
    assert "handoffs/validation-log-template.md" in inspection["validation_log"]["matched_paths"]
    assert "handoffs/alice/validation-log-template.md" in inspection["validation_log"]["related_paths"]
    assert inspection["validation_checklist"]["present"] is True
    assert inspection["validation_checklist"]["ready_for_operator_validation"] is True
    assert inspection["validation_checklist"]["operator_validation_visibility_policy"][
        "operator_only_fields"
    ] == ["operator_validation_checklist_path"]
    assert inspection["validation_checklist"]["safe_share_gate"]["schema"] == (
        "palette.web_labeling_safe_share_gate.v1"
    )
    assert inspection["validation_checklist"]["safe_share_gate_id"] == (
        "labeler_links_safe_to_share"
    )
    assert inspection["validation_checklist"][
        "safe_share_ready_to_send_is_sufficient"
    ] is False
    assert inspection["validation_checklist"]["safe_share_required_inspection_field"] == (
        "labeler_links_safe_to_share"
    )
    assert inspection["validation_checklist"]["labeler_landing_page_path"] == "/"
    assert inspection["validation_checklist"]["labeler_landing_url"] == "https://labeling.example.org"
    assert "handoffs/validation-checklist.json" in inspection["validation_checklist"]["matched_paths"]
    assert "handoffs/alice/validation-checklist.json" in inspection["validation_checklist"]["related_paths"]
    assert inspection["validation_checklist"]["dataset_queue_page_path"] == "/datasets"
    assert inspection["validation_checklist"]["dataset_queue_url"] == "https://labeling.example.org/datasets"
    assert inspection["personal_dataset_queue_page_path"] == "/my-datasets"
    assert inspection["personal_dataset_queue_url"] == "https://labeling.example.org/my-datasets"
    assert inspection["personal_work_page_path"] == "/my-work"
    assert inspection["personal_work_url"] == "https://labeling.example.org/my-work"
    assert inspection["personalized_labeler_entrypoint"] == "personal_datasets_waiting_queue"
    assert inspection["personalized_labeler_entry_url_count"] == 2
    assert inspection["all_handoffs_have_personalized_entry_url"] is True
    assert inspection["labeler_entrypoint_summary"]["personalized_labeler_entry_url_by_user"] == {
        "alice": "https://labeling.example.org/my-datasets?expected_user=alice",
        "bob": "https://labeling.example.org/my-datasets?expected_user=bob",
    }
    assert inspection["shareability"]["all_handoffs_have_personalized_entry_url"] is True
    assert "identity_probe_verification" in inspection["validation_checklist"]["pending_gate_ids"]
    assert "browser_response_security_headers" in inspection["validation_checklist"]["pending_gate_ids"]
    assert "identity_probe_verification" in inspection["validation_checklist"]["operator_evidence_gate_ids"]
    assert "browser_smoke" in inspection["validation_checklist"]["operator_evidence_gate_ids"]
    assert "static_readiness" in inspection["validation_checklist"]["generated_contract_gate_ids"]
    assert "identity_probe_verification" in inspection["validation_checklist"][
        "operator_evidence_pending_gate_ids"
    ]
    assert "browser_response_security_headers" in inspection["validation_checklist"][
        "operator_evidence_pending_gate_ids"
    ]
    assert inspection["validation_checklist"]["operator_evidence_needs_review_gate_ids"] == []
    assert inspection["validation_checklist"]["generated_contract_failed_gate_ids"] == []
    assert "identity_probe_verification" in inspection["validation_checklist"]["required_pending_gate_ids"]
    assert "browser_response_security_headers" in inspection["validation_checklist"]["required_pending_gate_ids"]
    assert inspection["validation_checklist"]["evidence_recorded_gate_ids"] == []
    assert "identity_probe_verification" in inspection["validation_checklist"]["required_missing_evidence_gate_ids"]
    assert "browser_response_security_headers" in inspection["validation_checklist"]["required_missing_evidence_gate_ids"]
    assert "static_readiness" not in inspection["validation_checklist"]["required_missing_evidence_gate_ids"]
    assert inspection["validation_checklist"]["evidence_recorded_gate_count"] == 0
    assert inspection["validation_checklist"]["needs_review_gate_ids"] == []
    assert inspection["earliest_links_expire_at_utc"]
    assert {handoff["user"] for handoff in inspection["handoffs"]} == {"alice", "bob"}
    assert {handoff["expected_user_labeler_landing_url"] for handoff in inspection["handoffs"]} == {
        "https://labeling.example.org?expected_user=alice",
        "https://labeling.example.org?expected_user=bob",
    }
    assert {handoff["expected_user_dashboard_url"] for handoff in inspection["handoffs"]} == {
        "https://labeling.example.org/work?expected_user=alice",
        "https://labeling.example.org/work?expected_user=bob",
    }
    assert {handoff["expected_user_identity_probe_url"] for handoff in inspection["handoffs"]} == {
        "https://labeling.example.org/identity?expected_user=alice",
        "https://labeling.example.org/identity?expected_user=bob",
    }
    assert {handoff["dataset_queue_summary"]["waiting_dataset_count"] for handoff in inspection["handoffs"]} == {1}
    assert {handoff["dataset_queue_state_code"] for handoff in inspection["handoffs"]} == {"has_open_dataset_work"}
    assert {handoff["dataset_queue_blocks_labeler_start"] for handoff in inspection["handoffs"]} == {False}
    assert {handoff["dataset_queue_start_ready"] for handoff in inspection["handoffs"]} == {True}
    assert {handoff["dataset_queue_start_status"] for handoff in inspection["handoffs"]} == {"passed"}
    assert {handoff["dataset_queue_start_operator_action"] for handoff in inspection["handoffs"]} == {""}
    assert archived_inspection["counts"] == inspection["counts"]

    checklist_path = output_dir / "validation-checklist.json"
    original_checklist_text = checklist_path.read_text()
    stale_checklist = json.loads(original_checklist_text)
    stale_checklist.pop("implementation_status_artifact")
    checklist_path.write_text(json.dumps(stale_checklist))
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )
    stale_implementation_status_inspection = json.loads(capsys.readouterr().out)
    checklist_path.write_text(original_checklist_text)
    stale_repair_commands = {
        row["id"]: row for row in stale_implementation_status_inspection["operator_repair_commands"]
    }
    assert rc == 2
    assert stale_implementation_status_inspection["labeler_links_safe_to_share"] is False
    assert "implementation_status_artifact_incomplete" in stale_implementation_status_inspection[
        "failure_reasons"
    ]
    assert "implementation_status_artifact_incomplete" in stale_implementation_status_inspection[
        "shareability_blocking_reason_ids"
    ]
    assert (
        "implementation_status_checklist_artifact_complete_required_value_mismatch"
        in stale_implementation_status_inspection["shareability_blocking_reason_ids"]
    )
    stale_failure_actions = "\n".join(
        stale_implementation_status_inspection["failure_actions"]
    )
    assert "Implementation status artifact missing required fields:" in stale_failure_actions
    assert "safe_share_gate" in stale_failure_actions
    assert "current implementation_status_artifact contract" in stale_failure_actions
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_present"
    ] is False
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_complete"
    ] is False
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_complete_required_value"
    ] is True
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_complete_matches_required_value"
    ] is False
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_gate"
    ]["observed_value"] is False
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_gate"
    ]["required_value"] is True
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_gate"
    ]["matches_required_value"] is False
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_gate"
    ]["fail_closed_reason"] == "implementation_status_artifact_incomplete"
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_gate"
    ]["required_value_mismatch_blocking_reason"] == (
        "implementation_status_checklist_artifact_complete_required_value_mismatch"
    )
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_missing_fields"
    ] == list(labeling_web_module._IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS)
    assert stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_missing_field_count"
    ] == len(labeling_web_module._IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS)
    assert stale_implementation_status_inspection["shareability"][
        "implementation_status_checklist_artifact_complete"
    ] is False
    assert stale_implementation_status_inspection["shareability"][
        "implementation_status_checklist_artifact_complete_required_value"
    ] is True
    assert stale_implementation_status_inspection["shareability"][
        "implementation_status_checklist_artifact_complete_matches_required_value"
    ] is False
    assert stale_implementation_status_inspection["shareability"][
        "implementation_status_checklist_artifact_gate"
    ] == stale_implementation_status_inspection[
        "implementation_status_checklist_artifact_gate"
    ]
    assert (
        "implementation_status_checklist_artifact_complete_required_value_mismatch"
        in stale_implementation_status_inspection["shareability"]["blocking_reason_ids"]
    )
    assert "regenerate_package_with_implementation_status_artifact" in stale_repair_commands
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "category"
    ] == "handoff_regeneration"
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "reason_ids"
    ] == ["implementation_status_artifact_incomplete"]
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "missing_fields"
    ] == list(labeling_web_module._IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS)
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "missing_field_count"
    ] == len(labeling_web_module._IMPLEMENTATION_STATUS_ARTIFACT_REQUIRED_FIELDS)
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "repair_mode"
    ] == "regenerate_package"
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "artifact_contract"
    ] == "implementation_status_artifact"
    assert stale_repair_commands["regenerate_package_with_implementation_status_artifact"][
        "safe_share_blocker"
    ] == "implementation_status_checklist_artifact_complete_required_value_mismatch"

    corrupted_manifest_path = output_dir / "alice" / "manifest.json"
    corrupted_manifest = json.loads(corrupted_manifest_path.read_text())
    corrupted_manifest["browser_mutation_write_policy"][
        "browser_label_write_target"
    ] = "handoff_csv"
    corrupted_manifest_path.write_text(json.dumps(corrupted_manifest))
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )
    corrupted_inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert corrupted_inspection["labeler_links_safe_to_share"] is False
    assert "browser_mutation_target_contract_mismatch" in corrupted_inspection[
        "failure_reasons"
    ]
    assert "browser_mutation_target_contract_mismatch" in corrupted_inspection[
        "shareability_blocking_reason_ids"
    ]
    assert corrupted_inspection["browser_mutation_target_contract_met"] is False
    assert corrupted_inspection["browser_mutation_target_mismatch_users"] == ["alice"]
    browser_label_write_target_mismatch = next(
        mismatch
        for mismatch in corrupted_inspection["browser_mutation_target_mismatches"]
        if mismatch["field"] == "browser_mutation_browser_label_write_target"
    )
    assert browser_label_write_target_mismatch["expected"] == "training_zarr"
    assert browser_label_write_target_mismatch["actual"] == "handoff_csv"
    assert corrupted_inspection["shareability"][
        "browser_mutation_target_contract_met"
    ] is False
    corrupted_repair_commands = {
        row["id"]: row for row in corrupted_inspection["operator_repair_commands"]
    }
    corrupted_shareability_repair_commands = {
        row["id"]: row for row in corrupted_inspection["shareability"]["repair_commands"]
    }
    assert corrupted_shareability_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ] == corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]
    assert (
        corrupted_repair_commands[
            "regenerate_handoffs_with_browser_mutation_target_contract"
        ]["category"]
        == "handoff_regeneration"
    )
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["reason_ids"] == ["browser_mutation_target_contract_mismatch"]
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["contract"] == "browser_mutation_target_contract"
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["repair_mode"] == "regenerate_handoff_package"
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["safe_share_blocker"] == "browser_mutation_target_contract_mismatch"
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["required_values"]["browser_mutation_browser_label_write_target"] == (
        "training_zarr"
    )
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["required_values"]["browser_mutation_browser_writes_handoff_csv"] is False
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["mismatch_count"] == 2
    assert corrupted_repair_commands[
        "regenerate_handoffs_with_browser_mutation_target_contract"
    ]["mismatch_users"] == ["alice"]
    repair_browser_label_write_target_mismatch = next(
        mismatch
        for mismatch in corrupted_repair_commands[
            "regenerate_handoffs_with_browser_mutation_target_contract"
        ]["mismatches"]
        if mismatch["field"] == "browser_mutation_browser_label_write_target"
    )
    assert repair_browser_label_write_target_mismatch["actual"] == "handoff_csv"
    assert (
        "regenerate_handoffs_with_browser_mutation_target_contract"
        in corrupted_inspection["shareability"]["repair_command_ids"]
    )
    assert "handoff_regeneration" in corrupted_inspection["shareability"][
        "repair_command_categories_required"
    ]

    direct_start_corrupted_manifest = json.loads(corrupted_manifest_path.read_text())
    direct_start_corrupted_manifest["browser_mutation_write_policy"][
        "browser_label_write_target"
    ] = "training_zarr"
    direct_start_corrupted_manifest["dataset_queue_direct_start_policy"][
        "browser_label_write_target"
    ] = "handoff_csv"
    corrupted_manifest_path.write_text(json.dumps(direct_start_corrupted_manifest))
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )
    direct_start_corrupted_inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert direct_start_corrupted_inspection["labeler_links_safe_to_share"] is False
    assert "direct_browser_start_contract_mismatch" in direct_start_corrupted_inspection[
        "failure_reasons"
    ]
    assert "direct_browser_start_contract_mismatch" in direct_start_corrupted_inspection[
        "shareability_blocking_reason_ids"
    ]
    assert direct_start_corrupted_inspection["direct_browser_start_contract_met"] is False
    assert direct_start_corrupted_inspection["direct_browser_start_mismatch_users"] == ["alice"]
    assert direct_start_corrupted_inspection["direct_browser_start_mismatches"][0][
        "field"
    ] == "dataset_queue_direct_start_browser_label_write_target"
    assert direct_start_corrupted_inspection["direct_browser_start_mismatches"][0][
        "expected"
    ] == "training_zarr"
    assert direct_start_corrupted_inspection["direct_browser_start_mismatches"][0][
        "actual"
    ] == "handoff_csv"
    assert direct_start_corrupted_inspection["shareability"][
        "direct_browser_start_contract_met"
    ] is False
    direct_start_repair_commands = {
        row["id"]: row
        for row in direct_start_corrupted_inspection["operator_repair_commands"]
    }
    direct_start_shareability_repair_commands = {
        row["id"]: row
        for row in direct_start_corrupted_inspection["shareability"]["repair_commands"]
    }
    assert direct_start_shareability_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ] == direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]
    assert (
        direct_start_repair_commands[
            "regenerate_handoffs_with_direct_browser_start_contract"
        ]["category"]
        == "handoff_regeneration"
    )
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["reason_ids"] == ["direct_browser_start_contract_mismatch"]
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["contract"] == "direct_browser_start_contract"
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["repair_mode"] == "regenerate_handoff_package"
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["safe_share_blocker"] == "direct_browser_start_contract_mismatch"
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["required_values"]["dataset_queue_direct_start_browser_label_write_target"] == (
        "training_zarr"
    )
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["mismatch_count"] == 1
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["mismatch_users"] == ["alice"]
    assert direct_start_repair_commands[
        "regenerate_handoffs_with_direct_browser_start_contract"
    ]["mismatches"][0]["actual"] == "handoff_csv"
    assert (
        "regenerate_handoffs_with_direct_browser_start_contract"
        in direct_start_corrupted_inspection["shareability"]["repair_command_ids"]
    )

    single_owner_manifest_path = output_dir / "bob" / "manifest.json"
    single_owner_manifest = json.loads(single_owner_manifest_path.read_text())
    single_owner_manifest["assignment_snapshot"]["assignments"].append(
        {
            "recording_id": "rec-a",
            "assignee_user": "bob",
            "status": "active",
        }
    )
    single_owner_manifest_path.write_text(json.dumps(single_owner_manifest))
    direct_start_repaired_manifest = json.loads(corrupted_manifest_path.read_text())
    direct_start_repaired_manifest["dataset_queue_direct_start_policy"][
        "browser_label_write_target"
    ] = "training_zarr"
    corrupted_manifest_path.write_text(json.dumps(direct_start_repaired_manifest))
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )
    single_owner_corrupted_inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert single_owner_corrupted_inspection["labeler_links_safe_to_share"] is False
    assert "single_owner_package_contract_mismatch" in single_owner_corrupted_inspection[
        "failure_reasons"
    ]
    assert "single_owner_package_contract_mismatch" in single_owner_corrupted_inspection[
        "shareability_blocking_reason_ids"
    ]
    assert single_owner_corrupted_inspection["single_owner_package_contract_met"] is False
    assert single_owner_corrupted_inspection["single_owner_package_mismatch_recording_ids"] == [
        "rec-a"
    ]
    assert single_owner_corrupted_inspection[
        "single_owner_package_duplicate_owners_by_recording"
    ] == {"rec-a": ["alice", "bob"]}
    assert single_owner_corrupted_inspection["shareability"][
        "single_owner_package_contract_met"
    ] is False
    single_owner_repair_commands = {
        row["id"]: row
        for row in single_owner_corrupted_inspection["operator_repair_commands"]
    }
    single_owner_shareability_repair_commands = {
        row["id"]: row
        for row in single_owner_corrupted_inspection["shareability"]["repair_commands"]
    }
    assert single_owner_shareability_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ] == single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]
    assert (
        single_owner_repair_commands[
            "regenerate_handoffs_with_single_owner_package_contract"
        ]["category"]
        == "handoff_regeneration"
    )
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["reason_ids"] == ["single_owner_package_contract_mismatch"]
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["contract"] == "single_owner_package_contract"
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["repair_mode"] == "regenerate_handoff_package"
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["safe_share_blocker"] == "single_owner_package_contract_mismatch"
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["mismatch_count"] == 1
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["mismatch_recording_ids"] == ["rec-a"]
    assert single_owner_repair_commands[
        "regenerate_handoffs_with_single_owner_package_contract"
    ]["duplicate_owners_by_recording"] == {"rec-a": ["alice", "bob"]}
    assert (
        "regenerate_handoffs_with_single_owner_package_contract"
        in single_owner_corrupted_inspection["shareability"]["repair_command_ids"]
    )

    single_owner_repaired_manifest = json.loads(single_owner_manifest_path.read_text())
    single_owner_repaired_manifest["assignment_snapshot"]["assignments"] = [
        assignment
        for assignment in single_owner_repaired_manifest["assignment_snapshot"][
            "assignments"
        ]
        if not (
            assignment.get("recording_id") == "rec-a"
            and assignment.get("assignee_user") == "bob"
        )
    ]
    single_owner_manifest_path.write_text(json.dumps(single_owner_repaired_manifest))
    route_checklist_corrupted_manifest = json.loads(corrupted_manifest_path.read_text())
    route_checklist_corrupted_manifest["labeler_route_authorization_checklist"][
        "single_owner_store_proof_ready"
    ] = False
    corrupted_manifest_path.write_text(json.dumps(route_checklist_corrupted_manifest))
    rc = labeling_work.main(
        [
            "--store",
            str(store_path),
            "inspect-handoff",
            "--path",
            str(output_dir),
        ]
    )
    route_checklist_corrupted_inspection = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert route_checklist_corrupted_inspection["labeler_links_safe_to_share"] is False
    assert (
        "labeler_route_authorization_runtime_checklist_mismatch"
        in route_checklist_corrupted_inspection["failure_reasons"]
    )
    assert (
        "labeler_route_authorization_runtime_checklist_mismatch"
        in route_checklist_corrupted_inspection["shareability_blocking_reason_ids"]
    )
    assert (
        route_checklist_corrupted_inspection[
            "labeler_route_authorization_runtime_checklist_gate_met"
        ]
        is False
    )
    assert route_checklist_corrupted_inspection[
        "labeler_route_authorization_runtime_checklist_mismatch_users"
    ] == ["alice"]
    assert route_checklist_corrupted_inspection[
        "labeler_route_authorization_runtime_checklist_mismatches"
    ][0]["field"] == "labeler_route_authorization_single_owner_store_proof_ready"
    assert route_checklist_corrupted_inspection[
        "labeler_route_authorization_runtime_checklist_mismatches"
    ][0]["expected"] is True
    assert route_checklist_corrupted_inspection[
        "labeler_route_authorization_runtime_checklist_mismatches"
    ][0]["actual"] is False
    assert route_checklist_corrupted_inspection["shareability"][
        "labeler_route_authorization_runtime_checklist_gate_met"
    ] is False
    assert route_checklist_corrupted_inspection["shareability_contract"][
        "labeler_route_authorization_runtime_checklist_gate_met"
    ] is False
    route_checklist_repair_commands = {
        row["id"]: row
        for row in route_checklist_corrupted_inspection["operator_repair_commands"]
    }
    assert (
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
        in route_checklist_repair_commands
    )
    assert route_checklist_repair_commands[
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
    ]["reason_ids"] == ["labeler_route_authorization_runtime_checklist_mismatch"]
    assert route_checklist_repair_commands[
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
    ]["contract"] == "labeler_route_authorization_runtime_checklist_gate"
    assert route_checklist_repair_commands[
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
    ]["safe_share_blocker"] == (
        "labeler_route_authorization_runtime_checklist_mismatch"
    )
    assert route_checklist_repair_commands[
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
    ]["required_values"][
        "labeler_route_authorization_single_owner_store_proof_ready"
    ] is True
    assert route_checklist_repair_commands[
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
    ]["mismatch_count"] == 1
    assert route_checklist_repair_commands[
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
    ]["mismatch_users"] == ["alice"]
    assert (
        "regenerate_handoffs_with_labeler_route_authorization_runtime_checklist"
        in route_checklist_corrupted_inspection["shareability"]["repair_command_ids"]
    )

    with pytest.raises(FileExistsError):
        labeling_work.main(
            [
                "--store",
                str(store_path),
                "export-user-handoffs",
                "--link-secret",
                "test-secret",
                "--output-dir",
                str(output_dir),
            ]
        )
