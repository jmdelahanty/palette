"""Operator evidence recording helpers for web-labeling launch checks."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _expected_user_query_value_from_url,
)
from .web_policy import LABELING_HOME_PATH, PERSONAL_WORK_PATH


def _browser_smoke_personalized_route_contract() -> dict[str, object]:
    return {
        "schema": "palette.web_labeling_browser_smoke_route_contract.v1",
        "preferred_queue_entrypoint": "personal_datasets_waiting_queue",
        "preferred_queue_path": PERSONAL_DATASET_QUEUE_PATH,
        "human_readable_queue_alias_path": LABELING_HOME_PATH,
        "human_readable_queue_alias_url_field": "labeling_home_url",
        "fallback_dashboard_entrypoint": "personal_work_dashboard",
        "fallback_dashboard_path": PERSONAL_WORK_PATH,
        "canonical_queue_fallback_path": DATASET_QUEUE_PATH,
        "canonical_dashboard_fallback_path": DASHBOARD_PATH,
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


def _operator_evidence_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() == "true"


def _identity_source_personal_queue_status(row: Mapping[str, object]) -> dict[str, object]:
    expected_dataset_queue_url = str(row.get("expected_user_dataset_queue_url") or "").strip()
    expected_personal_dataset_queue_url = str(
        row.get("expected_user_personal_dataset_queue_url") or ""
    ).strip()
    preferred_labeler_entry_url = str(row.get("preferred_labeler_entry_url") or "").strip()
    personalized_labeler_entry_url = str(row.get("personalized_labeler_entry_url") or "").strip()
    preferred_matches_dataset_queue = _operator_evidence_truthy(
        row.get("preferred_labeler_entry_url_matches_dataset_queue")
    ) and bool(
        preferred_labeler_entry_url
        and preferred_labeler_entry_url
        in {
            expected_dataset_queue_url,
            expected_personal_dataset_queue_url,
        }
    )
    preferred_matches_personal_queue = _operator_evidence_truthy(
        row.get("preferred_labeler_entry_url_matches_personal_dataset_queue")
    ) and bool(
        expected_personal_dataset_queue_url
        and preferred_labeler_entry_url == expected_personal_dataset_queue_url
    )
    personalized_matches_personal_queue = _operator_evidence_truthy(
        row.get("personalized_labeler_entry_url_matches_personal_dataset_queue")
    ) and bool(
        expected_personal_dataset_queue_url
        and personalized_labeler_entry_url == expected_personal_dataset_queue_url
    )
    missing_fields = []
    if not expected_personal_dataset_queue_url:
        missing_fields.append("expected_user_personal_dataset_queue_url")
    if not preferred_labeler_entry_url:
        missing_fields.append("preferred_labeler_entry_url")
    if not personalized_labeler_entry_url:
        missing_fields.append("personalized_labeler_entry_url")
    if not preferred_matches_dataset_queue:
        missing_fields.append("preferred_labeler_entry_url_matches_dataset_queue")
    if not preferred_matches_personal_queue:
        missing_fields.append("preferred_labeler_entry_url_matches_personal_dataset_queue")
    if not personalized_matches_personal_queue:
        missing_fields.append("personalized_labeler_entry_url_matches_personal_dataset_queue")
    return {
        "expected_user_dataset_queue_url": expected_dataset_queue_url,
        "expected_user_personal_dataset_queue_url": expected_personal_dataset_queue_url,
        "preferred_labeler_entry_url": preferred_labeler_entry_url,
        "personalized_labeler_entry_url": personalized_labeler_entry_url,
        "preferred_labeler_entry_url_matches_dataset_queue": preferred_matches_dataset_queue,
        "preferred_labeler_entry_url_matches_personal_dataset_queue": preferred_matches_personal_queue,
        "personalized_labeler_entry_url_matches_personal_dataset_queue": personalized_matches_personal_queue,
        "ready": not missing_fields,
        "missing_fields": missing_fields,
    }


def _identity_source_row_approved(row: Mapping[str, object]) -> bool:
    expected_user = str(row.get("expected_user") or "").strip()
    resolved_user = str(row.get("resolved_user") or "").strip()
    return bool(
        row.get("identity_matches_expected_user")
        and expected_user
        and resolved_user == expected_user
        and str(row.get("operator") or "").strip()
        and str(row.get("operator_approved_at_utc") or "").strip()
        and _identity_source_personal_queue_status(row)["ready"]
    )


def _record_identity_source_evidence(
    *,
    evidence_path: Path,
    expected_user: str,
    resolved_user: str,
    operator: str,
    authenticated_session_context: str | None = None,
    notes: str | None = None,
    output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("Identity source evidence template must be a JSON object.")
    destination_path = output or evidence_path
    if output is not None and destination_path.exists() and destination_path != evidence_path and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination_path}")
    users = evidence_payload.get("users") if isinstance(evidence_payload.get("users"), list) else []
    personalized_route_smoke_contract = (
        dict(evidence_payload.get("personalized_route_smoke_contract"))
        if isinstance(evidence_payload.get("personalized_route_smoke_contract"), Mapping)
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
            "before recording approval so personalized /my-datasets, /labeling, and /my-work route "
            "contract metadata is present."
        )
    )
    expected_user_value = str(expected_user or "").strip()
    resolved_user_value = str(resolved_user or "").strip()
    now = datetime.now(timezone.utc).isoformat()
    matched_row: dict[str, object] | None = None
    for row in users:
        if isinstance(row, dict) and str(row.get("expected_user") or "").strip() == expected_user_value:
            matched_row = row
            break
    errors: list[dict[str, object]] = []
    identity_matches = bool(expected_user_value and resolved_user_value and expected_user_value == resolved_user_value)
    matched_row_link_status: dict[str, object] = {}
    if matched_row is None:
        errors.append(
            {
                "expected_user": expected_user_value,
                "error": "expected_user_missing",
                "details": "The evidence template has no row for this expected user.",
            }
        )
    else:
        expected_dataset_queue_url = str(matched_row.get("expected_user_dataset_queue_url") or "").strip()
        expected_personal_dataset_queue_url = str(
            matched_row.get("expected_user_personal_dataset_queue_url") or ""
        ).strip()
        preferred_labeler_entry_url = str(matched_row.get("preferred_labeler_entry_url") or "").strip()
        personalized_labeler_entry_url = str(
            matched_row.get("personalized_labeler_entry_url") or ""
        ).strip()
        preferred_matches_dataset_queue = bool(
            preferred_labeler_entry_url
            and preferred_labeler_entry_url
            in {
                expected_dataset_queue_url,
                expected_personal_dataset_queue_url,
            }
        )
        preferred_matches_personal_queue = bool(
            expected_personal_dataset_queue_url
            and preferred_labeler_entry_url == expected_personal_dataset_queue_url
        )
        personalized_matches_personal_queue = bool(
            expected_personal_dataset_queue_url
            and personalized_labeler_entry_url == expected_personal_dataset_queue_url
        )
        matched_row.update(
            {
                "resolved_user": resolved_user_value,
                "identity_matches_expected_user": identity_matches,
                "preferred_labeler_entry_url_matches_dataset_queue": preferred_matches_dataset_queue,
                "preferred_labeler_entry_url_matches_personal_dataset_queue": preferred_matches_personal_queue,
                "personalized_labeler_entry_url_matches_personal_dataset_queue": personalized_matches_personal_queue,
                "captured_at_utc": now,
                "authenticated_session_context": str(authenticated_session_context or ""),
                "operator": operator,
            }
        )
        matched_row_link_status = _identity_source_personal_queue_status(matched_row)
        matched_row["operator_approved_at_utc"] = (
            now if identity_matches and bool(matched_row_link_status.get("ready")) else ""
        )
        if notes is not None:
            matched_row["notes"] = str(notes)
        if not identity_matches:
            errors.append(
                {
                    "expected_user": expected_user_value,
                    "resolved_user": resolved_user_value,
                    "error": "identity_mismatch",
                    "details": "Resolved user must exactly match expected user before this identity evidence can be approved.",
                }
            )
        elif not bool(matched_row_link_status.get("ready")):
            errors.append(
                {
                    "expected_user": expected_user_value,
                    "resolved_user": resolved_user_value,
                    "error": "personal_dataset_queue_link_evidence_incomplete",
                    "details": "Identity evidence must prove Start here uses the guarded /my-datasets?expected_user=<user> queue URL before approval.",
                    "missing_fields": matched_row_link_status.get("missing_fields", []),
                }
            )
    approved_count = sum(
        1
        for row in users
        if isinstance(row, Mapping) and _identity_source_row_approved(row)
    )
    user_count = len([row for row in users if isinstance(row, Mapping)])
    evidence_payload["updated_at_utc"] = now
    evidence_payload["updated_by"] = operator
    evidence_payload["counts"] = {
        **(
            evidence_payload.get("counts")
            if isinstance(evidence_payload.get("counts"), Mapping)
            else {}
        ),
        "users": user_count,
        "pending_operator_confirmation": max(user_count - approved_count, 0),
        "operator_approved": approved_count,
    }
    if matched_row is not None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "ok": identity_matches and not errors,
        "schema": "palette.web_labeling_identity_source_evidence_update_report.v1",
        "updated_at_utc": now,
        "operator": operator,
        "evidence_path": str(evidence_path),
        "output_path": str(destination_path),
        "expected_user": expected_user_value,
        "resolved_user": resolved_user_value,
        "identity_matches_expected_user": identity_matches,
        "personal_queue_evidence_ready": bool(matched_row_link_status.get("ready")),
        "personal_queue_evidence_missing_fields": matched_row_link_status.get("missing_fields", []),
        "preferred_labeler_entry_url_matches_dataset_queue": bool(
            matched_row_link_status.get("preferred_labeler_entry_url_matches_dataset_queue")
        ),
        "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
            matched_row_link_status.get("preferred_labeler_entry_url_matches_personal_dataset_queue")
        ),
        "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
            matched_row_link_status.get("personalized_labeler_entry_url_matches_personal_dataset_queue")
        ),
        "error_count": len(errors),
        "errors": errors,
        "counts": evidence_payload["counts"],
    }


BROWSER_RESPONSE_SECURITY_HEADER_CHECK_KEYS: dict[str, str] = {
    "cache-control": "cache_control_preserved",
    "pragma": "pragma_preserved",
    "expires": "expires_preserved",
    "x-frame-options": "x_frame_options_preserved",
    "x-content-type-options": "x_content_type_options_preserved",
    "referrer-policy": "referrer_policy_preserved",
    "content-security-policy": "content_security_policy_preserved",
    "permissions-policy": "permissions_policy_preserved",
}


def _parse_header_evidence_values(values: Sequence[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        text = str(value or "")
        separator = "=" if "=" in text else ":"
        if separator not in text:
            raise ValueError(f"Header evidence must use NAME=VALUE or NAME:VALUE format: {text}")
        name, header_value = text.split(separator, 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Header evidence name is empty: {text}")
        headers[name] = header_value.strip()
    return headers



def _record_browser_response_security_evidence(
    *,
    evidence_path: Path,
    headers: Mapping[str, str],
    operator: str,
    capture_url: str | None = None,
    authenticated_test_user: str | None = None,
    capture_note: str | None = None,
    proxy_or_deployment: str | None = None,
    notes: str | None = None,
    output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("Browser response security evidence template must be a JSON object.")
    destination_path = output or evidence_path
    if output is not None and destination_path.exists() and destination_path != evidence_path and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination_path}")
    expected_headers = (
        evidence_payload.get("expected_headers")
        if isinstance(evidence_payload.get("expected_headers"), Mapping)
        else {}
    )
    required_capture_contract = (
        dict(evidence_payload.get("required_capture_contract"))
        if isinstance(evidence_payload.get("required_capture_contract"), Mapping)
        else {}
    )
    captured_headers = (
        dict(evidence_payload.get("captured_headers"))
        if isinstance(evidence_payload.get("captured_headers"), Mapping)
        else {}
    )
    supplied_by_lower = {str(key).strip().lower(): str(value) for key, value in headers.items()}
    errors: list[dict[str, object]] = []
    matched_header_count = 0
    checks = (
        dict(evidence_payload.get("checks"))
        if isinstance(evidence_payload.get("checks"), Mapping)
        else {}
    )
    for expected_name, expected_value in expected_headers.items():
        header_name = str(expected_name)
        expected_text = str(expected_value)
        actual_text = str(
            supplied_by_lower.get(
                header_name.lower(),
                captured_headers.get(header_name, ""),
            )
            or ""
        ).strip()
        captured_headers[header_name] = actual_text
        preserved = bool(actual_text) and actual_text == expected_text
        check_key = BROWSER_RESPONSE_SECURITY_HEADER_CHECK_KEYS.get(header_name.lower())
        if check_key:
            checks[check_key] = preserved
        if preserved:
            matched_header_count += 1
        else:
            errors.append(
                {
                    "header": header_name,
                    "error": "header_missing_or_mismatched",
                    "expected": expected_text,
                    "captured": actual_text,
                }
            )
    all_headers_preserved = bool(expected_headers) and matched_header_count == len(expected_headers)
    checks["proxy_strips_or_weakens_no_headers"] = all_headers_preserved
    now = datetime.now(timezone.utc).isoformat()
    capture = (
        dict(evidence_payload.get("capture"))
        if isinstance(evidence_payload.get("capture"), Mapping)
        else {}
    )
    capture.update(
        {
            "url": str(capture_url or capture.get("url") or ""),
            "authenticated_test_user": str(authenticated_test_user or capture.get("authenticated_test_user") or ""),
            "captured_at_utc": now,
            "capture_command_or_browser_note": str(capture_note or capture.get("capture_command_or_browser_note") or ""),
            "proxy_or_deployment": str(proxy_or_deployment or capture.get("proxy_or_deployment") or ""),
        }
    )
    def _response_security_capture_path(value: str) -> str:
        text = str(value or "").split("?", 1)[0].strip()
        if "://" not in text:
            return text or "/"
        without_scheme = text.split("://", 1)[1]
        slash_index = without_scheme.find("/")
        if slash_index < 0:
            return "/"
        return without_scheme[slash_index:] or "/"

    sample_capture_urls = [
        str(url)
        for url in evidence_payload.get("sample_capture_urls", [])
        if str(url)
    ]
    sample_capture_url_paths = {
        _response_security_capture_path(url)
        for url in sample_capture_urls
    }
    preferred_capture_url = str(evidence_payload.get("preferred_capture_url") or "")
    preferred_capture_url_path = _response_security_capture_path(preferred_capture_url)
    capture_url_value = str(capture.get("url") or "")
    capture_url_path = _response_security_capture_path(capture_url_value)
    expected_user_capture_query_required = bool(
        evidence_payload.get("expected_user_capture_query_required")
    )
    expected_user_query_value = _expected_user_query_value_from_url(capture_url_value)
    expected_user_query_present = bool(expected_user_query_value)
    authenticated_test_user_present = bool(
        str(capture.get("authenticated_test_user") or "").strip()
    )
    authenticated_test_user_matches_expected_user = bool(
        expected_user_query_value
        and str(capture.get("authenticated_test_user") or "").strip()
        == expected_user_query_value
    )
    capture_url_matches_preferred_path = (
        bool(preferred_capture_url_path)
        and capture_url_path == preferred_capture_url_path
    )
    capture_url_matches_sample_path = (
        not sample_capture_url_paths
        or capture_url_path in sample_capture_url_paths
    )
    capture_url_contract_ready = capture_url_matches_sample_path and (
        not expected_user_capture_query_required
        or expected_user_query_present
    )
    authenticated_test_user_contract_ready = (
        not expected_user_capture_query_required
        or (
            authenticated_test_user_present
            and authenticated_test_user_matches_expected_user
        )
    )
    checks["expected_user_capture_query_present"] = expected_user_query_present
    checks["authenticated_test_user_present"] = authenticated_test_user_present
    checks["authenticated_test_user_matches_expected_user"] = (
        authenticated_test_user_matches_expected_user
    )
    checks["capture_url_matches_preferred_path"] = capture_url_matches_preferred_path
    checks["capture_url_matches_sample_path"] = capture_url_matches_sample_path
    checks["capture_url_contract_ready"] = capture_url_contract_ready
    checks["authenticated_test_user_contract_ready"] = authenticated_test_user_contract_ready
    capture["expected_user_capture_query_required"] = expected_user_capture_query_required
    capture["expected_user_query_present"] = expected_user_query_present
    capture["expected_user_query_value"] = expected_user_query_value
    capture["authenticated_test_user_present"] = authenticated_test_user_present
    capture["authenticated_test_user_matches_expected_user"] = (
        authenticated_test_user_matches_expected_user
    )
    capture["matches_preferred_capture_path"] = capture_url_matches_preferred_path
    capture["matches_sample_capture_path"] = capture_url_matches_sample_path
    capture["capture_url_contract_ready"] = capture_url_contract_ready
    capture["authenticated_test_user_contract_ready"] = authenticated_test_user_contract_ready
    if expected_user_capture_query_required and not expected_user_query_present:
        errors.append(
            {
                "error": "expected_user_capture_query_missing",
                "capture_url": capture_url_value,
            }
        )
    if expected_user_capture_query_required and not authenticated_test_user_present:
        errors.append(
            {
                "error": "authenticated_test_user_missing",
                "capture_url": capture_url_value,
            }
        )
    if (
        expected_user_capture_query_required
        and expected_user_query_present
        and authenticated_test_user_present
        and not authenticated_test_user_matches_expected_user
    ):
        errors.append(
            {
                "error": "authenticated_test_user_expected_user_mismatch",
                "capture_url": capture_url_value,
                "expected_user": expected_user_query_value,
                "authenticated_test_user": str(
                    capture.get("authenticated_test_user") or ""
                ).strip(),
            }
        )
    if sample_capture_url_paths and not capture_url_matches_sample_path:
        errors.append(
            {
                "error": "capture_url_outside_sample_paths",
                "capture_url": capture_url_value,
                "sample_capture_urls": sample_capture_urls,
            }
        )
    response_security_evidence_approved = (
        all_headers_preserved
        and capture_url_contract_ready
        and authenticated_test_user_contract_ready
    )
    operator_approval = (
        dict(evidence_payload.get("operator_approval"))
        if isinstance(evidence_payload.get("operator_approval"), Mapping)
        else {}
    )
    operator_approval.update(
        {
            "status": "operator_approved" if response_security_evidence_approved else "pending_operator_confirmation",
            "operator": operator,
            "approved_at_utc": now if response_security_evidence_approved else "",
            "notes": str(notes or operator_approval.get("notes") or ""),
        }
    )
    evidence_payload["captured_headers"] = captured_headers
    evidence_payload["checks"] = checks
    evidence_payload["capture"] = capture
    evidence_payload["operator_approval"] = operator_approval
    evidence_payload["updated_at_utc"] = now
    evidence_payload["updated_by"] = operator
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "ok": response_security_evidence_approved and not errors,
        "schema": "palette.web_labeling_browser_response_security_evidence_update_report.v1",
        "updated_at_utc": now,
        "operator": operator,
        "evidence_path": str(evidence_path),
        "output_path": str(destination_path),
        "matched_header_count": matched_header_count,
        "expected_header_count": len(expected_headers),
        "capture_url_contract_ready": capture_url_contract_ready,
        "required_capture_contract": required_capture_contract,
        "expected_user_capture_query_required": expected_user_capture_query_required,
        "expected_user_query_present": expected_user_query_present,
        "expected_user_query_value": expected_user_query_value,
        "authenticated_test_user_present": authenticated_test_user_present,
        "authenticated_test_user_matches_expected_user": (
            authenticated_test_user_matches_expected_user
        ),
        "authenticated_test_user_contract_ready": authenticated_test_user_contract_ready,
        "capture_url_matches_preferred_path": capture_url_matches_preferred_path,
        "capture_url_matches_sample_path": capture_url_matches_sample_path,
        "operator_approval_status": operator_approval["status"],
        "error_count": len(errors),
        "errors": errors,
    }


BROWSER_SMOKE_REQUIRED_FIELDS = (
    "identity_matches_expected_user",
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
)


def _record_browser_smoke_evidence(
    *,
    evidence_path: Path,
    expected_user: str,
    resolved_user: str,
    operator: str,
    checks: Mapping[str, bool],
    notes: str | None = None,
    output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("Browser smoke evidence template must be a JSON object.")
    destination_path = output or evidence_path
    if output is not None and destination_path.exists() and destination_path != evidence_path and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination_path}")
    users = evidence_payload.get("users") if isinstance(evidence_payload.get("users"), list) else []
    personalized_route_smoke_contract = (
        dict(evidence_payload.get("personalized_route_smoke_contract"))
        if isinstance(evidence_payload.get("personalized_route_smoke_contract"), Mapping)
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
            "before recording approval so personalized /my-datasets, /labeling, and /my-work route "
            "contract metadata is present."
        )
    )
    expected_user_value = str(expected_user or "").strip()
    resolved_user_value = str(resolved_user or "").strip()
    now = datetime.now(timezone.utc).isoformat()
    matched_row: dict[str, object] | None = None
    for row in users:
        if isinstance(row, dict) and str(row.get("expected_user") or "").strip() == expected_user_value:
            matched_row = row
            break
    errors: list[dict[str, object]] = []
    if not personalized_route_smoke_contract_ready:
        errors.append(
            {
                "error": "personalized_route_smoke_contract_stale",
                "missing_fields": personalized_route_smoke_contract_missing_fields,
                "expected_personalized_route_smoke_contract": (
                    expected_personalized_route_smoke_contract
                ),
                "actual_personalized_route_smoke_contract": personalized_route_smoke_contract,
                "operator_action": personalized_route_smoke_contract_operator_action,
            }
        )
    if matched_row is None:
        errors.append(
            {
                "expected_user": expected_user_value,
                "error": "expected_user_missing",
                "details": "The browser smoke evidence template has no row for this expected user.",
            }
        )
    else:
        identity_matches = bool(expected_user_value and resolved_user_value and expected_user_value == resolved_user_value)
        field_values = {
            "identity_matches_expected_user": identity_matches,
            **{field: bool(checks.get(field)) for field in BROWSER_SMOKE_REQUIRED_FIELDS if field != "identity_matches_expected_user"},
        }
        missing_fields = [field for field in BROWSER_SMOKE_REQUIRED_FIELDS if not bool(field_values.get(field))]
        if not identity_matches:
            errors.append(
                {
                    "expected_user": expected_user_value,
                    "resolved_user": resolved_user_value,
                    "error": "identity_mismatch",
                    "details": "Resolved user must exactly match expected user before browser smoke evidence can be approved.",
                }
            )
        if missing_fields:
            errors.append(
                {
                    "expected_user": expected_user_value,
                    "error": "browser_smoke_checks_incomplete",
                    "missing_fields": missing_fields,
                }
            )
        approved = not errors
        matched_row.update(
            {
                "run_status": "operator_approved" if approved else "pending_operator_confirmation",
                "resolved_user": resolved_user_value,
                "captured_at_utc": now,
                "operator": operator,
                "operator_approved_at_utc": now if approved else "",
                **field_values,
            }
        )
        if notes is not None:
            matched_row["notes"] = str(notes)
    approved_count = (
        0
        if not personalized_route_smoke_contract_ready
        else sum(
            1
            for row in users
            if isinstance(row, Mapping)
            and str(row.get("run_status") or "") == "operator_approved"
            and str(row.get("expected_user") or "").strip()
            and str(row.get("resolved_user") or "").strip() == str(row.get("expected_user") or "").strip()
            and bool(str(row.get("operator") or "").strip())
            and bool(str(row.get("operator_approved_at_utc") or "").strip())
            and all(bool(row.get(field)) for field in BROWSER_SMOKE_REQUIRED_FIELDS)
        )
    )
    candidate_count = len([row for row in users if isinstance(row, Mapping)])
    evidence_payload["updated_at_utc"] = now
    evidence_payload["updated_by"] = operator
    evidence_payload["counts"] = {
        **(
            evidence_payload.get("counts")
            if isinstance(evidence_payload.get("counts"), Mapping)
            else {}
        ),
        "candidate_users": candidate_count,
        "pending_operator_confirmation": max(candidate_count - approved_count, 0),
        "operator_approved": approved_count,
    }
    if matched_row is not None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "ok": matched_row is not None and not errors,
        "schema": "palette.web_labeling_browser_smoke_evidence_update_report.v1",
        "personalized_route_smoke_contract": personalized_route_smoke_contract,
        "actual_personalized_route_smoke_contract": personalized_route_smoke_contract,
        "expected_personalized_route_smoke_contract": expected_personalized_route_smoke_contract,
        "personalized_route_smoke_contract_ready": personalized_route_smoke_contract_ready,
        "personalized_route_smoke_contract_missing_fields": (
            personalized_route_smoke_contract_missing_fields
        ),
        "personalized_route_smoke_contract_operator_action": (
            personalized_route_smoke_contract_operator_action
        ),
        "updated_at_utc": now,
        "operator": operator,
        "evidence_path": str(evidence_path),
        "output_path": str(destination_path),
        "expected_user": expected_user_value,
        "resolved_user": resolved_user_value,
        "error_count": len(errors),
        "errors": errors,
        "counts": evidence_payload["counts"],
    }


DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS = (
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
)


def _disposable_zarr_smoke_workflow_contract_missing_fields(
    row: Mapping[str, object],
) -> list[str]:
    missing_fields: list[str] = []
    primary_target_kind = str(row.get("primary_mutation_target_kind") or "").strip()
    training_write_mode = str(row.get("training_zarr_write_mode") or "").strip()
    if str(row.get("data_plane_write_target") or "").strip() != "server_owned_assigned_task_zarr_scope":
        missing_fields.append("data_plane_write_target")
    if not primary_target_kind.startswith("task_scoped_"):
        missing_fields.append("primary_mutation_target_kind")
    if str(row.get("training_zarr_mutation_target_kind") or "").strip() != "task_scoped_training_zarr":
        missing_fields.append("training_zarr_mutation_target_kind")
    if str(row.get("browser_label_write_target") or "").strip() != "training_zarr":
        missing_fields.append("browser_label_write_target")
    if training_write_mode not in {"direct", "promotion_when_configured"}:
        missing_fields.append("training_zarr_write_mode")
    if str(row.get("csv_handoff_artifact_role") or "").strip() != "metadata_only_control_plane":
        missing_fields.append("csv_handoff_artifact_role")
    if bool(row.get("csv_handoff_artifacts_are_label_write_targets")):
        missing_fields.append("csv_handoff_artifacts_are_label_write_targets")
    if row.get("handoff_csv_artifacts_are_label_write_targets") is not False:
        missing_fields.append("handoff_csv_artifacts_are_label_write_targets")
    if row.get("intermediate_csv_artifacts_are_label_write_targets") is not False:
        missing_fields.append("intermediate_csv_artifacts_are_label_write_targets")
    if not bool(row.get("handoff_artifacts_are_metadata_only")):
        missing_fields.append("handoff_artifacts_are_metadata_only")
    if bool(row.get("browser_writes_csv_or_handoff_files")):
        missing_fields.append("browser_writes_csv_or_handoff_files")
    if row.get("browser_writes_handoff_csv") is not False:
        missing_fields.append("browser_writes_handoff_csv")
    if row.get("browser_writes_intermediate_csv") is not False:
        missing_fields.append("browser_writes_intermediate_csv")
    if bool(row.get("browser_receives_zarr_write_authority")):
        missing_fields.append("browser_receives_zarr_write_authority")
    if bool(row.get("browser_has_direct_zarr_write_authority")):
        missing_fields.append("browser_has_direct_zarr_write_authority")
    return missing_fields


def _record_disposable_zarr_mutation_smoke_evidence(
    *,
    evidence_path: Path,
    workflow_kind: str,
    operator: str,
    mutation_event_ids: Sequence[str],
    checks: Mapping[str, bool],
    event_lookup_reports: Sequence[Path] = (),
    registry_refresh_event_ids: Sequence[str] = (),
    disposable_recording_id: str | None = None,
    disposable_task_id: str | None = None,
    labeler_user: str | None = None,
    disposable_zarr_or_known_good_source: str | None = None,
    bad_mutation_recovery_mode: str | None = None,
    bad_mutation_recovery_report: str | None = None,
    notes: str | None = None,
    output: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence_payload, dict):
        raise ValueError("Disposable-Zarr mutation smoke evidence template must be a JSON object.")
    destination_path = output or evidence_path
    if output is not None and destination_path.exists() and destination_path != evidence_path and not overwrite:
        raise FileExistsError(f"Output file already exists: {destination_path}")
    workflows = evidence_payload.get("workflows") if isinstance(evidence_payload.get("workflows"), list) else []
    workflow_kind_value = str(workflow_kind or "").strip()
    now = datetime.now(timezone.utc).isoformat()
    matched_row: dict[str, object] | None = None
    for row in workflows:
        if isinstance(row, dict) and str(row.get("workflow_kind") or "").strip() == workflow_kind_value:
            matched_row = row
            break
    errors: list[dict[str, object]] = []
    normalized_event_ids = [str(event_id).strip() for event_id in mutation_event_ids if str(event_id).strip()]
    normalized_registry_event_ids = [
        str(event_id).strip()
        for event_id in registry_refresh_event_ids
        if str(event_id).strip()
    ]
    expected_task_id = str(disposable_task_id or "").strip()
    expected_recording_id = str(disposable_recording_id or "").strip()
    expected_labeler_user = str(labeler_user or "").strip()
    event_lookup_report_paths = [Path(report_path) for report_path in event_lookup_reports]
    event_lookup_report_path_values = [str(report_path) for report_path in event_lookup_report_paths]
    event_lookup_report_event_ids: list[str] = []
    event_lookup_report_errors: list[dict[str, object]] = []
    for report_path in event_lookup_report_paths:
        path_value = str(report_path)
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            event_lookup_report_errors.append(
                {
                    "error": "event_lookup_report_invalid",
                    "path": path_value,
                    "details": str(exc),
                }
            )
            continue
        if not isinstance(report_payload, Mapping):
            event_lookup_report_errors.append(
                {
                    "error": "event_lookup_report_invalid",
                    "path": path_value,
                    "details": "Event lookup report must be a JSON object.",
                }
            )
            continue
        event_payload = report_payload.get("event") if isinstance(report_payload.get("event"), Mapping) else {}
        report_event_id = str(report_payload.get("event_id") or event_payload.get("event_id") or "").strip()
        if not bool(report_payload.get("ok")):
            event_lookup_report_errors.append(
                {
                    "error": "event_lookup_report_not_ok",
                    "path": path_value,
                    "event_id": report_event_id,
                    "details": "Only successful lookup-event reports can prove operator event lookup.",
                }
            )
            continue
        if not report_event_id:
            event_lookup_report_errors.append(
                {
                    "error": "event_lookup_report_event_id_missing",
                    "path": path_value,
                    "details": "Lookup report did not include an event_id.",
                }
            )
            continue
        mismatches: list[dict[str, str]] = []
        for field_name, expected_value in (
            ("task_id", expected_task_id),
            ("recording_id", expected_recording_id),
            ("user", expected_labeler_user),
        ):
            if not expected_value:
                continue
            actual_value = str(event_payload.get(field_name) or "").strip()
            if actual_value != expected_value:
                mismatches.append(
                    {
                        "field": field_name,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
        report_workflow_kind = str(event_payload.get("workflow_kind") or "").strip()
        if report_workflow_kind and report_workflow_kind != workflow_kind_value:
            mismatches.append(
                {
                    "field": "workflow_kind",
                    "expected": workflow_kind_value,
                    "actual": report_workflow_kind,
                }
            )
        if mismatches:
            event_lookup_report_errors.append(
                {
                    "error": "event_lookup_report_context_mismatch",
                    "path": path_value,
                    "event_id": report_event_id,
                    "mismatches": mismatches,
                    "details": "Lookup report event context must match the disposable smoke task, recording, supplied labeler user, and reported workflow before lookup evidence is accepted.",
                }
            )
            continue
        if report_event_id not in event_lookup_report_event_ids:
            event_lookup_report_event_ids.append(report_event_id)
    event_lookup_report_event_id_set = set(event_lookup_report_event_ids)
    event_lookup_reports_cover_mutation_ids = bool(normalized_event_ids) and set(normalized_event_ids).issubset(
        event_lookup_report_event_id_set
    )
    operator_event_lookup_verified = (
        bool(checks.get("operator_event_lookup_verified")) or event_lookup_reports_cover_mutation_ids
    )
    if matched_row is None:
        errors.append(
            {
                "workflow_kind": workflow_kind_value,
                "error": "workflow_kind_missing",
                "details": "The disposable-Zarr mutation smoke evidence template has no row for this workflow kind.",
            }
        )
    else:
        effective_checks = dict(checks)
        effective_checks["operator_event_lookup_verified"] = operator_event_lookup_verified
        field_values = {
            field: bool(effective_checks.get(field))
            for field in DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS
        }
        missing_fields = [
            field
            for field in DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS
            if not bool(field_values.get(field))
        ]
        for report_error in event_lookup_report_errors:
            errors.append({"workflow_kind": workflow_kind_value, **report_error})
        if event_lookup_report_paths and normalized_event_ids and not event_lookup_reports_cover_mutation_ids:
            missing_lookup_event_ids = [
                event_id
                for event_id in normalized_event_ids
                if event_id not in event_lookup_report_event_id_set
            ]
            if missing_lookup_event_ids:
                errors.append(
                    {
                        "workflow_kind": workflow_kind_value,
                        "error": "operator_event_lookup_report_missing_mutation_ids",
                        "missing_mutation_event_ids": missing_lookup_event_ids,
                        "lookup_event_ids": event_lookup_report_event_ids,
                        "details": "Provided lookup-event reports must cover every mutation event ID before lookup evidence is accepted.",
                    }
                )
        if not normalized_event_ids:
            errors.append(
                {
                    "workflow_kind": workflow_kind_value,
                    "error": "mutation_event_ids_missing",
                    "details": "At least one mutation event ID is required before disposable-Zarr mutation smoke evidence can be approved.",
                }
            )
        if missing_fields:
            errors.append(
                {
                    "workflow_kind": workflow_kind_value,
                    "error": "disposable_zarr_mutation_smoke_checks_incomplete",
                    "missing_fields": missing_fields,
                }
            )
        approved = not errors
        matched_row.update(
            {
                "status": "operator_approved" if approved else "pending_operator_confirmation",
                "mutation_event_ids": normalized_event_ids,
                "registry_refresh_event_ids": normalized_registry_event_ids,
                "operator_event_lookup_report_paths": event_lookup_report_path_values,
                "operator_event_lookup_event_ids": event_lookup_report_event_ids,
                "operator": operator,
                "operator_approved_at_utc": now if approved else "",
                **field_values,
            }
        )
        if disposable_recording_id is not None:
            matched_row["disposable_recording_id"] = str(disposable_recording_id)
        if disposable_task_id is not None:
            matched_row["disposable_task_id"] = str(disposable_task_id)
        if labeler_user is not None:
            matched_row["labeler_user"] = str(labeler_user)
        if disposable_zarr_or_known_good_source is not None:
            matched_row["disposable_zarr_or_known_good_source"] = str(disposable_zarr_or_known_good_source)
        if bad_mutation_recovery_mode is not None:
            matched_row["bad_mutation_recovery_mode"] = str(bad_mutation_recovery_mode)
        if bad_mutation_recovery_report is not None:
            matched_row["bad_mutation_recovery_report"] = str(bad_mutation_recovery_report)
        if notes is not None:
            matched_row["notes"] = str(notes)
    approved_count = sum(
        1
        for row in workflows
        if isinstance(row, Mapping)
        and str(row.get("status") or "") == "operator_approved"
        and all(bool(row.get(field)) for field in DISPOSABLE_ZARR_MUTATION_SMOKE_REQUIRED_FIELDS)
        and bool(row.get("mutation_event_ids"))
        and not _disposable_zarr_smoke_workflow_contract_missing_fields(row)
    )
    workflow_count = len([row for row in workflows if isinstance(row, Mapping)])
    evidence_payload["updated_at_utc"] = now
    evidence_payload["updated_by"] = operator
    evidence_payload["counts"] = {
        **(
            evidence_payload.get("counts")
            if isinstance(evidence_payload.get("counts"), Mapping)
            else {}
        ),
        "workflow_kinds": workflow_count,
        "pending_operator_confirmation": max(workflow_count - approved_count, 0),
        "operator_approved": approved_count,
    }
    if matched_row is not None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(
            json.dumps(evidence_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return {
        "ok": matched_row is not None and not errors,
        "schema": "palette.web_labeling_disposable_zarr_mutation_smoke_evidence_update_report.v1",
        "updated_at_utc": now,
        "operator": operator,
        "evidence_path": str(evidence_path),
        "output_path": str(destination_path),
        "workflow_kind": workflow_kind_value,
        "labeler_user": expected_labeler_user,
        "mutation_event_ids": normalized_event_ids,
        "event_lookup_report_paths": event_lookup_report_path_values,
        "event_lookup_event_ids": event_lookup_report_event_ids,
        "operator_event_lookup_verified": operator_event_lookup_verified,
        "bad_mutation_recovery_mode": str(bad_mutation_recovery_mode or ""),
        "bad_mutation_recovery_report": str(bad_mutation_recovery_report or ""),
        "error_count": len(errors),
        "errors": errors,
        "counts": evidence_payload["counts"],
    }
