"""Identity-probe HTML rendering for the labeling web app."""

from __future__ import annotations

import html
import json
from typing import Mapping

from .admin_dashboard import (
    _assignment_ownership_policy,
    _dataset_queue_direct_start_policy,
    _dataset_queue_direct_start_policy_fields,
    _identity_source_policy,
    _labeler_safety_policy,
    _queue_first_entry_contract_policy,
    _single_owner_assignment_live_contract_fields,
    _single_owner_policy_fields,
)
from .assignment_store import LabelingStore
from .web_auth import (
    DASHBOARD_PATH,
    DATASET_QUEUE_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
)
from .web_diagnostics import (
    _add_payload_contract_compact_fields,
    _browser_mutation_target_contract_compact_fields,
    _direct_browser_start_contract_compact_fields,
)
from .web_policy import (
    IDENTITY_PROBE_PATH,
    LABELING_HOME_PATH,
    PERSONAL_WORK_PATH,
    _browser_mutation_write_policy,
    _browser_mutation_write_runtime_checklist,
)


def _identity_probe_payload(
    *,
    user: str,
    auth_source: str,
    expected_user: str | None = None,
    config: ServerConfig | None = None,
    store: LabelingStore | None = None,
    assignment_ownership_integrity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    expected = str(expected_user or "").strip()
    resolved_user = str(user or "").strip()
    expected_user_guard_present = bool(expected)
    matches_expected = bool(expected_user_guard_present and resolved_user == expected)
    expected_user_dataset_queue_url = (
        _dashboard_url_for_expected_user(DATASET_QUEUE_PATH, expected) if expected else ""
    )
    expected_user_labeling_home_url = (
        _dashboard_url_for_expected_user(LABELING_HOME_PATH, expected) if expected else LABELING_HOME_PATH
    )
    expected_user_personal_dataset_queue_url = (
        _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, expected) if expected else ""
    )
    expected_user_labeler_landing_url = (
        _dashboard_url_for_expected_user("/", expected) if expected else "/"
    )
    expected_user_dashboard_url = (
        _dashboard_url_for_expected_user(DASHBOARD_PATH, expected) if expected else ""
    )
    expected_user_personal_work_url = (
        _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, expected) if expected else ""
    )
    expected_user_identity_probe_url = (
        _dashboard_url_for_expected_user(IDENTITY_PROBE_PATH, expected)
        if expected
        else IDENTITY_PROBE_PATH
    )
    preferred_labeler_entry_url = (
        expected_user_personal_dataset_queue_url
        or expected_user_dataset_queue_url
        or (_dashboard_url_for_expected_user("/", expected) if expected else "/")
    )
    labeler_safety = _labeler_safety_policy()
    queue_first_entry_contract = _queue_first_entry_contract_policy(
        labeler_safety=labeler_safety,
        labeler_landing_page_path="/",
        labeler_landing_url="/",
        expected_user_labeler_landing_url=expected_user_labeler_landing_url,
        labeling_home_page_path=LABELING_HOME_PATH,
        labeling_home_url=LABELING_HOME_PATH,
        expected_user_labeling_home_url=expected_user_labeling_home_url,
        dataset_queue_page_path=DATASET_QUEUE_PATH,
        dataset_queue_url=DATASET_QUEUE_PATH,
        expected_user_dataset_queue_url=expected_user_dataset_queue_url,
        dashboard_url=DASHBOARD_PATH,
        expected_user_dashboard_url=expected_user_dashboard_url,
        personal_dataset_queue_page_path=PERSONAL_DATASET_QUEUE_PATH,
        personal_dataset_queue_url=PERSONAL_DATASET_QUEUE_PATH,
        expected_user_personal_dataset_queue_url=expected_user_personal_dataset_queue_url,
        personal_work_page_path=PERSONAL_WORK_PATH,
        personal_work_url=PERSONAL_WORK_PATH,
        expected_user_personal_work_url=expected_user_personal_work_url,
    )
    single_owner_policy = _assignment_ownership_policy()
    single_owner_live_contract_fields = (
        _single_owner_assignment_live_contract_fields(
            store,
            integrity=assignment_ownership_integrity,
        )
        if store is not None
        else {}
    )
    browser_mutation_write_policy = _browser_mutation_write_policy()
    browser_mutation_write_checklist = _browser_mutation_write_runtime_checklist(
        browser_mutation_write_policy
    )
    dataset_queue_direct_start_policy = _dataset_queue_direct_start_policy()
    identity: dict[str, object] = {
        "resolved_user": resolved_user,
        "auth_source": str(auth_source or ""),
        "expected_user": expected,
        "matches_expected_user": matches_expected,
        "assignment_user_match_required": True,
        "dashboard_path": DASHBOARD_PATH,
        "dataset_queue_page_path": DATASET_QUEUE_PATH,
        "labeling_home_page_path": LABELING_HOME_PATH,
        "personal_work_page_path": PERSONAL_WORK_PATH,
        "personal_dataset_queue_page_path": PERSONAL_DATASET_QUEUE_PATH,
        "personal_work_alias_for": DASHBOARD_PATH,
        "personal_dataset_queue_alias_for": DATASET_QUEUE_PATH,
        "labeler_landing_page_path": "/",
        "identity_probe_path": IDENTITY_PROBE_PATH,
        "identity_probe_api_path": "/api/me/identity",
        "identity_probe_expected_user_guard_required": bool(
            labeler_safety.get("identity_probe_expected_user_guard_required")
        ),
        "identity_probe_diagnostic_only": bool(
            labeler_safety.get("identity_probe_diagnostic_only")
        ),
        "identity_probe_does_not_authorize_work": bool(
            labeler_safety.get("identity_probe_does_not_authorize_work")
        ),
        "identity_probe_unknown_user_blocks_work_surfaces": bool(
            labeler_safety.get("identity_probe_unknown_user_blocks_work_surfaces")
        ),
        "expected_user_guard_present": expected_user_guard_present,
        "identity_probe_launch_ctas_rendered": matches_expected,
        "identity_probe_launch_ctas_suppressed": not matches_expected,
        "identity_probe_failed_support_urls_diagnostic_only": not matches_expected,
        "expected_user_labeler_landing_url": expected_user_labeler_landing_url,
        "expected_user_labeling_home_url": expected_user_labeling_home_url,
        "expected_user_dashboard_url": expected_user_dashboard_url,
        "expected_user_dataset_queue_url": expected_user_dataset_queue_url,
        "expected_user_personal_work_url": expected_user_personal_work_url,
        "expected_user_personal_dataset_queue_url": expected_user_personal_dataset_queue_url,
        "expected_user_identity_probe_url": expected_user_identity_probe_url,
        "preferred_labeler_entrypoint": "personal_datasets_waiting_queue",
        "preferred_labeler_entry_url": preferred_labeler_entry_url,
        "personalized_labeler_entrypoint": "personal_datasets_waiting_queue",
        "personalized_labeler_entry_url": preferred_labeler_entry_url,
        "preferred_labeler_entry_url_matches_dataset_queue": bool(
            preferred_labeler_entry_url
            and preferred_labeler_entry_url
            in {
                expected_user_personal_dataset_queue_url,
                expected_user_dataset_queue_url,
            }
        ),
        "preferred_labeler_entry_url_matches_personal_dataset_queue": bool(
            expected_user_personal_dataset_queue_url
            and preferred_labeler_entry_url == expected_user_personal_dataset_queue_url
        ),
        "personalized_labeler_entry_url_matches_personal_dataset_queue": bool(
            expected_user_personal_dataset_queue_url
            and preferred_labeler_entry_url == expected_user_personal_dataset_queue_url
        ),
        "personal_dataset_queue_link_role": "preferred_queue",
        "dataset_queue_link_role": "canonical_queue_fallback",
        "canonical_dataset_queue_link_role": "canonical_queue_fallback",
        "labeler_safety": labeler_safety,
        "queue_first_entry_contract": queue_first_entry_contract,
        "single_owner_policy": single_owner_policy,
        "single_owner_assignment_contract": single_owner_live_contract_fields.get(
            "single_owner_assignment_contract",
            {
                "schema": "palette.web_labeling_assignment_single_owner_contract.v1",
                "ready": True,
                "browser_mutation_target_resolved_server_side": True,
                "browser_mutation_target_source": "recording_assignments.active_assignment",
                "labelers_mutate_assigned_training_zarrs": True,
                "labelers_mutate_intermediate_csvs": False,
            },
        ),
        **_single_owner_policy_fields(single_owner_policy),
        **single_owner_live_contract_fields,
        "browser_mutation_write_policy": browser_mutation_write_policy,
        "browser_mutation_write_checklist": browser_mutation_write_checklist,
        **_browser_mutation_target_contract_compact_fields(
            browser_mutation_write_checklist,
            user=resolved_user,
        ),
        "dataset_queue_direct_start_policy": dataset_queue_direct_start_policy,
        **_dataset_queue_direct_start_policy_fields(dataset_queue_direct_start_policy),
        **_direct_browser_start_contract_compact_fields(
            dataset_queue_direct_start_policy,
            user=resolved_user,
        ),
        "browser_label_write_target": str(
            browser_mutation_write_checklist.get("browser_label_write_target") or ""
        ),
        "label_mutation_target_kind": str(
            browser_mutation_write_checklist.get("label_mutation_target_kind") or ""
        ),
        "csv_handoff_artifact_role": str(
            browser_mutation_write_checklist.get("csv_handoff_artifact_role") or ""
        ),
        "csv_handoff_artifacts_are_label_write_targets": bool(
            browser_mutation_write_checklist.get("csv_handoff_artifacts_are_label_write_targets")
        ),
        "handoff_csv_artifacts_are_label_write_targets": bool(
            browser_mutation_write_checklist.get("handoff_csv_artifacts_are_label_write_targets")
        ),
        "intermediate_csv_artifacts_are_label_write_targets": bool(
            browser_mutation_write_checklist.get(
                "intermediate_csv_artifacts_are_label_write_targets"
            )
        ),
        "browser_writes_csv_or_handoff_files": bool(
            browser_mutation_write_checklist.get("browser_writes_csv_or_handoff_files")
        ),
        "browser_has_direct_zarr_write_authority": bool(
            browser_mutation_write_checklist.get("browser_has_direct_zarr_write_authority")
        ),
        "handoff_artifacts_are_metadata_only": bool(
            browser_mutation_write_policy.get("handoff_artifacts_are_metadata_only")
        ),
        "signed_links_are_not_identity": True,
        "operator_action": (
            "Identity matches the expected assignment user; open the guarded dataset queue first and use /work only as the full-dashboard fallback."
            if matches_expected
            else (
                "Stop before labeling. The operator must provide an expected-user guarded identity probe before sharing labeling links."
                if not expected_user_guard_present
                else "Stop before labeling. The operator must fix the signed-in browser identity or the recording assignment user."
            )
        ),
    }
    if config is not None:
        identity["identity_source_policy"] = _identity_source_policy(config)
    _add_payload_contract_compact_fields(identity)
    payload: dict[str, object] = {
        "ok": matches_expected,
        "identity": identity,
        "identity_probe_diagnostic_only": identity["identity_probe_diagnostic_only"],
        "identity_probe_does_not_authorize_work": identity[
            "identity_probe_does_not_authorize_work"
        ],
        "identity_probe_unknown_user_blocks_work_surfaces": identity[
            "identity_probe_unknown_user_blocks_work_surfaces"
        ],
        "identity_probe_expected_user_guard_required": identity[
            "identity_probe_expected_user_guard_required"
        ],
        "identity_probe_launch_ctas_rendered": identity[
            "identity_probe_launch_ctas_rendered"
        ],
        "identity_probe_launch_ctas_suppressed": identity[
            "identity_probe_launch_ctas_suppressed"
        ],
        "identity_probe_failed_support_urls_diagnostic_only": identity[
            "identity_probe_failed_support_urls_diagnostic_only"
        ],
        "personalized_launch_readiness": identity["personalized_launch_readiness"],
    }
    if not expected_user_guard_present:
        payload["error"] = "identity_expected_user_required"
        payload["details"] = (
            "This identity probe is missing an expected_user guard. "
            "Stop and contact the operator before labeling."
        )
    elif not matches_expected:
        payload["error"] = "identity_user_mismatch"
        payload["details"] = (
            f"This identity probe is for {expected}, but the browser is authenticated as {resolved_user}. "
            "Stop and contact the operator before labeling."
        )
    return payload

def _mark_identity_probe_unknown_labeling_user(payload: dict[str, object]) -> None:
    action = (
        "Stop before labeling. This browser identity is not assigned any active labeling "
        "recording; ask the operator to assign a recording or confirm the expected login identity."
    )
    payload["ok"] = False
    payload["error"] = "unknown_labeling_user"
    payload["details"] = (
        "This browser identity is not present in the labeling assignment store. "
        "Ask the operator to assign a recording or confirm the expected login identity."
    )
    payload["identity_probe_diagnostic_only"] = True
    payload["identity_probe_does_not_authorize_work"] = True
    payload["identity_probe_unknown_user_blocks_work_surfaces"] = True
    payload["identity_probe_expected_user_guard_required"] = True
    payload["identity_probe_launch_ctas_rendered"] = False
    payload["identity_probe_launch_ctas_suppressed"] = True
    payload["identity_probe_failed_support_urls_diagnostic_only"] = True
    identity = payload.get("identity")
    if isinstance(identity, dict):
        identity["operator_action"] = action
        identity["matches_known_labeling_user"] = False
        identity["known_assignment_store_user"] = False
        identity["identity_probe_diagnostic_only"] = True
        identity["identity_probe_does_not_authorize_work"] = True
        identity["identity_probe_unknown_user_blocks_work_surfaces"] = True
        identity["identity_probe_expected_user_guard_required"] = True
        identity["identity_probe_launch_ctas_rendered"] = False
        identity["identity_probe_launch_ctas_suppressed"] = True
        identity["identity_probe_failed_support_urls_diagnostic_only"] = True

def _identity_probe_html(payload: Mapping[str, object]) -> bytes:
    identity = payload.get("identity") if isinstance(payload.get("identity"), Mapping) else {}
    resolved_user = str(identity.get("resolved_user") or "")
    expected_user = str(identity.get("expected_user") or "")
    matches_expected = bool(identity.get("matches_expected_user"))
    ok = bool(payload.get("ok"))
    error = str(payload.get("error") or "")
    known_user_status = payload.get("known_user_status") if isinstance(payload.get("known_user_status"), Mapping) else {}
    known_labeler = bool(known_user_status.get("is_known_labeler"))
    if ok:
        status_text = "Identity matches expected user"
    elif error == "unknown_labeling_user":
        status_text = "Unknown labeling user: stop before labeling"
    elif error == "identity_expected_user_required":
        status_text = "Expected user required: stop before labeling"
    else:
        status_text = "Identity mismatch: stop before labeling"
    status_class = "ok" if ok else "bad"
    expected_text = expected_user or "No expected user supplied"
    dashboard_url = str(identity.get("expected_user_dashboard_url") or DASHBOARD_PATH)
    dataset_queue_url = str(identity.get("expected_user_dataset_queue_url") or DATASET_QUEUE_PATH)
    personal_work_url = str(identity.get("expected_user_personal_work_url") or dashboard_url)
    personal_dataset_queue_url = str(
        identity.get("expected_user_personal_dataset_queue_url") or dataset_queue_url
    )
    landing_url = str(identity.get("expected_user_labeler_landing_url") or "/")
    readiness = (
        identity.get("personalized_launch_readiness")
        if isinstance(identity.get("personalized_launch_readiness"), Mapping)
        else {}
    )
    if ok:
        entry_links_html = f"""
    <p><a href="{html.escape(personal_dataset_queue_url)}">Open your personalized dataset queue</a></p>
    <p><a href="{html.escape(landing_url)}">Open your datasets-waiting landing page</a></p>
    <p><a href="{html.escape(personal_work_url)}">Open your full personalized work dashboard</a></p>
    <p><a href="{html.escape(dataset_queue_url)}">Open canonical dataset queue fallback</a></p>
    <p><a href="{html.escape(dashboard_url)}">Open canonical work dashboard fallback</a></p>"""
    else:
        entry_links_html = """
    <p class="bad"><b>Do not open labeling work from this browser identity.</b> Copy the identity details below and contact the operator before labeling.</p>"""
    support_payload = json.dumps(payload, indent=2, sort_keys=True)
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Palette labeling identity check</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: radial-gradient(circle at top left, #e8f1dc 0, transparent 32rem), #fbfaf5;
      color: #17211c;
      font-family: "Aptos", "Segoe UI", sans-serif;
    }}
    main {{
      width: min(44rem, calc(100vw - 2rem));
      border: 1px solid #d8d1bf;
      border-radius: 1.2rem;
      background: #fffdf8;
      padding: 1.4rem;
      box-shadow: 0 1rem 3rem rgba(30, 38, 28, 0.12);
    }}
    dl {{
      display: grid;
      grid-template-columns: 12rem 1fr;
      gap: 0.6rem 0.9rem;
    }}
    dt {{
      color: #5e6b61;
      font-weight: 700;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .ok {{
      color: #0f6f5c;
    }}
    .bad {{
      color: #9b2f24;
    }}
    pre {{
      overflow: auto;
      border: 1px solid #d8d1bf;
      border-radius: 0.9rem;
      background: #fbfaf5;
      padding: 0.9rem;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      background: #0f6b5f;
      color: white;
      padding: 0.7rem 1rem;
      font-weight: 800;
      cursor: pointer;
    }}
    a {{
      color: #0f6f5c;
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Palette labeling identity check</h1>
    <p class="{status_class}"><b>{html.escape(status_text)}</b></p>
    <dl>
      <dt>Resolved user</dt><dd>{html.escape(resolved_user)}</dd>
      <dt>Expected user</dt><dd>{html.escape(expected_text)}</dd>
      <dt>Auth source</dt><dd>{html.escape(str(identity.get('auth_source') or ''))}</dd>
      <dt>Known labeler</dt><dd>{html.escape(str(known_labeler))}</dd>
      <dt>Required match</dt><dd>{html.escape(str(identity.get('assignment_user_match_required') or True))}</dd>
      <dt>Assignment scope</dt><dd>{html.escape(str(identity.get('single_owner_policy_assignment_scope') or 'recording'))}</dd>
      <dt>One active owner</dt><dd>{html.escape(str(identity.get('single_owner_policy_one_active_owner') or False))}</dd>
      <dt>Current-owner mutation</dt><dd>{html.escape(str(identity.get('single_owner_policy_browser_mutation_requires_current_assignment_owner') or False))}</dd>
      <dt>Browser label target</dt><dd>{html.escape(str(identity.get('browser_label_write_target') or ''))}</dd>
      <dt>CSV/handoff label target</dt><dd>{html.escape(str(identity.get('csv_handoff_artifacts_are_label_write_targets') or False))}</dd>
      <dt>Handoff CSV label target</dt><dd>{html.escape(str(identity.get('handoff_csv_artifacts_are_label_write_targets') or False))}</dd>
      <dt>Intermediate CSV label target</dt><dd>{html.escape(str(identity.get('intermediate_csv_artifacts_are_label_write_targets') or False))}</dd>
      <dt>Direct Zarr authority</dt><dd>{html.escape(str(identity.get('browser_has_direct_zarr_write_authority') or False))}</dd>
      <dt>Identity diagnostic only</dt><dd>{html.escape(str(identity.get('identity_probe_diagnostic_only') or False))}</dd>
      <dt>Identity authorizes work</dt><dd>{html.escape(str(not bool(identity.get('identity_probe_does_not_authorize_work'))))}</dd>
      <dt>Unknown identity blocks work</dt><dd>{html.escape(str(identity.get('identity_probe_unknown_user_blocks_work_surfaces') or False))}</dd>
      <dt>Preferred entry</dt><dd>{html.escape(str(identity.get('preferred_labeler_entrypoint') or ''))}</dd>
      <dt>Preferred entry URL</dt><dd>{html.escape(str(identity.get('preferred_labeler_entry_url') or ''))}</dd>
      <dt>Launch readiness schema</dt><dd>{html.escape(str(readiness.get('schema') or ''))}</dd>
      <dt>Launch personal queue URL</dt><dd>{html.escape(str(readiness.get('personalized_labeler_entry_url') or ''))}</dd>
      <dt>Launch browser target</dt><dd>{html.escape(str(readiness.get('browser_label_write_target') or ''))}</dd>
      <dt>Launch writes CSV/handoff</dt><dd>{html.escape(str(readiness.get('browser_writes_csv_or_handoff_files') if 'browser_writes_csv_or_handoff_files' in readiness else ''))}</dd>
      <dt>Launch direct Zarr authority</dt><dd>{html.escape(str(readiness.get('browser_has_direct_zarr_write_authority') if 'browser_has_direct_zarr_write_authority' in readiness else ''))}</dd>
      <dt>Personal queue role</dt><dd>{html.escape(str(identity.get('personal_dataset_queue_link_role') or ''))}</dd>
      <dt>Canonical queue role</dt><dd>{html.escape(str(identity.get('canonical_dataset_queue_link_role') or ''))}</dd>
      <dt>Preferred matches personal queue</dt><dd>{html.escape(str(identity.get('preferred_labeler_entry_url_matches_personal_dataset_queue') or False))}</dd>
      <dt>Personalized matches personal queue</dt><dd>{html.escape(str(identity.get('personalized_labeler_entry_url_matches_personal_dataset_queue') or False))}</dd>
    </dl>
    <p>{html.escape(str(identity.get('operator_action') or ''))}</p>
    <p>Browser saves run through server-side assigned task/training Zarr writers; CSV, HTML, JSON, and handoff files are metadata only. Each recording has one active assigned owner, and only that current assignee can open or save browser labeling work.</p>
{entry_links_html}
    <details>
      <summary>What to send the operator</summary>
      <pre id="identity-support">{html.escape(support_payload)}</pre>
      <button type="button" onclick="copyIdentitySupport(this)">Copy identity details</button>
    </details>
  </main>
  <script>
    function copyIdentitySupport(button) {{
      const text = document.getElementById("identity-support").textContent;
      const markCopied = () => {{
        button.textContent = "Copied";
        window.setTimeout(() => {{ button.textContent = "Copy identity details"; }}, 1800);
      }};
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(markCopied).catch(() => {{
          const textarea = document.createElement("textarea");
          textarea.value = text;
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
          markCopied();
        }});
        return;
      }}
      const textarea = document.createElement("textarea");
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
      markCopied();
    }}
  </script>
</body>
</html>
"""
    return body.encode("utf-8")




__all__ = [
    "_identity_probe_html",
    "_identity_probe_payload",
    "_mark_identity_probe_unknown_labeling_user",
]
