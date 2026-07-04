"""Identity-probe HTML rendering for the labeling web app."""

from __future__ import annotations

import html
import json
from typing import Mapping

from .web_auth import DASHBOARD_PATH, DATASET_QUEUE_PATH


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




__all__ = ["_identity_probe_html"]
