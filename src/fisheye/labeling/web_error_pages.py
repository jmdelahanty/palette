"""Shared browser error-page rendering for the labeling web app."""

from __future__ import annotations

import html
import json
from http import HTTPStatus
from typing import Mapping

from .web_auth import (
    DASHBOARD_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
)
from .web_policy import PERSONAL_WORK_PATH


def _browser_error_html(payload: Mapping[str, object]) -> bytes:
    status = int(payload.get("status") or HTTPStatus.BAD_REQUEST)
    error = str(payload.get("error") or "request_failed")
    details = str(payload.get("details") or "Return to your landing page or ask the operator to inspect your assignment.")
    closure_event = payload.get("session_closure_event")
    closure_support_line = ""
    if isinstance(closure_event, Mapping):
        closure_support_line = "\nsession_closure_event=" + html.escape(json.dumps(dict(closure_event), sort_keys=True))
    authorization_context = payload.get("authorization_context")
    authorization_support_line = ""
    return_expected_user = ""
    if isinstance(authorization_context, Mapping):
        authorization_support_line = "\nauthorization_context=" + html.escape(
            json.dumps(dict(authorization_context), sort_keys=True)
        )
        return_expected_user = str(
            authorization_context.get("expected_user")
            or authorization_context.get("assignee_user")
            or authorization_context.get("session_user")
            or authorization_context.get("resolved_user")
            or ""
        ).strip()
    read_authorization_contract = payload.get("labeler_read_authorization_contract")
    read_authorization_support_line = ""
    if isinstance(read_authorization_contract, Mapping):
        read_authorization_support_line = (
            "\nlabeler_read_authorization_contract="
            + html.escape(json.dumps(dict(read_authorization_contract), sort_keys=True))
        )
        if not return_expected_user:
            return_expected_user = str(
                read_authorization_contract.get("return_expected_user")
                or read_authorization_contract.get("expected_user")
                or read_authorization_contract.get("assignee_user")
                or read_authorization_contract.get("session_user")
                or read_authorization_contract.get("resolved_user")
                or ""
            ).strip()
    if not return_expected_user:
        return_expected_user = str(
            payload.get("return_expected_user")
            or payload.get("expected_user")
            or payload.get("assignee_user")
            or payload.get("session_user")
            or payload.get("resolved_user")
            or ""
        ).strip()
    signed_link_policy = payload.get("signed_link_policy")
    signed_link_policy_support_line = ""
    if isinstance(signed_link_policy, Mapping):
        signed_link_policy_support_line = (
            "\nsigned_link_policy="
            + html.escape(json.dumps(dict(signed_link_policy), sort_keys=True))
        )
    extra_support_lines = ""
    for support_key in (
        "personalized_launch_readiness",
        "task_open_authorization_contract",
        "signed_link_contract",
        "browser_mutation_write_policy",
        "browser_mutation_write_contract",
    ):
        support_value = payload.get(support_key)
        if isinstance(support_value, Mapping):
            extra_support_lines += (
                f"\n{support_key}="
                + html.escape(json.dumps(dict(support_value), sort_keys=True))
            )
    for support_key in ("signed_links_enabled",):
        if support_key in payload:
            extra_support_lines += (
                f"\n{support_key}="
                + html.escape(str(payload.get(support_key)))
            )
    personal_dataset_queue_url = html.escape(
        _dashboard_url_for_expected_user(PERSONAL_DATASET_QUEUE_PATH, return_expected_user)
        if return_expected_user
        else PERSONAL_DATASET_QUEUE_PATH
    )
    personal_work_url = html.escape(
        _dashboard_url_for_expected_user(PERSONAL_WORK_PATH, return_expected_user)
        if return_expected_user
        else PERSONAL_WORK_PATH
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Palette labeling access problem</title>
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
      width: min(42rem, calc(100vw - 2rem));
      border: 1px solid #d8d1bf;
      border-radius: 1.2rem;
      background: rgba(255, 255, 255, .78);
      padding: 2rem;
      box-shadow: 0 1rem 3rem rgba(23, 33, 28, .1);
    }}
    h1 {{
      margin: 0 0 .75rem;
      font-size: clamp(2rem, 5vw, 3.5rem);
      line-height: .95;
      letter-spacing: -.04em;
    }}
    code {{
      display: inline-block;
      border-radius: .5rem;
      padding: .2rem .45rem;
      background: #f6eadf;
      color: #8b3f19;
      font-weight: 700;
    }}
    a {{
      color: #0f6b5f;
      font-weight: 800;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid #d8d1bf;
      border-radius: .75rem;
      background: #fffdf6;
      padding: .85rem;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      background: #0f6b5f;
      color: white;
      padding: .7rem 1rem;
      font-weight: 800;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Palette labeling access problem</h1>
    <p><code>{html.escape(error)}</code> status {status}</p>
    <p>{html.escape(details)}</p>
    <details>
      <summary>What to send the operator</summary>
      <pre id="operator-support">error={html.escape(error)}
status={status}
details={html.escape(details)}
return_expected_user={html.escape(return_expected_user)}
return_personal_dataset_queue_url={personal_dataset_queue_url}
return_personal_dataset_queue_expected_user_guarded={bool(return_expected_user)}
return_personal_work_url={personal_work_url}
return_personal_work_expected_user_guarded={bool(return_expected_user)}{closure_support_line}{authorization_support_line}{read_authorization_support_line}{signed_link_policy_support_line}{extra_support_lines}</pre>
      <button type="button" onclick="copyOperatorSupport(this)">Copy support details</button>
    </details>
    <p><a href="/">Return to your labeling landing page</a></p>
    <p><a href="{html.escape(DASHBOARD_PATH)}">Return to the work dashboard</a></p>
    <p><a href="{personal_dataset_queue_url}">Return to your personalized dataset queue</a></p>
    <p><a href="{personal_work_url}">Return to your personalized work dashboard</a></p>
  </main>
  <script>
    function copyOperatorSupport(button) {{
      const text = document.getElementById("operator-support").textContent;
      const markCopied = () => {{
        button.textContent = "Copied";
        window.setTimeout(() => {{ button.textContent = "Copy support details"; }}, 1800);
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













__all__ = ["_browser_error_html"]
