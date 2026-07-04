"""Session/editor HTML renderers for labeling web surfaces."""

from __future__ import annotations

import html
import json
from typing import Mapping

from .admin_registry import _task_title
from .template_assets import read_labeling_asset, render_labeling_template
from .web_auth import (
    DASHBOARD_PATH,
    PERSONAL_DATASET_QUEUE_PATH,
    _dashboard_url_for_expected_user,
)
from .web_policy import PERSONAL_WORK_PATH

__all__ = [
    "_BROWSER_MUTATION_STATUS_JS",
    "_IMAGE_CANVAS_VIEWPORT_JS",
    "_SESSION_OPERATOR_SUPPORT_CSS",
    "_SESSION_OPERATOR_SUPPORT_HTML",
    "_SESSION_OPERATOR_SUPPORT_JS",
    "_detect_session_html",
    "_keypoint_session_html",
    "_session_html",
    "_session_return_links_html",
    "_session_return_url",
    "_session_status_banner",
    "_subject_mask_session_html",
    "_video_detect_session_html",
]

_SESSION_OPERATOR_SUPPORT_CSS = read_labeling_asset("static/css/session_operator_support.css")
_SESSION_OPERATOR_SUPPORT_HTML = read_labeling_asset("templates/partials/session_operator_support.html")
_SESSION_OPERATOR_SUPPORT_JS = read_labeling_asset("static/js/operator_support.js")
_BROWSER_MUTATION_STATUS_JS = read_labeling_asset("static/js/browser_mutation_status.js")
_IMAGE_CANVAS_VIEWPORT_JS = read_labeling_asset("static/js/image_canvas_viewport.js")


def _session_status_banner(session: Mapping[str, object]) -> str:
    session_id = html.escape(str(session.get("session_id") or ""))
    expires_at = html.escape(str(session.get("expires_at_utc") or "unknown"))
    task_id = html.escape(str(session.get("task_id") or ""))
    recording_id = html.escape(str(session.get("recording_id") or ""))
    personal_queue_url = html.escape(_session_return_url(session, PERSONAL_DATASET_QUEUE_PATH))
    personal_work_url = html.escape(_session_return_url(session, PERSONAL_WORK_PATH))
    closed_at = str(session.get("closed_at_utc") or "").strip()
    state_text = "closed" if closed_at else "active"
    closed_bits = f" Closed at {html.escape(closed_at)}." if closed_at else ""
    return f"""
    <section style="border:1px solid #d7ded5;border-radius:18px;background:rgba(255,253,245,.82);padding:12px 14px;margin:-4px 0 18px;color:#5f6d62;box-shadow:0 10px 28px rgba(23,32,26,.08);">
      <b style="color:#17201a;">Session {state_text}</b>
      <span>Task <code>{task_id}</code> for recording <code>{recording_id}</code> expires at <code>{expires_at}</code>.{closed_bits}</span>
      <span style="display:block;margin-top:4px;">If this tab reports a superseded session, expired session, or completed task, return to <a href="{personal_queue_url}">your personalized dataset queue</a> or <a href="{personal_work_url}">your personalized work dashboard</a> and reopen the task.</span>
      <span style="display:block;margin-top:4px;font-size:.88rem;">Session <code>{session_id}</code></span>
    </section>
"""


def _session_return_url(session: Mapping[str, object], path: str) -> str:
    expected_user = str(
        session.get("user")
        or session.get("assignee_user")
        or session.get("expected_user")
        or ""
    ).strip()
    return _dashboard_url_for_expected_user(path, expected_user) if expected_user else path


def _session_return_links_html(session: Mapping[str, object]) -> str:
    personal_queue_url = html.escape(_session_return_url(session, PERSONAL_DATASET_QUEUE_PATH))
    personal_work_url = html.escape(_session_return_url(session, PERSONAL_WORK_PATH))
    return (
        f'<a href="{personal_queue_url}" class="meta" data-session-return="dataset-queue">Personalized dataset queue</a> - '
        f'<a href="{personal_work_url}" class="meta" data-session-return="work-dashboard">Personalized work dashboard</a>'
    )


def _keypoint_session_html(session: Mapping[str, object]) -> bytes:
    safe_title = html.escape(_task_title(session))
    session_id = html.escape(str(session.get("session_id") or ""))
    status_banner = _session_status_banner(session)
    return_links = _session_return_links_html(session)
    body = render_labeling_template(
        "sessions/keypoint.html.j2",
        {
            "safe_title": safe_title,
            "session_id": session_id,
            "return_links": return_links,
            "status_banner": status_banner,
            "session_operator_support_css": _SESSION_OPERATOR_SUPPORT_CSS,
            "session_operator_support_html": _SESSION_OPERATOR_SUPPORT_HTML,
            "operator_support_js": _SESSION_OPERATOR_SUPPORT_JS,
            "browser_mutation_status_js": _BROWSER_MUTATION_STATUS_JS,
            "image_canvas_viewport_js": _IMAGE_CANVAS_VIEWPORT_JS,
            "keypoint_editor_js": read_labeling_asset("static/js/keypoint_editor.js"),
            "session_id_json": json.dumps(str(session.get("session_id") or "")),
        },
    )
    return body.encode("utf-8")



def _detect_session_html(session: Mapping[str, object]) -> bytes:
    safe_title = html.escape(_task_title(session))
    session_id = html.escape(str(session.get("session_id") or ""))
    status_banner = _session_status_banner(session)
    return_links = _session_return_links_html(session)
    body = render_labeling_template(
        "sessions/detect.html.j2",
        {
            "safe_title": safe_title,
            "session_id": session_id,
            "return_links": return_links,
            "status_banner": status_banner,
            "session_operator_support_css": _SESSION_OPERATOR_SUPPORT_CSS,
            "session_operator_support_html": _SESSION_OPERATOR_SUPPORT_HTML,
            "operator_support_js": _SESSION_OPERATOR_SUPPORT_JS,
            "browser_mutation_status_js": _BROWSER_MUTATION_STATUS_JS,
            "image_canvas_viewport_js": _IMAGE_CANVAS_VIEWPORT_JS,
            "detect_editor_js": read_labeling_asset("static/js/detect_editor.js"),
            "session_id_json": json.dumps(str(session.get("session_id") or "")),
        },
    )
    return body.encode("utf-8")



def _video_detect_session_html(session: Mapping[str, object]) -> bytes:
    safe_title = html.escape(_task_title(session))
    session_id = html.escape(str(session.get("session_id") or ""))
    status_banner = _session_status_banner(session)
    return_links = _session_return_links_html(session)
    body = render_labeling_template(
        "sessions/video_detect.html.j2",
        {
            "safe_title": safe_title,
            "session_id": session_id,
            "return_links": return_links,
            "status_banner": status_banner,
            "session_operator_support_css": _SESSION_OPERATOR_SUPPORT_CSS,
            "session_operator_support_html": _SESSION_OPERATOR_SUPPORT_HTML,
            "operator_support_js": _SESSION_OPERATOR_SUPPORT_JS,
            "browser_mutation_status_js": _BROWSER_MUTATION_STATUS_JS,
            "video_detect_editor_js": read_labeling_asset("static/js/video_detect_editor.js"),
            "session_id_json": json.dumps(str(session.get("session_id") or "")),
        },
    )
    return body.encode("utf-8")



def _subject_mask_session_html(session: Mapping[str, object]) -> bytes:
    safe_title = html.escape(_task_title(session))
    session_id = html.escape(str(session.get("session_id") or ""))
    status_banner = _session_status_banner(session)
    return_links = _session_return_links_html(session)
    body = render_labeling_template(
        "sessions/subject_mask.html.j2",
        {
            "safe_title": safe_title,
            "session_id": session_id,
            "return_links": return_links,
            "status_banner": status_banner,
            "session_operator_support_css": _SESSION_OPERATOR_SUPPORT_CSS,
            "session_operator_support_html": _SESSION_OPERATOR_SUPPORT_HTML,
            "operator_support_js": _SESSION_OPERATOR_SUPPORT_JS,
            "browser_mutation_status_js": _BROWSER_MUTATION_STATUS_JS,
            "image_canvas_viewport_js": _IMAGE_CANVAS_VIEWPORT_JS,
            "subject_mask_editor_js": read_labeling_asset("static/js/subject_mask_editor.js"),
            "session_id_json": json.dumps(str(session.get("session_id") or "")),
        },
    )
    return body.encode("utf-8")



def _session_html(session: Mapping[str, object]) -> bytes:
    workflow_kind = str(session.get("workflow_kind") or "")
    if workflow_kind == "keypoints":
        return _keypoint_session_html(session)
    if workflow_kind == "detect_training":
        return _detect_session_html(session)
    if workflow_kind == "detect_analysis":
        return _video_detect_session_html(session)
    if workflow_kind == "subject_mask_component":
        return _subject_mask_session_html(session)
    title = _task_title(session)
    safe_title = html.escape(title)
    status_banner = _session_status_banner(session)
    personal_queue_url = html.escape(_session_return_url(session, PERSONAL_DATASET_QUEUE_PATH))
    personal_work_url = html.escape(_session_return_url(session, PERSONAL_WORK_PATH))
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #f6f0dc, #eaf3ef);
      color: #17201a;
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }}
    article {{
      max-width: 760px;
      margin: 24px;
      padding: 28px;
      border-radius: 24px;
      background: rgba(255, 253, 245, 0.9);
      box-shadow: 0 18px 48px rgba(23, 32, 26, 0.14);
      border: 1px solid #d7ded5;
    }}
    h1 {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2rem, 6vw, 4rem);
      line-height: 0.95;
      margin: 0 0 16px;
      letter-spacing: -0.05em;
    }}
    code {{
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <article>
    <h1>{safe_title}</h1>
    {status_banner}
    <p><b>No browser editor is configured for this workflow.</b> Return to <a href="{personal_queue_url}">your personalized dataset queue</a> or <a href="{personal_work_url}">your personalized work dashboard</a> and ask the operator to inspect this task definition before doing any work.</p>
    <p><a href="/">Return to your labeling landing page</a></p>
    <p><a href="{html.escape(DASHBOARD_PATH)}">Return to the work dashboard</a></p>
    <p><b>Recording:</b> <code>{html.escape(str(session.get("recording_id") or ""))}</code></p>
    <p><b>Workflow:</b> <code>{html.escape(str(session.get("workflow_kind") or ""))}</code></p>
    <p><b>Task:</b> <code>{html.escape(str(session.get("task_id") or ""))}</code></p>
    <p><b>Session:</b> <code>{html.escape(str(session.get("session_id") or ""))}</code></p>
  </article>
</body>
</html>
"""
    return body.encode("utf-8")
