# Web Labeling Modularization Plan

## Decision

Do not migrate the Palette web-labeling workflow directly to Django or Flask as the next step.

The current problem is not primarily the HTTP server. The main issue is that `src/fisheye/labeling/web.py` mixes routing, authorization, HTML templates, browser JavaScript, admin pages, operator validation, task generation, and Zarr mutation APIs in one large module. A framework would improve some routing/template ergonomics, but it would also create migration risk around the parts that matter most: assignment ownership, active sessions, target-token checks, browser mutation authorization, and server-owned Zarr writes.

The preferred path is incremental:

1. Extract templates, JavaScript, and CSS from Python strings.
2. Continue splitting pure backend helpers into focused modules.
3. Convert route handlers into standalone handler functions.
4. Only then consider replacing `ThreadingHTTPServer` with Flask, FastAPI, Starlette, or Django.

## Framework stance

### Avoid Django for now

Django is appropriate if this becomes a formal internal product with Django ORM models, migrations, built-in admin, institutional auth integration, and broader user-management needs. Today it would be a larger rewrite than necessary because Palette already has:

- a labeling SQLite sidecar store,
- assignment/session/task semantics,
- registry integration,
- custom Zarr mutation rules,
- direct operator validation and launch evidence workflows.

### Flask is plausible later, but not first

Flask would make route registration and template rendering cleaner. It does not automatically solve the hardest parts: mutation authorization, target-token enforcement, stale session handling, or active assignment checks. Flask becomes a safer option after templates and route logic are already separated.

### Best near-term fit

Use `Jinja2` templates plus the current server first. This removes the largest maintainability problem without changing deployment, auth, or mutation semantics.

Later, consider:

- `FastAPI` or `Starlette` if API structure, middleware, async/static serving, and typed routes become important.
- `Flask` if we want a minimal WSGI-style app with familiar route decorators.
- `Django` only if this becomes a larger multi-lab web product with formal admin/auth requirements.

## Guardrails

- Keep browser mutation authorization and target-token checks together until route extraction is deliberate.
- Do not move Zarr write boundaries casually.
- Preserve labeler redaction rules: browser payloads must not expose raw Zarr paths unless explicitly allowed by policy.
- Preserve active-assignment checks before task open/save/complete.
- Preserve server-owned write semantics: browser submits edits, server mutates task-scoped training Zarr targets.
- Keep current server runnable throughout; each phase should be deployable independently.

## Current extracted modules

These slices have already been moved out of `web.py`:

- `src/fisheye/labeling/admin_registry.py`
- `src/fisheye/labeling/notification_events.py`
- `src/fisheye/labeling/notifications.py`
- `src/fisheye/labeling/report_io.py`
- `src/fisheye/labeling/task_generation.py`
- `src/fisheye/labeling/web_auth.py`

These are good precedents: low-risk helper extraction, no route rewrite, no change to Zarr write semantics.

## Phase 1: finish backend helper extraction

Goal: reduce `web.py` by moving pure helpers and policy payload builders into modules while keeping the current server and route behavior intact.

### 1. Runtime/session helpers

Candidate target: `src/fisheye/labeling/web_runtimes.py`

Current slices:

- `_keypoint_runtime_state`
- `_refresh_keypoint_queue`
- `_advance_keypoint`
- `_get_keypoint_runtime`
- `_detect_runtime_state`
- `_detect_bbox_size_hint_payload`
- `_get_detect_runtime`
- `_video_detect_runtime_state`
- `_get_video_detect_runtime`
- `_video_detect_frame_payload`
- `_subject_mask_runtime_state`
- `_get_subject_mask_runtime`
- `_subject_mask_current_payload`
- subject-mask helpers: target run path, rowset path, edit revision, row identity, checkpoint mask, unapplied checkpoint count, component review state, completion guard

Risk notes:

- These helpers touch live runtime objects and Zarr-backed backend modules.
- Keep mutation handlers in `web.py` for now.
- Avoid circular imports by passing dependencies explicitly or colocating dataclasses in the new module.

### 2. Browser mutation and task-open policy helpers

Candidate target: `src/fisheye/labeling/web_policy.py`

Current slices:

- `_task_open_preflight_error`
- `_task_open_authorization_contract`
- `_task_open_response_metadata`
- `_task_open_failure_metadata`
- `_task_completion_authorization_contract`
- `_task_completion_failure_metadata`
- `_browser_mutation_response_metadata`
- `_browser_mutation_write_policy`
- `_browser_mutation_write_contract_policy`
- `_browser_mutation_write_runtime_checklist`
- `_browser_mutation_failure_metadata`
- `_browser_mutation_target_contract_policy`
- `_browser_workflow_scope_contract_policy`
- `_browser_signed_link_policy`
- `_signed_link_contract_policy`
- `_browser_response_security_contract_policy`
- `_session_guard_policy`

Risk notes:

- This is security-sensitive.
- Move only pure payload/contract builders first.
- Do not split actual enforcement order until route handlers are extracted.

### 3. Labeler queue/work dashboard helpers

Candidate target: `src/fisheye/labeling/work_queue.py`

Current slices:

- `_work_empty_state`
- `_work_progress_summary`
- `_work_dataset_queue_task`
- `_work_dataset_queue`
- `_work_dataset_queue_summary`
- `_first_dataset_queue_url`
- `_add_work_summary_fields`
- `_dataset_queue_labeler_start_fields`
- `_labeler_work_completion_contract`
- `_dataset_queue_state`
- `_dashboard_dataset_queue_counts`
- `_dataset_queue_start_readiness_from_counts`
- `_assignment_operator_status_rows`

Risk notes:

- These mostly shape read payloads.
- Keep route methods and authorization checks in place until phase 3.

### 4. Operator validation helpers

Candidate target: `src/fisheye/labeling/operator_validation.py`

Current slices:

- `_runtime_operator_validation_start_gate`
- `_runtime_operator_validation_mutation_gate`
- `_operator_validation_command_templates`
- `_operator_validation_command_template_fields`
- `_dashboard_operator_validation_fields`
- `_operator_validation_public_fields`
- `_operator_validation_gate_flat_fields`
- `_operator_validation_visibility_fields`
- `_browser_smoke_evidence_template`
- `_browser_response_security_evidence_template`
- evidence record helpers for identity/source/browser smoke/response security

Risk notes:

- These are policy/reporting heavy, but mostly data shaping.
- Keep CLI command registration in `web.py` initially.

### 5. Handoff/report generation helpers

Candidate targets:

- `src/fisheye/labeling/handoff_reports.py`
- `src/fisheye/labeling/launch_evidence.py`

Current slices:

- `_write_user_handoffs_html_index`
- `_write_user_handoffs_readme`
- `_write_user_handoffs_roster_csv`
- `_write_web_labeling_validation_log`
- `_web_labeling_validation_checklist_payload`
- `_write_web_labeling_validation_checklist`
- `_refresh_user_handoff_visible_files`
- `_write_launch_bundle_operator_evidence_commands`
- `_write_launch_bundle_html_index`
- `_inspect_handoff_package`
- `_write_user_handoff_html_index`
- `_write_user_handoff_quickstart`
- `_write_user_handoff_message`
- `_write_user_handoff_bundle`

Risk notes:

- These are large but operationally separate from live browser mutation.
- They are good extraction candidates after queue/policy helpers.

## Phase 2: extract templates, CSS, and browser JavaScript

Goal: remove large inline HTML/JS/CSS strings from `web.py` without changing routes.

Candidate structure:

```text
src/fisheye/labeling/templates/
  base.html.j2
  identity_probe.html.j2
  browser_error.html.j2
  dashboard.html.j2
  datasets.html.j2
  admin.html.j2
  admin_datasets.html.j2
  admin_recording.html.j2
  admin_task.html.j2
  admin_users.html.j2
  admin_user.html.j2
  sessions/keypoint.html.j2
  sessions/detect.html.j2
  sessions/video_detect.html.j2
  sessions/subject_mask.html.j2
  sessions/unsupported.html.j2

src/fisheye/labeling/static/
  css/labeling.css
  js/operator_support.js
  js/browser_mutation_status.js
  js/image_canvas_viewport.js
  js/keypoint_editor.js
  js/detect_editor.js
  js/video_detect_editor.js
  js/subject_mask_editor.js
  js/dashboard.js
  js/admin.js
```

### 1. Shared browser assets

Current slices:

- `_SESSION_OPERATOR_SUPPORT_CSS`
- `_SESSION_OPERATOR_SUPPORT_HTML`
- `_SESSION_OPERATOR_SUPPORT_JS`
- `_BROWSER_MUTATION_STATUS_JS`
- `_IMAGE_CANVAS_VIEWPORT_JS`

Recommended first extraction:

- Move JS bodies to `static/js/*.js`.
- Keep a tiny Python helper to inline or serve assets.
- Initially inline the file contents into templates to avoid static-file route changes.

### 2. Browser editor session templates

Current slices:

- `_keypoint_session_html`
- `_detect_session_html`
- `_video_detect_session_html`
- `_subject_mask_session_html`
- `_session_html`

Recommended order:

1. `detect` editor first because it is the smallest active editor.
2. `keypoint` editor next.
3. `subject_mask` editor next because it has the largest JS surface and recent performance changes.
4. `video_detect` editor after detect semantics are stable.
5. unsupported session fallback last.

### 3. Labeler dashboard templates

Current slices:

- `_dashboard_html`
- `_datasets_html`
- `_identity_probe_html`
- `_browser_error_html`

Recommended order:

1. `_browser_error_html`
2. `_identity_probe_html`
3. `_dashboard_html`
4. `_datasets_html`

### 4. Admin templates

Current slices:

- `_admin_datasets_html`
- `_admin_html`
- `_admin_recording_html`
- `_admin_task_html`
- `_admin_users_html`
- `_admin_user_html`

Recommended order:

1. `_admin_task_html`
2. `_admin_user_html`
3. `_admin_users_html`
4. `_admin_recording_html`
5. `_admin_datasets_html`
6. `_admin_html`

### 5. Generated report templates

Current slices:

- `_write_batch_plan_html_report`
- `_dashboard_roster_html`
- `_write_user_handoffs_html_index`
- `_write_launch_bundle_html_index`
- `_write_user_handoff_html_index`

These are lower priority than live browser/editor templates because they do not affect day-to-day labeling latency or editor usability.

## Phase 3: extract route handlers

Goal: move route logic out of the nested `LabelingWorkHandler` class while preserving exact authorization and mutation order.

Candidate structure:

```text
src/fisheye/labeling/routes/
  public.py
  labeler.py
  admin.py
  sessions_keypoints.py
  sessions_detect.py
  sessions_video_detect.py
  sessions_subject_masks.py
  api_tasks.py
  operator_validation.py
```

Initial approach:

- Keep `ThreadingHTTPServer` and `BaseHTTPRequestHandler`.
- Make route functions accept a small adapter object containing request, state, user, body, and response writer callbacks.
- Do not introduce Flask/FastAPI yet.

High-risk routes to keep together until explicitly extracted:

- task open route
- session creation route
- `/api/sessions/{session_id}/.../save`
- `/api/sessions/{session_id}/.../apply`
- `/api/sessions/{session_id}/complete`
- direct browser start route
- assignment/reassignment mutation routes

Low-risk route candidates:

- health/status pages
- read-only admin JSON payloads
- static/read-only dashboard pages
- CLI-generated HTML reports

## Phase 4: introduce a framework wrapper if still useful

Only after phases 1-3 should we evaluate framework migration.

### Flask option

Pros:

- minimal route decorators,
- simple Jinja integration,
- low conceptual overhead,
- easy transition from current sync handlers.

Cons:

- auth/session/mutation safety remains custom,
- less structured API typing,
- deployment still needs WSGI/process management decisions.

### FastAPI/Starlette option

Pros:

- cleaner API schemas,
- ASGI middleware/static support,
- better long-term separation of API and HTML surfaces,
- easier future proxy/header-auth integration.

Cons:

- async/sync boundary must be handled carefully for Zarr and SQLite writes,
- more moving parts than Flask.

### Django option

Pros:

- built-in admin/auth/ORM/migrations,
- good if this becomes a larger internal product.

Cons:

- major rewrite,
- duplicates/replaces existing SQLite store semantics,
- higher risk for current labeling workflows.

## Phase 5: product hardening

After modularization, consider product-level improvements:

- explicit static asset versioning,
- browser cache policy for editor JS/CSS,
- structured API response schemas,
- per-workflow frontend smoke tests,
- real auth-header proxy integration,
- user/assignment admin improvements,
- operational dashboard for active sessions, pending checkpoints, and completed assignments.

## Suggested first goal

A good bounded first goal is:

> Extract shared browser assets and the detection editor template from `web.py` into template/static files without changing route behavior.

Why this first:

- It directly addresses inline HTML/JS boilerplate.
- Detection editor is smaller than keypoints/subject masks.
- It exercises the asset/template pattern before touching the most complex editor.
- It avoids mutation authorization changes.

Concrete files for that goal:

- add `src/fisheye/labeling/templates/sessions/detect.html.j2`
- add `src/fisheye/labeling/static/js/image_canvas_viewport.js`
- add `src/fisheye/labeling/static/js/operator_support.js`
- add `src/fisheye/labeling/static/js/browser_mutation_status.js`
- add `src/fisheye/labeling/static/js/detect_editor.js`
- add `src/fisheye/labeling/static/css/session_editor.css`
- add a small renderer/helper module, for example `src/fisheye/labeling/templates.py`
- update `_detect_session_html` in `web.py` to call the renderer

Acceptance criteria:

- Detection session HTML is rendered from a template file.
- Detection editor JS is no longer embedded directly in `web.py`.
- The route path and API endpoints are unchanged.
- Existing detection save/navigation behavior is unchanged.
- No auth, assignment, target-token, or Zarr mutation logic moves in this goal.

## Goal-ready modularization path

This section turns the framework decision into an implementation inventory. The path is to keep the current `ThreadingHTTPServer` application running while carving out stable seams. A framework migration should be a final adapter step, not the first refactor.

### Operating principle

The app should become framework-ready before it becomes framework-backed. Each phase should move code out of `web.py` without changing route behavior, authorization order, assignment ownership, or Zarr mutation semantics.

### Phase A: browser assets and session templates

Purpose: remove the highest-churn HTML, CSS, and JavaScript from Python string literals while preserving exact routes and responses.

Current source slices:

- `web.py`: `_SESSION_OPERATOR_SUPPORT_CSS`
- `web.py`: `_SESSION_OPERATOR_SUPPORT_HTML`
- `web.py`: `_SESSION_OPERATOR_SUPPORT_JS`
- `web.py`: `_BROWSER_MUTATION_STATUS_JS`
- `web.py`: `_IMAGE_CANVAS_VIEWPORT_JS`
- `web.py`: `_detect_session_html`
- Follow-on session renderers: `_keypoint_session_html`, `_video_detect_session_html`, `_subject_mask_session_html`, `_session_html`

Target files:

- `src/fisheye/labeling/templates.py`
- `src/fisheye/labeling/templates/partials/session_operator_support.html`
- `src/fisheye/labeling/templates/sessions/detect.html.j2`
- `src/fisheye/labeling/templates/sessions/keypoint.html.j2`
- `src/fisheye/labeling/templates/sessions/video_detect.html.j2`
- `src/fisheye/labeling/templates/sessions/subject_mask.html.j2`
- `src/fisheye/labeling/static/css/session_operator_support.css`
- `src/fisheye/labeling/static/js/operator_support.js`
- `src/fisheye/labeling/static/js/browser_mutation_status.js`
- `src/fisheye/labeling/static/js/image_canvas_viewport.js`
- `src/fisheye/labeling/static/js/detect_editor.js`
- `src/fisheye/labeling/static/js/keypoint_editor.js`
- `src/fisheye/labeling/static/js/video_detect_editor.js`
- `src/fisheye/labeling/static/js/subject_mask_editor.js`

First safe slice:

- Extract shared browser assets.
- Extract only the detection session template and detection editor JavaScript.
- Inline loaded assets into the rendered HTML initially, so no new static-file route is required.

Acceptance checks:

- Detection review opens from the same task URL.
- Response still includes operator support, mutation status reporting, and shared pan/zoom viewport behavior.
- No authorization, task-open, save, apply, or completion code moves in this phase.

Risk level: low if limited to render assembly; medium if JavaScript behavior is hand-edited during extraction.

### Phase B: pure policy and contract builders

Purpose: separate payload-building from enforcement while keeping security-sensitive checks in the current route order.

Current source slices:

- `web.py`: `_task_open_authorization_contract`
- `web.py`: `_task_open_response_metadata`
- `web.py`: `_task_open_failure_metadata`
- `web.py`: `_task_completion_authorization_contract`
- `web.py`: `_task_completion_failure_metadata`
- `web.py`: `_browser_mutation_response_metadata`
- `web.py`: `_browser_mutation_write_policy`
- `web.py`: `_browser_mutation_write_contract_policy`
- `web.py`: `_browser_mutation_write_runtime_checklist`
- `web.py`: `_browser_mutation_failure_metadata`
- `web.py`: `_browser_mutation_target_contract_policy`
- `web.py`: `_browser_workflow_scope_contract_policy`
- `web.py`: `_browser_signed_link_policy`
- `web.py`: `_signed_link_contract_policy`
- `web.py`: `_browser_response_security_contract_policy`
- `web.py`: `_session_guard_policy`

Target files:

- `src/fisheye/labeling/web_policy.py`
- Optional later split: `src/fisheye/labeling/web_contracts.py`

First safe slice:

- Move only deterministic dictionary builders and summary functions.
- Leave preflight enforcement and mutation handlers in `web.py`.

Acceptance checks:

- Unknown-user, wrong-user, stale-session, missing-target-token, and stale-edit-revision errors return the same public contract fields.
- Labeler-visible payload redaction is unchanged.

Risk level: medium because these fields are part of the safety surface, even when they are only reports.

### Phase C: runtime/session state helpers

Purpose: isolate editor runtime state loading and payload construction from HTTP handler code.

Current source slices:

- `web.py`: `_keypoint_runtime_state`
- `web.py`: `_refresh_keypoint_queue`
- `web.py`: `_advance_keypoint`
- `web.py`: `_get_keypoint_runtime`
- `web.py`: `_detect_runtime_state`
- `web.py`: `_detect_bbox_size_hint_payload`
- `web.py`: `_get_detect_runtime`
- `web.py`: `_video_detect_runtime_state`
- `web.py`: `_get_video_detect_runtime`
- `web.py`: `_video_detect_frame_payload`
- `web.py`: `_subject_mask_runtime_state`
- `web.py`: `_get_subject_mask_runtime`
- `web.py`: `_subject_mask_current_payload`
- `web.py`: `_subject_mask_target_run_path`
- `web.py`: `_subject_mask_source_rowset_path`
- `web.py`: `_subject_mask_edit_revision`
- `web.py`: `_subject_mask_row_identity`
- `web.py`: `_subject_mask_checkpoint_mask`
- `web.py`: `_subject_mask_unapplied_checkpoint_count`
- `web.py`: `_subject_mask_component_review_state`
- `web.py`: `_subject_mask_component_completion_guard`

Target files:

- `src/fisheye/labeling/web_runtimes.py`
- Optional later split: `src/fisheye/labeling/subject_mask_sessions.py`

First safe slice:

- Move read-only runtime-state payload builders before moving runtime creation or mutation helpers.
- Pass backend modules and store objects explicitly to avoid hidden globals and circular imports.

Acceptance checks:

- Keypoint, detection, video detection, and subject-mask editors load the same first ROI/frame/task payloads.
- Subject-mask checkpoint/apply semantics remain unchanged.

Risk level: medium-high because these helpers touch editor state and Zarr-backed runtime objects.

### Phase D: labeler queue and admin dashboard shaping

Purpose: make the admin and labeler views more maintainable without changing the assignment database or route layout.

Current source slices:

- `web.py`: `_work_empty_state`
- `web.py`: `_work_progress_summary`
- `web.py`: `_work_dataset_queue_task`
- `web.py`: `_work_dataset_queue`
- `web.py`: `_work_dataset_queue_summary`
- `web.py`: `_first_dataset_queue_url`
- `web.py`: `_add_work_summary_fields`
- `web.py`: `_dataset_queue_labeler_start_fields`
- `web.py`: `_labeler_work_completion_contract`
- `web.py`: `_labeler_work_completion_fields`
- `web.py`: `_dataset_queue_state`
- `web.py`: `_dashboard_roster_rows`
- `web.py`: `_dashboard_dataset_queue_counts`
- `web.py`: `_dashboard_status_report`
- `web.py`: `_assignment_operator_status_rows`

Target files:

- `src/fisheye/labeling/work_queue.py`
- `src/fisheye/labeling/admin_dashboard.py`

First safe slice:

- Move queue/task row shaping first.
- Keep route handlers and database reads in `web.py` until the data contracts are stable.

Acceptance checks:

- `/my-work`, `/my-datasets`, and admin dashboard counts agree with the current implementation.
- Blocked-recording reasons and assignment ownership fields are preserved.

Risk level: medium.

### Phase E: operator validation and handoff/report generation

Purpose: remove operational-report code from the live server module.

Current source slices:

- `web.py`: `_runtime_operator_validation_start_gate`
- `web.py`: `_runtime_operator_validation_mutation_gate`
- `web.py`: `_operator_validation_command_templates`
- `web.py`: `_operator_validation_command_template_fields`
- `web.py`: `_dashboard_operator_validation_fields`
- `web.py`: `_operator_validation_public_fields`
- `web.py`: `_operator_validation_gate_flat_fields`
- `web.py`: `_operator_validation_visibility_fields`
- `web.py`: `_write_user_handoffs_html_index`
- `web.py`: `_write_user_handoffs_readme`
- `web.py`: `_write_user_handoffs_roster_csv`
- `web.py`: `_write_web_labeling_validation_log`
- `web.py`: `_web_labeling_validation_checklist_payload`
- `web.py`: `_write_web_labeling_validation_checklist`
- `web.py`: `_refresh_user_handoff_visible_files`
- `web.py`: `_write_launch_bundle_operator_evidence_commands`
- `web.py`: `_write_launch_bundle_html_index`
- `web.py`: `_inspect_handoff_package`
- `web.py`: `_write_user_handoff_html_index`
- `web.py`: `_write_user_handoff_quickstart`
- `web.py`: `_write_user_handoff_message`
- `web.py`: `_write_user_handoff_bundle`

Target files:

- `src/fisheye/labeling/operator_validation.py`
- `src/fisheye/labeling/handoff_reports.py`
- `src/fisheye/labeling/launch_evidence.py`

First safe slice:

- Move report writers and evidence-template builders that are not called during ordinary browser save/apply operations.

Acceptance checks:

- Launch bundle generation still writes the same manifest, checklist, validation log, and handoff files.
- Labeler-visible handoff payloads still exclude operator-only fields.

Risk level: low-medium for file writers; medium for operator validation gates.

### Phase F: route-handler extraction

Purpose: create framework-shaped handler functions while continuing to use the current `BaseHTTPRequestHandler` dispatch.

Current source slices:

- `web.py`: `_make_handler`
- Nested `Handler.do_GET`
- Nested `Handler.do_POST`
- Nested `Handler.do_HEAD`
- Nested API handlers for task open/save/apply/complete/admin actions

Target files:

- `src/fisheye/labeling/web_handlers.py`
- `src/fisheye/labeling/web_responses.py`
- `src/fisheye/labeling/web_routes.py`

First safe slice:

- Introduce response helpers and route functions with explicit `(state, request)` inputs.
- Keep `_make_handler` as the adapter from `BaseHTTPRequestHandler` to those functions.

Acceptance checks:

- Every existing route path and method keeps the same status codes and response content type.
- Browser mutation routes preserve the same enforcement order: same-origin, operator gate, active assignment, current session, target token, row/revision checks, then write.

Risk level: high because this is where behavior can change accidentally.

### Phase G: optional framework adapter

Purpose: switch the HTTP adapter only after templates, handlers, policy, runtime state, and persistence are already modular.

Candidate targets:

- `Flask`: smallest migration from current sync handlers.
- `FastAPI` or `Starlette`: better if typed JSON APIs, middleware, and static serving become central.
- `Django`: only if institutional auth/admin/user-management requirements become larger than the current SQLite sidecar model.

Migration condition:

- Do not start this phase until `web.py` is mostly server bootstrap plus compatibility glue.
- Do not move Zarr write semantics into framework-specific request objects.
- Keep the domain functions framework-neutral.

Acceptance checks:

- Same labeler URLs work behind the existing campus/VPN/tunnel deployment path.
- Trusted-header/fixed-user identity behavior is unchanged or explicitly replaced by a proxy-auth contract.
- Admin and labeler pages work with multiple simultaneous users under the chosen deployment model.

Risk level: high, but much lower after Phases A-F.

## Suggested next active goal

Use a narrow first goal:

> Extract shared browser assets and the detection editor template from `web.py` into template/static files without changing route behavior.

Scope for that goal:

- Add a tiny template/asset loader.
- Move operator-support CSS/HTML/JS, browser mutation status JS, and image-canvas viewport JS out of `web.py`.
- Move detection editor markup to `templates/sessions/detect.html.j2`.
- Move detection editor JavaScript to `static/js/detect_editor.js`.
- Keep assets inlined into the HTML response for now.
- Do not change any URL, API, task-open behavior, save behavior, apply behavior, assignment policy, or Zarr write code.

Defer out of that goal:

- Flask/FastAPI/Django migration.
- Static file serving routes.
- Moving keypoint/video/subject-mask editors.
- Moving mutation handlers.
- Moving task-open enforcement.
- Changing authentication or deployment behavior.
