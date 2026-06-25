# Web Labeling Deployment Runbook

<!-- design-meta
status: draft
last_updated: 2026-06-24
scope: operational guidance for assigned browser labeling service
-->

## Purpose

This runbook covers the operational pieces for the assigned web-labeling service:

- starting the service safely
- generating signed links
- backing up the sidecar assignment database
- backing up mutable per-recording training zarrs
- recovering from failed promotion or stale-session issues

## Safe Launch Checklist

Before sharing links with labelers:

1. Review `docs/web_labeling_implementation_checklist_clean.md` for the
   consolidated implementation checklist and remaining deployment decisions.
2. Review `docs/web_labeling_multi_user_workflow_checklist.md` for the
   assignment, launch, labeler, and acceptance checklist that ties the full
   workflow together.
3. For the first real multi-user batch sequence, follow
   `docs/web_labeling_first_batch_operator_checklist.md`.
4. For a local fixed-user operator pass, follow
   `docs/web_labeling_first_operator_test_plan.md`.
5. Fill out `docs/web_labeling_production_decision_record.md` for production
   use.
6. Adapt the safe deployment templates in
   `docs/web_labeling_deployment_examples.md` if you need a systemd/proxy
   starting point.
7. Start the service with `--user` only for local loopback testing, or with
   `--trust-auth-header --auth-header <name>` only behind a trusted proxy.
8. Configure at least one `--admin-user`.
9. Configure `--link-secret` or `PALETTE_LABELING_LINK_SECRET` if signed links
   will be used.
10. Leave same-origin POST protection enabled unless you have a controlled proxy
   reason to disable it.
11. Confirm the proxy strips any inbound copy of the trusted auth header before
   setting its own authenticated-user value.
12. Enable access logging at the proxy or start the service with `--access-log`.
13. Keep the service bound to `127.0.0.1` behind the proxy unless direct
   non-loopback exposure is intentional.
14. Run `check-store` and fix hard issues before sharing dashboard access or
   signed links.
15. Open `/admin` and confirm the deployment preflight has no unexpected
   warnings.
16. Confirm assigned users can see only their recordings on `/work`, and that
   the user shown on the dashboard matches the signed-in labeler.
17. Ask at least one real labeler to open
   `/identity?expected_user=<assignment-user>` and confirm the page reports an
   identity match before sending real work.
18. Spot-check `/admin/recordings/<recording_id>` for at least one assigned
   recording and confirm the owner, tasks, active sessions, and recent audit
   events match the one-user-per-recording plan.

Safe-share launch rule:

- Do not send labeler links just because a per-user handoff says
  `ready_to_send=true` or a dashboard row says `ready_to_invite=true`.
- A launch is shareable only after inspection reports
  `labeler_links_safe_to_share=true`.
- Fresh handoffs should fail closed until the safe-share evidence gates are
  complete: mutable Zarr backup confirmation, browser response-security
  headers, identity-source verification, representative browser smoke,
  disposable-Zarr mutation smoke, and operator-recovery contract evidence.
- Use the guarded personalized dataset queue as the preferred labeler entry:
  `/my-datasets?expected_user=<assignment-user>`. Treat `/datasets` as the
  canonical fallback and `/work` as the full dashboard fallback.
- Browser saves mutate the server-owned assigned task/training Zarr scope.
  CSVs, roster files, manifests, HTML, JSON, and intermediate handoff exports
  are metadata/control-plane artifacts only; they are not label-write targets.
- Each recording should have exactly one active assigned owner before links are
  shared. Reassignment must close stale previous-owner sessions before the new
  owner can mutate labels.

Example local development launch:

```bash
scripts/py -m fisheye.utils.labeling_work --store /tmp/palette_labeling.sqlite \
  serve --user alice --admin-user alice --link-secret dev-secret \
  --host 127.0.0.1 --port 8795
```

Example proxy-backed launch shape:

```bash
PALETTE_LABELING_LINK_SECRET='<secret-from-secret-store>' \
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  preflight --trust-auth-header --auth-header X-Forwarded-User \
  --admin-user admin@example.org \
  --host 127.0.0.1 --port 8795 --access-log --production
```

Example sidecar consistency check:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  check-store
```

`check-store` returns nonzero for hard safety issues such as tasks without
recording assignments or active sessions that no longer match current
assignment/task state. Warnings, such as incomplete tasks under paused
assignments or expired active sessions, are reported without failing the command.

```bash
PALETTE_LABELING_LINK_SECRET='<secret-from-secret-store>' \
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  serve --trust-auth-header --auth-header X-Forwarded-User \
  --admin-user admin@example.org \
  --host 127.0.0.1 --port 8795 --access-log --production
```

The service intentionally refuses to start in header-auth mode unless
`--trust-auth-header` is present. This keeps accidental direct exposure from
silently trusting user-supplied HTTP headers.

The service also refuses to bind to a non-loopback host unless
`--allow-non-loopback` is set. Prefer keeping the service on `127.0.0.1` and
exposing it through a TLS/auth proxy. Use `--allow-non-loopback` only when the
network boundary is intentional and separately protected.

Production launches should use `--production`. In production mode, the service
rejects local fixed-user auth and requires a trusted auth header plus at least
one admin user.

Required proxy header behavior:

1. Authenticate the browser user before forwarding to Palette.
2. Strip any inbound client-supplied copy of the trusted auth header.
3. Set exactly one trusted auth header, for example `X-Forwarded-User`, from the
   authenticated identity.
4. Preserve `Host`/`X-Forwarded-Host` consistently so same-origin POST checks see
   the browser-visible host.
5. Preserve the service's no-cache response headers:
   `Cache-Control: no-store, no-cache, must-revalidate, max-age=0`,
   `Pragma: no-cache`, and `Expires: 0`.
6. Preserve the service's browser security headers:
   `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, and
   `Referrer-Policy: no-referrer`. Also preserve the narrow
   `Content-Security-Policy` that blocks framing, restricts base/form
   destinations, and disables plugin/object embedding, plus
   `Permissions-Policy` disabling camera, microphone, and geolocation.
7. Terminate TLS or forward only across a separately protected internal network.

Generated `validation-checklist.json` files include a required
`browser_response_security_headers` gate. Before inviting labelers, capture the
deployed response headers from `/datasets` or `/api/me/tasks` as an
authenticated test labeler and attach evidence that the proxy preserved the
headers listed in `browser_response_security_policy.headers`.

## Access Logging

Use proxy access logs when the proxy terminates TLS/auth or when you need the
original client IP. The service can also emit JSON request logs to stderr with:

```bash
--access-log
```

Each service log line includes the resolved user, auth source, remote socket
address, request line, and HTTP handler message.

## Signed Links

Signed links are convenience links, not standalone authorization grants.

They resolve to a task id, then the server still requires the current
authenticated user to be the active assignee for the task recording.
Dashboard invitation links may include `expected_user=USER`; this is also not
an authorization grant. It only fails closed when the authenticated browser user
does not match the intended labeler, preventing a forwarded or mis-opened
dashboard link from showing another user's work queue.
`/work` is the canonical labeler dashboard. `/datasets` is a dedicated
lightweight, authenticated, expected-user-guarded queue page for labelers or
operators who want to start from the personalized datasets waiting for
completion list before opening the full work dashboard.
Task-specific signed links are short-lived convenience links rather than
authorization grants. They still require an authenticated user, active
assignment, and open task before the server creates a guarded session. Newly
generated signed task links also bind the intended `expected_user` inside the
signed payload; a copied link opened by a different authenticated browser fails
with `signed_link_user_mismatch` before session creation. After a task link
opens, the browser works through that session; `/work` remains the canonical
entry point for multi-task labeling.
The live admin/preflight payload exposes `identity_source_policy`, including
`assignment_user_source`, `auth_header`, `assignment_user_match_required`,
`labeler_landing_page_path`, `queue_first_landing_paths`,
`queue_first_landing_expected_user_guard_supported`,
`dashboard_expected_user_guard_supported`,
`dataset_queue_page_expected_user_guard_supported`,
`personal_work_expected_user_guard_supported`,
`dataset_queue_expected_user_guard_supported`, `task_open_expected_user_guard_supported`,
`task_complete_expected_user_guard_supported`,
`promotion_retry_expected_user_guard_supported`,
`promotion_retry_current_session_required`,
`promotion_retry_dashboard_action`,
`signed_task_link_expected_user_binding_supported`, `expected_user_guards`,
`signed_links_are_not_identity`, and `production_ready`. Before inviting
labelers, confirm the deployed reverse proxy or auth layer resolves browser
users to the same IDs stored as `assignee_user`; neither signed task links nor
expected-user query parameters are authorization grants.

The live admin/preflight payload also exposes `operator_authorization_policy`.
Treat the deployment as not ready for broad use unless
`admin_routes_require_operator=true`, `admin_users_configured=true`, and
`operator_boundary_ready=true`. Operator routes are separate from labeler
routes: a resolved labeler can open their queue and sessions, but `/admin` and
`/api/admin/*` require the resolved user to be listed with `--admin-user`.
Failed-promotion retry is part of this operator recovery boundary through
`/api/admin/events/{event_id}/retry-promotion`.
Stale-tab and blocked-session recovery also has an operator-only inspection
route: `/api/admin/sessions/{session_id}/closure` returns the redacted session
summary and latest closure event, if one was recorded.
The `/admin` preflight card displays this operator-boundary state directly so
operators do not have to inspect raw JSON before launch.
The live `/api/admin/summary` payload and `/admin` page also expose
`dataset_queue_start_readiness`, derived from the assigned-user dashboard roster,
so operators can see users whose dataset queue blocks labeler start before
generating handoff artifacts.
Generated `validation-checklist.json` files include a required
`operator_authorization_boundary` gate; attach the runtime preflight output and
a non-admin `/admin` rejection smoke as evidence before inviting labelers.
Static handoff artifacts may report `operator_boundary_known=false` and
`runtime_preflight_required=true`; that is expected until the operator attaches
the deployed preflight evidence.
Runtime/admin payloads and generated handoff artifacts expose
`zarr_backup_policy` as safe metadata. This policy identifies the backup
validation gate and operator-owned rollback requirements without giving
labelers raw backup paths; raw Zarr paths remain confined to operator backup
plans such as `zarr-backup-plan.json`.
Those operator backup plans summarize mutable Zarr targets by role, for example
`training` versus `analysis`, so operators can confirm which training Zarrs are
in the write/backup set without exposing those paths to labelers.
They also expose `mutation_audit_policy`, which states that browser mutations
are recorded server-side in `labeling_task_events` with append-only audit
semantics and no browser-side audit-store write credentials.
They expose `browser_mutation_write_policy` as well: handoff CSV, HTML, and
JSON files are metadata-only, while browser saves mutate server-owned Zarr
targets from the assigned task scope and append audit events.
They also expose `labeler_route_authorization_policy`, which records that
personal queues require the resolved browser user to exist in the assignment
store, unknown users are rejected before queue data is returned, forwarded
`expected_user` links are rechecked against identity, task opens require active
assignment and open task state, mutations require the current session and target
token, and signed links are entry hints rather than authorization grants.
Task and session authorization failures include a redacted
`authorization_context` support object with task, recording, assignee,
assignment, state, session, and current-session identifiers when available, so
labelers can copy support details without exposing Zarr paths or credentials.
They expose `session_guard_policy` as well, documenting that only current,
unexpired sessions can save, stale tabs are rejected, and stale-session errors
include `session_closure_event` support metadata when available.
`expected_user` dashboard guards provide identity.
When a labeler or operator opens a stale, unauthorized, revoked, or
unauthenticated browser entry URL, the service shows a human-readable
access-problem page with return links to the queue-first `/` landing page and
the full `/work` dashboard, plus a copyable error/status/details block for
operator support; API routes continue to return JSON errors for automation.
In-session editor load, save, review, navigation, and completion failures also
show copyable operator support details and root-landing recovery links instead
of leaving the labeler with only a terse JavaScript status message.
When a stale tab is rejected after reassignment, task completion, supersession,
stale cleanup, or manual close, session API errors include safe closure-event
metadata so operators can distinguish the reason without exposing storage
paths.
The lightweight `/datasets` queue page carries the same identity warning
posture, links directly to the guarded root landing page and identity probe,
provides a copyable start link, and shows copyable operator-support details if
the queue API fails after the page loads. It also shows a copyable safe
backup-policy block that identifies the required backup validation gate without
exposing backup paths or raw Zarr paths.
When no open dataset work is waiting, the page shows a copyable queue-state
support block that distinguishes no active assignments, all assigned work
complete, and assigned recordings that need operator action before more labeling.
The page also shows an always-visible queue-state panel with the stable state
code and whether labeler start is allowed or blocked, so labelers do not have to
infer launchability from task counts.
The live and exported queue payloads expose the same decision as
`dataset_queue_state` so lightweight clients do not have to infer it from
summary counts.
It renders dataset, recording, and task-level queue entries, but task links
still route through guarded `task_id`-filtered `/work` views so session creation
and mutation remain centralized in the main browser labeling dashboard. Dataset,
recording, and task rows include copy buttons for redacted support details so
labelers can send exact support context to the
operator. `/api/me/datasets` also includes redacted `operator_support` blocks
for dataset, recording, and task rows with safe IDs, counts, workflow state, and
guarded work URLs but no task scope or Zarr paths. Labeler-facing payloads
expose `labeler_safety.labeler_api_redaction` so operators can confirm browsers
do not receive task scope, raw Zarr paths, direct storage paths, or write
credentials. Runtime/admin preflight, labeler payloads, and generated handoff
artifacts expose `browser_response_security_policy` so operators can confirm the
no-store/no-cache, clickjacking, MIME-sniffing, referrer, CSP, and permissions
headers that the proxy must preserve. If a guarded `/work` filter link matches no assigned tasks, `/work`
shows copyable filter support details instead of a bare empty state. Its open-work cards
come from `dataset_queue_summary`; its completed-task card uses the user's
overall `progress_summary` because completed tasks are intentionally omitted
from the open dataset queue. The full `/work` dashboard also exposes a guarded
root landing link and copyable start-link control, matching the lightweight
queue page. The page also surfaces blocked/no-open assignment
state from `progress_summary` so assigned recordings that need operator action
do not disappear from the queue-first view, with copyable blocked-state details
for operator support.
Generated handoff HTML, quickstart, and message files tell labelers to confirm
the dashboard shows their expected authenticated user before opening work. They
also print the per-user `dataset_queue_state` and whether labeler start is
allowed or blocked, so a copied handoff remains clear even before the live queue
page loads.
Handoff and launch manifests also expose this as `labeler_safety` metadata so
operator tooling can audit the policy without scraping prose artifacts. The
`labeler_safety.expected_user_guards` map lists the expected-user-guarded
browser/API entry points and their mismatch error codes. `labeler_safety` also
includes `labeler_landing_page_path=/`, `queue_first_landing_paths=["/",
"/me", "/datasets"]`, and `dataset_queue_page_path=/datasets` for tools that
want a lightweight personal queue entry point rather than the full canonical
dashboard path. `work_filter_query_keys` documents the supported `/work` filters:
`expected_user`, `dataset_id`, `recording_id`, `task_id`, and `workflow`.
Failed-promotion retry is reported as
`labeler_failed_promotion_retry_action=operator_support_only`; labeler
dashboards expose redacted support context, and operators retry from
`/api/admin/events/{event_id}/retry-promotion` after repair.
Server, roster, work-summary, handoff, and launch payloads expose the link
decision as `signed_link_policy`: `task_specific_links` is
`short_lived_convenience_links`, `authorization_grant=false`,
`requires_authenticated_user=true`, `requires_active_assignment=true`,
`requires_open_task=true`, `binds_expected_user_in_new_links=true`,
`expected_user_mismatch_error=signed_link_user_mismatch`,
`opens_guarded_session=true`, and `dashboard_preferred_for_multi_task_work=true`.
Generated handoff and launch manifests/indexes also carry
`browser_workflows` and `task_state_policy`, so a copied or archived launch
bundle remains self-describing without querying the live server.
Launch bundles include `validation-log-template.md`, prefilled with batch
metadata and validation sections for static checks, browser smoke, real-zarr
smoke, one-labeler dry run, multi-user dry run, rollback drill, and final
operator sign-off. They also include `validation-checklist.json`, a
machine-readable gate list that keeps generated readiness separate from
operator-only evidence. The checklist records the queue-first entry metadata
as `labeler_landing_page_path`, `labeler_landing_url`,
`expected_user_labeler_landing_url`, `dataset_queue_page_path`,
`dataset_queue_url`, and, for per-user handoffs,
`expected_user_dataset_queue_url` when available.
Not-ready handoffs include `sendability_actions` alongside
`sendability_reasons`, so missing base URLs, missing tasks, failed store checks,
and unshareable links have concrete operator repair steps.
Those repair actions are also printed into not-ready handoff HTML, quickstart,
and message files to make review artifacts actionable without opening JSON.
`inspect-handoff` includes the same no-open-task and sendability action metadata
when checking a copied directory or ZIP. It also verifies that
`validation-checklist.json` is present and flags generated checklist gates whose
status is `needs_review`.

Generate a link:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  sign-link --task-id <task_id> --base-url https://labeling.example.org
```

Generate a batch manifest of signed task links for one assignee:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  sign-links --user LABELER --base-url https://labeling.example.org \
  --format jsonl --output /path/to/link-manifests/LABELER-links.jsonl
```

`sign-links` includes only tasks under active recording assignments. Completed
tasks are excluded unless `--include-completed` is supplied. Signed-link output
files are not overwritten unless `--overwrite` is supplied. Prefer passing
`--base-url` for any link output that will be sent to a labeler. Without it,
`sign-link` and `sign-links` emit service-relative paths and report
`missing_base_url` shareability warnings. `sign-link` also reports whether the
single target task is currently launchable; inactive recording assignments and
completed tasks are reported as shareability warnings because the server will not
open a labeling session for them. `sign-links --include-completed` similarly
marks completed-task rows as not task-launchable even when the URL is otherwise
well formed. Batch signed-link reports include ready-to-share and
not-ready-to-share counts for quick review.

For the normal multi-user launch path, generate the current-state snapshots,
readiness report, handoff pages, package manifest, README, and optional ZIP in
one command:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-launch-bundle --dry-run --base-url https://labeling.example.org \
  --output-dir /path/to/handoffs/current-batch-launch
```

The dry run is read-only and does not require a link secret. It reports planned
users, output paths, output-directory state, readiness counts, and mutable-Zarr
backup-plan counts. It also includes a `validation_checklist` preview so
operators can see pending identity, browser-smoke, disposable-zarr, backup, and
dry-run gates before writing files. Dry-run checklist output marks generated
handoff/store-check evidence as pending because that evidence only exists after
the non-dry-run export writes the package. Add `--warnings-as-errors` when
readiness warnings should block launch automation.

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-launch-bundle --base-url https://labeling.example.org \
  --output-dir /path/to/handoffs/current-batch-launch \
  --zip-output /path/to/handoffs/current-batch-launch.zip \
  --include-audit-events
```

`export-launch-bundle` does not mutate assignments, tasks, sessions, or zarrs.
It writes `assignments.json`, `tasks.json`, `zarr-backup-plan.json`,
`batch-readiness.json`,
`handoffs/`, `manifest.json`, `index.html`, `launch-readme.txt`,
`validation-log-template.md`, `validation-checklist.json`, and `checksums.json`.
The launch `assignments.json` includes `single_owner_policy` so archived launch
plans remain explicit that each recording has one active owner and reassignment
replaces the previous owner.
The launch `manifest.json`, `assignments.json`, handoff index, README, and HTML
index also surface `assignment_ownership_integrity`, including duplicate active
owner count and unique active recording count, so multi-user ownership safety is
visible without opening the nested readiness report.
It also writes `inspect-command.txt` and
`inspection-targets.json` with read-only inspection commands to run after copying
or before re-sharing. The nested `handoffs/labeler-roster.csv` gives a
spreadsheet-friendly labeler roster for tracking users, task counts, ready-to-send
status, dashboard URLs, link expiration, message files, quickstart files,
manifests, `dataset_queue_state_code`, whether the queue blocks labeler start,
assigned recordings without open task links, and redacted user-summary field
counts. Open the top-level `index.html` first after extracting the bundle.
Use `checksums.json` to audit copied package contents. With
`--include-audit-events`, it also writes `audit/task-events.jsonl`,
`audit/assignment-events.jsonl`, and `audit/task-definition-events.jsonl`. Use
`--audit-since-utc`, `--audit-until-utc`, and `--audit-limit` to constrain audit
capture. Use the lower-level commands below when you need a custom archive
layout or want to regenerate only one artifact.

Prefer a fresh `--output-dir` for each launch. `--overwrite` refuses stale
handoff user directories rather than silently carrying removed labelers into a
new package. It also refuses stale `audit/` artifacts unless the new export uses
`--include-audit-events`.

Before announcing or handing off a batch, archive the current assignment/task
plan separately from audit events:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-assignments --format json \
  --output /path/to/handoffs/current-batch-assignments.json

scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-tasks --format json \
  --output /path/to/handoffs/current-batch-tasks.json
```

`export-assignments` and `export-tasks` are read-only current-state snapshots.
Assignment JSON snapshots include `single_owner_policy`, documenting that
`recording_id` is the ownership key, the SQLite schema enforces that ownership
key with `schema_enforced_recording_primary_key=true`, and reassignment replaces
the active owner.
The policy also records `raw_assignment_change_blocks_open_sessions=true`:
low-level assignment changes fail closed when open browser sessions would be
left stale. Operator-facing assignment routes and imports use the reassignment
transition path so stale sessions are closed and reported.
They complement audit exports by recording the batch plan at the moment links or
handoff packages are generated. Use `--format csv` when the snapshot should be
reviewed in a spreadsheet; nested task fields such as `scope` are encoded as JSON
inside the CSV cell.

Before announcing a multi-user batch, archive a readiness report:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  batch-readiness --output /path/to/handoffs/current-batch-readiness.json
```

`batch-readiness` is read-only. It reports active assignees, active recordings,
active open tasks, empty active assignments, users with no open work, active
sessions, one-owner assignment integrity, and the same hard consistency issues
surfaced by `check-store`. Add
`--warnings-as-errors` when readiness warnings should return nonzero. Strict
reports include `blocking_warning_count` and `blocking_warning_codes` for
automation-friendly triage.

For a per-labeler handoff directory, generate the work preview, signed links,
and store-check report together:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-user-handoff --user LABELER --base-url https://labeling.example.org \
  --output-dir /path/to/handoffs/LABELER \
  --zip-output /path/to/handoffs/LABELER.zip
```

The handoff directory contains:

- `work-summary.json` with labeler-visible task metadata and redacted
  server-only scope/path fields
- `signed-links.jsonl`
- `check-store.json`
- `manifest.json`
- `index.html`
- `message.txt`
- `labeler-quickstart.txt`
- `validation-log-template.md`
- `validation-checklist.json`

`export-user-handoff` returns nonzero when `check-store` reports hard safety
issues, but it still writes the files so the operator can inspect the failure
context.
Review `validation-checklist.json` and fill `validation-log-template.md` if
this single-labeler handoff is being used as the launch archive rather than
being wrapped in a multi-user handoff or launch bundle. Standalone handoffs do
not contain the top-level mutable-Zarr backup plan, so the checklist keeps the
backup/rollback gate pending until an operator references a launch-bundle
`zarr-backup-plan.json` or a separately archived `zarr-backup-plan` export.

Use `message.txt` as the concise labeler-facing text to send or adapt only after
checking `ready_to_send`. Ready handoffs say the work is ready and name the
guarded dataset queue as the preferred queue-first entry point; not-ready
handoffs tell the labeler to wait for operator review and list the sendability
reasons. If `dataset_queue_state.blocks_labeler_start=true`, sendability
includes `dataset_queue_blocks_labeler_start` so the operator resolves the queue
state before sharing a start link. Per-user manifests include compact `sendability_reasons`, verbose
`sendability_warnings`, `dataset_queue_state`, and `progress_summary` for
operator triage. It points to the queue page and dashboard when `--base-url` is
set, otherwise it tells the labeler to open their local `index.html`. Include `labeler-quickstart.txt`
when the labeler needs the browser-only safety rules: no Palette/Crimson install,
no direct zarr edits, and no forwarding handoffs. Quickstarts also reflect
`ready_to_send`: not-ready handoffs switch from start instructions to
preview/wait instructions. Sendability also requires `known_user_status`
evidence that the handoff user is a known assignment-store labeler with at least
one active assignment; missing or stale evidence is reported as
`known_user_status_missing`, `unknown_labeling_user`, or
`no_active_assignment`. Sendability also requires assignment ownership integrity
evidence showing no duplicate active owners; missing or failed evidence is
reported as `assignment_ownership_missing` or `assignment_ownership_conflict`.
Handoff manifests, link rows, HTML, messages, and
quickstarts include generated-at and link-expiration timestamps so stale
packages are easier to identify. The per-user HTML also shows active assigned
recordings that have no open task links, so task-generation gaps or completed
recordings are visible instead of disappearing from the handoff. Not-ready
per-user HTML pages warn labelers to wait for operator review and do not render
clickable task open links unless signed-link rows are ready to share. Handoff
manifests, rosters, readmes, launch bundles, and `inspect-handoff` reports also
include no-open-task reason breakdowns from both readiness and handoff summaries
so operators can separate not-yet-generated work from already-completed
assignments. Handoff manifests, rosters, and `inspect-handoff` reports preserve
`expected_user_identity_probe_url`, `expected_user_labeler_landing_url`,
`expected_user_dataset_queue_url`, and `expected_user_dashboard_url` so
operators can re-check copied packages before re-sharing. Handoff indexes and
rosters also include `dataset_queue_state_code`, `dataset_queue_state_title`,
and `dataset_queue_blocks_labeler_start` so spreadsheet workflows can separate
open work from completed or operator-blocked queues. `inspect-handoff`
also surfaces validation-checklist queue entry metadata under
`validation_checklist.labeler_landing_page_path`,
`validation_checklist.labeler_landing_url`,
`validation_checklist.expected_user_labeler_landing_url`,
`validation_checklist.dataset_queue_page_path`,
`validation_checklist.dataset_queue_url`, and
`validation_checklist.expected_user_dataset_queue_url` when those fields are
available in the inspected package. It also reports validation evidence
summaries under `validation_checklist.gates`, including per-gate
`evidence_recorded`, evidence counts, `evidence_recorded_gate_ids`, and
`required_missing_evidence_gate_ids` for required gates still marked
`pending_operator_evidence` or `needs_review`, so operators can see which gates
still need proof without opening the raw checklist JSON. All-user handoff
indexes and rosters include per-labeler sendability reasons for not-ready
handoffs; handoff and launch summaries include aggregate not-ready reason
counts. `inspect-handoff` treats not-ready labeler packages as `needs_review`
with `handoff_not_ready` before re-sharing, and its counts include aggregate
sendability reasons. In all-user handoff indexes,
top-level `ok` means the package is safe to send, while `store_checks_ok`
preserves whether the underlying per-user store checks succeeded. Launch bundles
mirror that split with sendability-focused `handoffs_ok` and structural
`handoff_store_checks_ok`; `inspect-handoff` reports the same split after copy
or ZIP transfer.

Prefer setting `--base-url` for any handoff that labelers will use directly. If
`--base-url` is omitted, the generated pages are safe previews of assigned work,
but labelers must get the service dashboard URL before opening tasks.

For a full multi-labeler batch, generate every active assignee's handoff bundle
in one pass:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-user-handoffs --base-url https://labeling.example.org \
  --output-dir /path/to/handoffs/current-batch \
  --zip-output /path/to/handoffs/current-batch.zip
```

`export-user-handoffs` writes one subdirectory per active assigned user plus
top-level `index.json`, `index.html`, `labeler-roster.csv`,
`handoff-readme.txt`, `validation-log-template.md`, and
`validation-checklist.json` files. The batch
`index.html` links to each labeler's
`index.html`, `message.txt`, `labeler-quickstart.txt`, `dataset-queue.json`,
and `manifest.json`. Each per-user `dataset-queue.json` is a self-contained,
redacted open-work queue with `dataset_queue_summary`, guarded
`expected_user_work_url` values, `expected_user_dataset_queue_url`, the
dashboard URL, identity probe URL, `empty_state`, `progress_summary`,
`dataset_queue_state`,
`datasets` and `dataset_queue` aliases, and
browser policy metadata. Use
`labeler-roster.csv` for spreadsheet-friendly tracking of users, known-labeler
status, active assignment counts, assignment ownership readiness, task counts,
ready-to-send status, dashboard URLs, link expiration, message files, quickstart
files, dataset queue files, manifests, waiting dataset counts, first guarded
dataset queue links, guarded `/datasets?expected_user=<user>` queue-page links,
`dataset_queue_state_code`, queue-blocked-start state, queue-start ready/status
and action fields, assigned recordings without open task links, and redacted
user-summary field counts. Batch handoff
and launch indexes include top-level
`dataset_queue_page_path` and `dataset_queue_url` fields alongside
`dashboard_path` and `dashboard_url`. Batch handoff and launch counts also expose
aggregate `dataset_queue_states` and `dataset_queue_blocked_start_users` for
validation checklist input. Batch handoff indexes include aggregate
`progress_summary`, and each row also carries that user's `progress_summary`,
including waiting, complete, and blocked/no-open recording counts, plus
aggregate `dataset_queue_summary` for open dataset work. Open the top-level `index.html` locally for a quick
operator-facing overview, and keep `handoff-readme.txt` with the exported
package so another operator can interpret the bundle. Review
`validation-checklist.json` for the required gates and fill
`validation-log-template.md` while running static checks, browser smoke,
real-Zarr smoke, and deployed dry runs so the handoff batch can be archived with
operator sign-off evidence. Like single-user handoffs, the multi-user handoff
bundle requires external backup-plan evidence unless it is wrapped by an
`export-launch-bundle` package that includes `zarr-backup-plan.json`. The command refuses to
overwrite existing handoff files unless `--overwrite` is supplied.

The handoff `index.json` and launch `manifest.json` also summarize
ready-to-send and not-ready-to-send counts. Inspect `sendability_warnings` before
sharing if any user has zero tasks, zero signed links, a missing service base
URL, or a failed store check.

Per-user `signed-links.jsonl` rows also include `ready_to_share`,
`task_launchable`, and `shareability_warnings`. Completed-task links are not
task-launchable unless the task is reopened.

`--zip-output` is optional for both handoff commands. When supplied, it must
point outside the generated handoff directory so the archive does not include
itself. Existing ZIP files are not overwritten unless `--overwrite` is supplied.

Before re-sending an old handoff or launch package, inspect it for freshness:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  inspect-handoff --path /path/to/handoffs/current-batch-launch.zip \
  --output /path/to/handoffs/current-batch-launch-inspection.json
```

`inspect-handoff` is read-only. It reports whether the package is fresh,
expired, missing expiration metadata, missing its validation log, missing or
invalid `validation-checklist.json`, or marked as needing review by the stored
handoff manifests or generated checklist gates. For full launch bundles, the
inspection also reports whether the stored readiness and handoff checks were ok.
Inspection handoff rows include `dataset_queue_state_code`,
`dataset_queue_blocks_labeler_start`, and flat queue-start ready/status/action
fields, and aggregate counts include
`dataset_queue_states` and `dataset_queue_blocked_start_users` for quick
copied-package triage. If any inspected handoff blocks labeler start, inspection
also includes `dataset_queue_blocks_labeler_start` in `failure_reasons` with a
repair action before the package is re-shared.
When run with an explicit global `--store`, inspection compares the package's
archived `assignment_snapshot` against that current assignment store. If any
included recording is now assigned to a different user/status, or if the current
store has active assigned work omitted from the package, inspection returns
`status=stale_assignment` with `assignment_freshness_mismatch` or
`assignment_freshness_incomplete`, so copied handoff links are not treated as
current after reassignment or assignment expansion. Without explicit `--store`,
inspection remains a static package check and reports
`assignment_freshness.status=not_checked` rather than comparing against an
unintended default store; the static path does not initialize a default
assignment database.
Required checklist gates still marked `pending_operator_evidence` produce
`validation_evidence_pending` and keep the package from inspecting as `fresh`;
record validation evidence with `update-validation-checklist` after each smoke
or dry-run gate passes.
When `checksums.json` is present, inspection verifies copied file contents and
fails if any listed file is missing or modified. Read top-level `status`,
`failure_reasons`, and `failure_actions` first for quick triage, then inspect
nested details when status is not `fresh`.

Use `--link-secret` or `PALETTE_LABELING_LINK_SECRET` consistently between link
creation and the running service.

Generated signed-link rows include token-derived `issued_at_utc`,
`expires_at_utc`, and effective `expires_in_seconds` fields. Use those values,
not local email timestamps, when deciding whether to regenerate a handoff.

New signed links include an issuance timestamp. To revoke older links without
changing assignments or task ids, restart the service with a not-before floor:

```bash
--link-not-before-utc 2026-06-23T12:00:00Z
```

or set:

```bash
PALETTE_LABELING_LINK_NOT_BEFORE_UTC=2026-06-23T12:00:00Z
```

When this floor is configured, legacy tokens that do not contain an issuance
timestamp are rejected. Tokens issued before the floor return
`signed_link_revoked` and cannot open a session.

## Sidecar Store Backup

The sidecar SQLite database stores:

- recording assignments
- task definitions and scopes
- browser sessions
- audit events

Default path:

```text
~/.palette/labeling_work.sqlite
```

Preferred backup procedure:

1. Use the `backup-store` command for SQLite-consistent backups.
2. Store backups with timestamped names.
3. If using filesystem copies instead, stop the service or use a
   SQLite-aware snapshot tool and copy SQLite sidecar files such as `-wal` and
   `-shm`.
4. Restore by stopping the service, replacing the database files, then starting
   the service again.

Example SQLite-safe backup:

```bash
backup_dir=/path/to/backups/labeling_work_$(date +%Y%m%d_%H%M%S)
mkdir -p "$backup_dir"
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  backup-store --output "$backup_dir/labeling_work.sqlite"
```

Before copying mutable per-recording Zarrs, generate the read-only operator plan
from the current task scopes:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  zarr-backup-plan --output "$backup_dir/zarr-backup-plan.json"
```

The plan groups task-scoped Zarr paths by recording, assignee, task, workflow,
dataset, and registry path. It does not copy data. Use it as the manifest for
the storage-specific copy command and keep the completed backup manifest with
the launch bundle.

## Assignment Changes

Admins can create or update recording assignments from `/admin`.

Assignment notes are shown to the assignee on the personalized work dashboard as
recording-level instructions.

The personalized dashboard also includes browser-only safety guidance for
labelers: no local Palette/Crimson install is needed, zarr files should not be
edited directly, and links or handoff files should not be forwarded.

For batch setup, import a CSV, JSON, or JSONL assignment manifest. The command
is dry-run by default:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  write-manifest-templates --output-dir /path/to/manifest-templates
```

When using `import-assignments` directly for launch-bound ownership changes, add
`--warnings-as-errors` so duplicate recording rows or reassignment warnings block
apply before active sessions are closed.
Use `--output /path/to/assignment-import-report.json` to archive the exact
dry-run or apply report; existing report files are not overwritten unless
`--overwrite` is supplied.
Assignment import reports include `single_owner_policy`, documenting that
recording ownership is keyed by `recording_id`, only one active owner is
allowed, reassignment replaces the previous owner, stale sessions close on
reassignment, and assignment/user matching is required before mutation.
If duplicate recording rows are intentionally applied without
`--warnings-as-errors`, the report keeps every input row visible, but only the
final row for each recording mutates ownership/status. Earlier duplicate rows
are marked with `duplicate_assignment_row_skipped_for_apply`; use
`skipped_duplicate_apply_count` or
`skipped_duplicate_assignment_apply_count` for automation.

When using `import-tasks` directly for launch-bound task manifests, add
`--warnings-as-errors` so missing/inactive assignment warnings or duplicate
logical task-scope warnings block apply before tasks are written.
Use `--output /path/to/task-import-report.json` to archive the exact dry-run or
apply report; existing report files are not overwritten unless `--overwrite` is
supplied.

When using `add-task` for one-off launch-bound work, add `--warnings-as-errors`
so missing or inactive recording assignments block task creation instead of
creating work that no labeler can see.
Use `--output /path/to/add-task-report.json` to archive the single-task report;
existing report files are not overwritten unless `--overwrite` is supplied.

Start from the generated `assignments-template.csv` and `tasks-template.csv`
when building spreadsheet-driven batches. The generated
`manifest-templates-readme.txt` includes the dry-run/apply commands and CSV
validation rules.

When assignment and task spreadsheets are reviewed together, dry-run both files
as one batch plan:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  import-batch-plan --assignments assignments.csv --tasks tasks.csv \
  --assigned-by OPERATOR --actor OPERATOR \
  --output batch-plan-dry-run.json \
  --html-output batch-plan-dry-run.html
```

Apply only after reviewing the combined dry-run output:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  import-batch-plan --assignments assignments.csv --tasks tasks.csv \
  --assigned-by OPERATOR --actor OPERATOR --apply
```

`import-batch-plan` validates that every task recording has an assignment after
combining the imported assignments with existing store assignments. It applies
assignments before tasks and refuses to apply if cross-file validation fails.
It warns, without blocking apply, when a task recording will be assigned but not
active after the plan because that task will not be available to the labeler. It
also warns when an assignment manifest contains multiple rows for the same
recording, when an imported assignment changes the current owner of a recording,
when a recording has tasks across multiple workflow kinds so mixed labeler
workload is explicit before launch, and when multiple task IDs point at the same
recording/workflow/component/run/scope logical work item.
Batch-plan reports include the same `single_owner_policy` metadata as direct
assignment import reports, so reviewed dry-run and apply artifacts preserve the
recording ownership rule that protects browser mutation.
Use `--warnings-as-errors` when warnings should block apply for a launch-bound
batch.
Use `--output` to archive the reviewed dry-run or apply report; existing report
files are not overwritten unless `--overwrite` is supplied. Use `--html-output`
for a human-readable review page with cross-file issues, row source lines,
summary counts, assignments, closed-session counts from reassignment, and tasks.
JSON reports include compact `issue_codes`, `warning_codes`, and blocking
warning fields for automation-friendly triage. HTML reports show those issue and
warning code summaries above the detailed row tables.

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  import-assignments --input assignments.json --assigned-by OPERATOR
```

Apply after reviewing the dry-run output:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  import-assignments --input assignments.json --assigned-by OPERATOR --apply
```

JSON manifest shape:

```json
{
  "assignments": [
    {
      "recording_id": "recording-a",
      "assignee_user": "alice",
      "status": "active",
      "notes": "Review keypoints first."
    }
  ]
}
```

CSV manifests are useful when the assignment plan comes from a spreadsheet:

```csv
recording_id,assignee_user,status,notes
recording-a,alice,active,Review keypoints first.
recording-b,bob,active,Review subject masks.
```

Fully blank trailing CSV rows are ignored. Partially filled rows still fail
validation so missing recording IDs or users are not applied accidentally.
CSV headers are validated before row parsing, and dry-run/apply output includes
`source_line` for CSV rows.

Each row assigns exactly one user to one recording. Duplicate `recording_id`
entries in the same manifest are rejected before any assignment is applied.
Reapplying an unchanged assignment is idempotent: it does not refresh
`assigned_at_utc` and does not close active sessions. Changing the assignee or
assignment status closes active sessions for that recording.

Assignment authority changes are audited separately from task mutation events:

```text
assignment_created
assignment_changed
```

Export assignment audit history with:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  export-assignment-events --recording-id RECORDING_ID --format jsonl \
  --output /path/to/audit-exports/RECORDING_ID-assignment-events.jsonl
```

The personalized dashboard includes workflow and text filters for large
assignments. These are client-side view filters only; assignment/session
authorization remains enforced on the server.

Task priority is shown on the personalized dashboard, and visible tasks are
ordered by descending priority within each recording.

Use assignment notes for recording-level instructions and task notes for
specific per-task guidance. Both are visible and searchable on the personalized
dashboard.

The dashboard includes a `Refresh work` control that refetches the current
server-side assignment/task list for the authenticated user after operator
changes.

Operators can preview the personalized dashboard payload for a user before
sharing access:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  work-summary --user LABELER \
  --output /path/to/work-previews/LABELER-work-summary.json
```

Use `--include-completed` to include completed tasks in the preview.

Changing a recording's assignee or assignment status closes active sessions for
that recording and records:

```text
session_closed_by_assignment_change
```

Every task-session read or write request also re-checks the current user,
current assignment, task state, and current writer session before reaching
workflow mutation code. This means reassignment, task completion, session
expiration, or supersession takes effect immediately for old browser tabs.

## Task Manifest Import

Workflow-specific task generators should be preferred when tasks can be derived
from registry rows. For manual batches or mixed workflow handoffs, import a CSV,
JSON, or JSONL task manifest. The command is dry-run by default:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  import-tasks --input tasks.json
```

Apply after reviewing the dry-run output:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  import-tasks --input tasks.json --apply
```

JSON manifest shape:

```json
{
  "tasks": [
    {
      "task_id": "recording-a-keypoints-review",
      "recording_id": "recording-a",
      "workflow_kind": "keypoints",
      "title": "Review keypoints",
      "scope": {"frames": [1, 2, 3]},
      "priority": 5
    }
  ]
}
```

CSV manifests are useful for spreadsheet-driven task plans. Use `scope_json`
when a task needs scoped frames, components, or other JSON metadata:

```csv
task_id,recording_id,workflow_kind,title,scope_json,priority,notes
recording-a-keypoints-review,recording-a,keypoints,Review keypoints,"{""frames"":[1,2,3]}",5,First pass
recording-b-box-review,recording-b,detect_analysis,Review boxes,"{""frames"":[10,11]}",3,
```

Fully blank trailing CSV rows are ignored. Partially filled task rows still fail
validation so missing task IDs, recordings, or workflow kinds are caught before
apply. CSV headers are validated before row parsing, and dry-run/apply output
includes `source_line` for CSV rows.

Task manifests require explicit `task_id` values so a reviewed dry-run and a
later apply refer to the same task ids. Duplicate `task_id` entries in the same
manifest are rejected before any task is applied.

The first browser-only rollout supports `keypoints`, `detect_training`,
`detect_analysis`, and `subject_mask_component` tasks. `detect_analysis` tasks
are reviewable in the browser and should be treated as mutable only when the
task scope enables analysis-box edits. The service exposes this scope as
`browser_workflows` in `/api/admin/summary`, `/api/admin/preflight`,
`/api/me/tasks`, `work-summary`, `dashboard-roster`, and generated handoff work
summaries.

Each `browser_workflows` row includes the session-scoped save contract used by
the browser and a `client_authority` block. `client_authority` should report
`mutation_executor=server`, `browser_can_submit_edits=true`,
`browser_can_write_zarr=false`, `browser_can_write_filesystem=false`,
`browser_receives_write_credentials=false`, and
`browser_receives_direct_zarr_handles=false`.
Mutable browser workflow `write_contract` entries must list `target_token` in
both `payload_fields` and `required_fields`. The server owns the mutation target
and rejects client-supplied target selectors such as `position`, `roi_idx`, or
`frame_idx`; `target_token` proves the browser is saving the currently loaded
server-held target and prevents stale same-session tabs from saving after a
different tab navigates the session.
Each `write_contract` also includes `audit_provenance`, which requires a
`labeling_task_events` row containing `event_id`, `task_id`, `recording_id`,
`user`, `event_type`, `created_at_utc`, `target`, `before`, and `after` for
each browser mutation.
Each `write_contract` includes `retry_policy`. Browser saves use replacement
semantics for the addressed target payload, so retrying the same payload is
data-safe where practical. Audit events are append-only, so duplicate audit rows
can appear after a lost browser response or manual retry. Editable
`detect_analysis` saves can also record secondary `promotion_success` or
`promotion_failed` events when promotion is enabled.

| Workflow | Save route | Required payload | Audit event |
| --- | --- | --- | --- |
| `keypoints` | `POST /api/sessions/{session_id}/keypoints/save` | `points` | `save_keypoints` |
| `detect_training` | `POST /api/sessions/{session_id}/detect/save` | `bbox_norm` | `save_detect_bbox` |
| `detect_analysis` | `POST /api/sessions/{session_id}/detect-analysis/save` | `bbox_norm`, with task scope `editable=true` | `save_detect_analysis_bbox` |
| `subject_mask_component` | `POST /api/sessions/{session_id}/subject-mask/save` | `mask` | `save_subject_mask_roi` |

All save routes require the same current-user, active-assignment, open-task, and
current-session guard before workflow mutation code runs. Successful saves
return `ok`, `result`, and `state`; detection-analysis saves may additionally
record `promotion_success` or `promotion_failed` when task scope enables
promotion into a training zarr.

Task definition changes are audited separately from task mutation events:

```text
task_definition_created
task_definition_changed
```

Use `--actor OPERATOR` on `add-task` or `import-tasks` when you want operator
identity captured in task-definition audit events. Reapplying an unchanged task
definition is idempotent and does not refresh task timestamps.

Export task-definition audit history with:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  export-task-definition-events --recording-id RECORDING_ID --format jsonl \
  --output /path/to/audit-exports/RECORDING_ID-task-definition-events.jsonl
```

For handoff or completed-batch archives, export all audit families into one
bundle:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  export-audit-bundle --recording-id RECORDING_ID \
  --output-dir /path/to/audit-exports/RECORDING_ID-audit-bundle
```

The bundle contains:

- `task-events.jsonl`
- `assignment-events.jsonl`
- `task-definition-events.jsonl`
- `manifest.json`

Existing bundle files are not overwritten unless `--overwrite` is supplied.

## Mutable Training Zarr Backup

Per-recording training zarrs are mutable curated data stores. They are not the
same thing as unified exported training artifacts.

Back up per-recording training zarrs before:

- large labeling batches
- enabling analysis-to-training promotion for new users
- running repair scripts
- migrating zarr layout or registry schema

Recommended policy:

1. Pause the labeling service or temporarily remove assignments for recordings
   being backed up.
2. Ensure no active sessions are writing to the target recording.
3. Copy the whole zarr directory atomically with a tool appropriate for the
   filesystem.
4. Keep the backup next to a manifest containing timestamp, recording id,
   dataset id, zarr path, and registry path.
5. Restore only while the service is stopped or the recording is unassigned.
6. After restore, refresh registry quality/status rows for the restored dataset.

Example manifest shape:

```json
{
  "backup_schema": "palette.web_labeling_zarr_backup.v1",
  "created_utc": "2026-06-23T00:00:00Z",
  "recording_id": "recording_a",
  "dataset_id": "dataset_a_training",
  "zarr_path": "/path/to/recording_training.zarr",
  "registry_path": "/path/to/palette_registry.sqlite",
  "reason": "before web labeling batch"
}
```

## Failed Promotion Repair

Assigned users see unresolved promotion failures on their dashboard, but the
labeler-facing row does not expose backend zarr paths or offer direct dashboard
retry. Treat these rows as operator-support prompts: repair the underlying
target-zarr or registry issue, then retry from the admin recovery view or from
session-bound client tooling that supplies the current guarded `session_id`.
Path-like strings in labeler-facing failed-promotion `target`, `after.error`,
and `after.details` fields are redacted before they reach `/api/me/tasks` or
the dashboard page. Exact app-local support URLs with known browser routes
such as `/work`, `/datasets`, `/identity`, and `/api/sessions/` are preserved
only when they are standalone values and do not contain `.zarr`; mixed
sentences and storage/Zarr references are still redacted. This is reported as
`labeler_safety.labeler_api_redaction.redacts_user_summary_path_like_string_values=true`.
Generated work-summary, dataset-queue, handoff, and launch manifest artifacts
carry the same flag so copied packages can be audited offline.

Admins can inspect failures across users from `/admin`.

Admin failed-promotion rows link to a read-only task detail page for the
affected task. The detail page shows task metadata, task scope, and recent audit
events for repair context only; it does not create a labeling session or bypass
assignment ownership.

A retry records either:

```text
promotion_success
promotion_failed
```

A successful retry hides the original failure from the user dashboard.

Retries are idempotency-guarded. When a retry starts, the service records:

```text
promotion_retry_started
```

If another user/admin click races with that retry, the second request returns a
`promotion_retry_in_progress` conflict instead of running a second promotion.
If the retry errors before recording a promotion outcome, the service records:

```text
promotion_retry_abandoned
```

After `promotion_failed` or `promotion_retry_abandoned`, the failure can be
retried again. After `promotion_success`, later retry clicks return the existing
success instead of promoting duplicate rows.

Export promotion repair context with:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \\
  export-events --task-id TASK_ID --limit 200
```

For a broader failed-promotion review:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \\
  export-events --event-type promotion_failed --limit 500
```

Use `--user LABELER` to filter by the current recording assignee. Use
`--actor LABELER` to filter by the user who created the audit event.
Use `--since-utc` and `--until-utc` to archive a specific labeling batch or
handoff window.

For a persistent archive, write JSON or JSONL to an explicit output path:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \\
  export-events --recording-id RECORDING_ID --format jsonl \\
  --since-utc 2026-06-23T00:00:00+00:00 \\
  --output /path/to/audit-exports/RECORDING_ID-events.jsonl
```

`export-events` refuses to overwrite an existing output file unless
`--overwrite` is supplied.

## Stale Session Cleanup

Admins can close expired open sessions from `/admin` using `Close stale
sessions`.

Operators can inspect browser sessions from the CLI:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  list-sessions --user LABELER --limit 100
```

Expired open sessions can be closed from the CLI:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  cleanup-stale-sessions --user OPERATOR
```

Cleanup records:

```text
stale_session_closed
```

This is an accident-prevention cleanup. It does not revert any saved labels.

## Concurrent Browser Sessions

Only one active browser session owns writes for a task at a time.

When a user opens the same task again from the dashboard or a signed link, the
new session supersedes older open sessions for that task. Old tabs will return a
`session_closed`, `session_expired`, or `session_superseded` conflict instead of
continuing to write labels.

Task pages show the session id and expiration timestamp. If an old tab reports a
session conflict, the user should return to the work dashboard and reopen the
task.

The sidecar store is opened with SQLite settings suitable for the threaded HTTP
server. Session creation is transactional, so simultaneous opens for the same
task settle to one active writer.

Superseding records:

```text
session_superseded
```

## Task Completion

Marking a task complete closes open sessions for that task and prevents signed
links or old dashboard links from opening new labeling sessions for it.
The service exposes this as `task_state_policy` in `/api/admin/summary`,
`/api/admin/preflight`, `/api/me/tasks`, `work-summary`, `dashboard-roster`,
and generated handoff work summaries. The policy is:
`completed_tasks_read_only=true`, `completed_tasks_open_new_sessions=false`,
`completed_task_open_requests=reject_task_complete`,
`completed_task_save_requests=reject_task_complete`,
`absolute_navigation_out_of_scope=reject_nav_error`,
`browser_mutation_target_selectors=server_owned_reject_client_fields`,
`browser_mutation_target_token=required_current_target_token`,
`labeler_promotion_retry_requires_current_session=true`,
`completion_closes_open_sessions=true`, `reopen_authority=operator`, and
`reopen_required_for_more_labeling=true`.

Labelers can mark a task complete from the browser editor after finishing the
assigned review/edit. The browser calls the guarded session completion endpoint,
which re-checks the current user, assignment, task state, and current writer
session before closing the task. Operators can also mark a task complete or
reopen it from `/admin/tasks/TASK_ID`; the admin API returns
`task_state_transition` and `closed_session_ids` for validation logs. The same
workflow is available from the CLI:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  set-task-state --task-id TASK_ID --state complete --user OPERATOR
```

Use `--output /path/to/task-state-report.json` when the completion or reopen
decision needs an archived operator report. Existing report files are not
overwritten unless `--overwrite` is supplied.

If a task must be reopened after an operator mistake, use the same command with
the desired non-complete state:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  set-task-state --task-id TASK_ID --state pending --user OPERATOR
```

Setting a task to its current state is idempotent and does not create duplicate
completion or reopen audit events.

Completion closes sessions with:

```text
session_closed_by_task_completion
```

Reopening a completed task records:

```text
task_reopened
```

Old tabs receive a `task_complete` conflict before workflow mutation code runs.

## Operator Process

Use this process for routine multi-user labeling operations:

1. Back up the sidecar store with `backup-store` before large assignment or task
   generation changes.
2. Back up mutable per-recording training zarrs before enabling promotion or
   before large labeling batches.
3. Assign each recording to exactly one user from `/admin`, the `assign` CLI, or
   a reviewed `import-assignments --apply` manifest.
4. Add assignment notes when the labeler needs recording-specific instructions.
5. Generate tasks from registry rows or apply a reviewed `import-tasks --apply`
   manifest after assignments are current.
   Keep the first browser-only launch to `keypoints`, `detect_training`,
   `detect_analysis`, and `subject_mask_component`; confirm the service
   `browser_workflows` metadata before inviting labelers if new workflow kinds
   appear in the task manifest.
6. For dashboard-only launches, export a roster without signed links:
   `dashboard-roster --base-url https://labeling.example.org --output dashboard-roster.json`.
   The roster includes a per-user `invitation_message` field containing
   ready-row draft text that can be copied into email or chat only after
   `ready_to_invite` is true and safe-share
   inspection reports `labeler_links_safe_to_share=true`. If readiness is false,
   inspect `invite_reasons` such as `missing_base_url`, `no_active_recordings`,
   or `no_open_tasks` before contacting that user. If readiness is true but
   `safe_share_next_action_summary` still lists missing or unsatisfied evidence,
   keep the draft text internal until those evidence gates pass. An empty roster
   fails with `no_users`; check assignment imports or the `--user` filter before
   sending.
   `warning_codes` also includes `dataset_queue_blocks_labeler_start` whenever
   a user's queue-state contract says the labeler should not start from the
   queue yet, even though the more specific `invite_reasons` remain available.
	   Each row also includes legacy `ready_state` (`ready_to_invite` or
	   `not_ready_to_invite`) for row readiness. Treat those legacy
	   invitation-named values as work/readiness signals only, not link-sharing
	   approval. Roster and `/admin` summaries include aggregate `ready_states`
	   counts plus `ready_to_invite_users` and `not_ready_to_invite_users` lists
	   for compatibility, and draft-safe aliases `ready_row_draft_count`,
	   `diagnostic_note_count`, `ready_row_draft_users`, and
	   `diagnostic_note_users` keyed to the copy-intent contract.
   They also include aggregate identity-probe readiness fields:
   `identity_probe_required`, `identity_probe_available`,
   `identity_probe_missing`, and `identity_probe_missing_users`. Use these
   fields to confirm every invite-ready user has an expected-user identity probe
   link before copying ready-row draft text.
   Dashboard-roster JSON/status/HTML output also includes
   `dataset_queue_start_readiness`, matching the runtime admin summary and
   generated validation checklist gate. Treat `status=needs_review` as a block
   on sending start links until the listed users' queue states are resolved.
   Dashboard-roster JSON/HTML exports and `/admin` summaries also include
   `completion_states`, aggregate `completion_percent`, and per-user
   `completion_state` / `completion_percent` fields so operators can monitor
   not-started, in-progress, and complete users from the launch surface.
   `/admin/users/LABELER` and `/api/admin/users/LABELER` give focused, redacted
   per-user views with expected dashboard URL, ready-row draft text, progress
   summary, `dataset_queue_state`, assigned recordings, copy-intent labeling,
   and links or IDs for task-state controls.
   Personal work payloads include `empty_state` with stable codes such as
   `no_active_assignments`, `no_open_tasks`, `all_tasks_complete`, and
   `has_open_work`, plus the operator action needed when the labeler sees no
   work. They also include `progress_summary`, which separates waiting,
   complete, and blocked/no-open recordings for the personalized `/work` page
   and generated work summaries. They also include `dataset_queue` and
   `dataset_queue_summary`, a user-scoped open-work dataset/recording grouping
   derived from that user's assigned task rows after redaction; completed tasks
   are omitted from this queue even when the main task list is showing
   completed rows for read-only review. `/api/me/datasets` returns the same
   personalized dataset queue for automation or lightweight "work waiting"
   pages, with both `datasets` and `dataset_queue` fields for compatibility
   with work-summary payloads. `/api/me/tasks` and `/api/me/datasets` both
   return guarded self-links for the resolved user:
   `expected_user_labeler_landing_url`, `expected_user_dashboard_url`,
   `expected_user_dataset_queue_url`, and `expected_user_identity_probe_url`.
   `/api/me/tasks` carries those fields
   inside the returned `work` object, while `/api/me/datasets` exposes them at
   the top level for lightweight queue clients. Both endpoints also expose
   `dataset_queue_state` with stable queue-state codes such as
   `has_open_dataset_work`, `no_active_assignments`,
   `all_assigned_work_complete`, `assigned_recordings_need_operator_action`, and
   `no_open_dataset_work`. Both endpoints also return
   `labeler_safety` metadata so client tooling can audit browser-only,
   no-forwarding, no-direct-Zarr-edit, and expected-user guard policy without
   scraping HTML. Both endpoints also return safe `zarr_backup_policy` metadata
   so client tooling can see the backup validation gate and confirm labelers do
   not receive backup paths without exposing raw Zarr paths. They also return
   safe `mutation_audit_policy` metadata so client tooling can confirm browser
   mutations are expected to be recorded server-side in `labeling_task_events`.
   Both endpoints honor
   `expected_user`; a mismatch returns
   `dashboard_user_mismatch` instead of returning another authenticated user's
   assigned work. In `dataset_queue_summary`, `dataset_count` counts queue
   groups, including the explicit `Unspecified dataset` group for tasks without
   `dataset_id`; `dataset_ids` lists only named dataset IDs. Task-open requests
   from a guarded dashboard echo the same expected user and fail with
   `task_open_user_mismatch` if the browser identity changes before session
   creation. Direct task-complete API requests can also carry `expected_user`,
   must include the current guarded `session_id`, and fail with
   `task_complete_user_mismatch`, `session_required`, or
   `session_task_mismatch` before task state is changed.
   Labeler promotion-retry API requests carry the same expected user and fail
   with `promotion_retry_user_mismatch` before a retry
   claim is created. Labeler retries also require the task to still be open
   and must include the current guarded `session_id`; missing or wrong-task
   sessions fail with `session_required` or `session_task_mismatch` before any
   retry claim is created. Completed tasks return `task_complete` until an
   operator reopens or retries through the admin recovery route.
   Per-user handoff bundles also write the same queue to
   `dataset-queue.json`, and batch handoff rosters include queue counts plus a
   first guarded queue link for spreadsheet review. Dataset, recording, and
   queue-task rows include plain relative
   `work_url` filter values such as `/work?dataset_id=...` plus guarded
   `expected_user_work_url` values such as
   `/work?expected_user=LABELER&dataset_id=...&recording_id=...`; these are
   filter links only, not authorization grants. Prefer `expected_user_work_url`
   when copying a specific dataset or recording link for a labeler. Generated
   signed task-link rows also include `expected_user`, matching the signed
   token binding and the current assignment user at export time. The
   dashboard applies `dataset_id`, `recording_id`, and `workflow` query
   parameters after resolving the authenticated user and loading that user's
   assigned work. `/work` can request
   `/datasets` serves a dedicated lightweight queue-first page and honors the
   same `expected_user` mismatch guard.
   `/work` can request `/api/me/tasks?include_completed=1` to show completed task rows read-only;
   completed tasks still do not open labeling sessions unless an operator
   reopens them.
   The dashboard-roster JSON export includes `status_report`, an exportable
   multi-user batch report with aggregate readiness, progress, queue-state,
   warnings, and per-user status rows. The HTML export labels itself as
   `multi_user_labeling_status` for archive review.
   `/admin` also includes `active_session_user_counts`,
   `active_session_workflow_counts`, `stale_session_user_counts`, and
   `stale_session_workflow_counts` so operators can see who currently has open
   browser sessions and which stale sessions may need cleanup.
   `/admin` and `/api/admin/summary` also include
   `assignment_ownership_integrity`, with active assignment counts, unique
   active recording counts, and duplicate active owner count.
   Assignment monitoring uses `assignment_work_state_counts` and
   `assignment_operator_rows` to distinguish active work, complete assignments,
   non-active assignments, and blocked active assignments with no tasks or no
   open work.
   `/admin` also lists recent task rows under `Task state controls`; each task
   links to `/admin/tasks/TASK_ID`, where operators can inspect audit events,
   mark completion, or reopen completed work when correction is required.
   `/admin` includes recent audit summaries as `recent_audit_event_user_counts`,
   `recent_audit_event_type_counts`, `recent_audit_event_workflow_counts`, and
   `recent_audit_events`, giving an operator a quick "who changed what" view
   before opening detailed task audit pages or exporting audit bundles.
   Per-user admin pages at `/admin/users/<user>` show the guarded landing,
   dashboard, dataset queue, and identity-probe links for that user. Recording
   admin pages at `/admin/recordings/<recording_id>` also expose the current
   owner's guarded landing and dataset queue links.
   Each row also includes `expected_user_identity_probe_url`,
   `expected_user_labeler_landing_url`, `expected_user_dataset_queue_url`, and
   `expected_user_dashboard_url`.
	   Ready-row draft text tells the labeler to
	   open the identity probe first, confirm it reports the expected assignment
	   user, then continue to the guarded `/` landing page or `/datasets` queue;
   `/work` is listed as the full dashboard fallback rather than the preferred
   first click. The identity probe JSON and page also expose the guarded
   landing and dataset queue URLs as `expected_user_labeler_landing_url` and
   `expected_user_dataset_queue_url`.
   The roster payload and each user row include `labeler_safety` metadata with
   the same identity-check policy, plus flat spreadsheet-friendly safety fields
   such as `identity_probe_required`, `identity_probe_available`,
   `dashboard_identity_check_required`, `browser_only`, and
   `no_direct_zarr_edits`.
   Not-ready rows include `invite_actions` so `missing_base_url`,
   `no_active_recordings`, `no_open_tasks`, and `no_users` states have a concrete
   operator next step. The dashboard roster HTML overview also prints the
   aggregate invite action so the repair path is visible without opening JSON.
	   Not-ready dashboard roster and `/admin` rows use `Copy not-ready note`
	   rather than ready-row draft controls so diagnostic text is not mistaken for
	   sendable work text. Roster rows expose the same distinction as `copy_label` for
   CSV/JSON consumers, and expose stable `copy_intent` values
   (`ready_row_draft` or `diagnostic_note`) for automation. Roster and `/admin`
   summaries also include aggregate `copy_intents` counts. Not-ready copied notes
   include the repair action text so the diagnostic remains actionable when
   pasted outside the roster.
	   Dashboard roster HTML and `/admin` include a ready-row draft bulk copy block
	   that excludes not-ready diagnostic notes but still requires safe-share
	   inspection before sharing.
   Rows and aggregate reports also include
   `recordings_without_open_tasks_actions` so `tasks_not_generated`,
   `all_tasks_complete`, and `no_open_tasks_in_current_summary` states indicate
   whether to generate tasks, reopen completed work, or inspect task visibility.
   Dashboard roster rows and `/admin` summaries also expose live
   `waiting_datasets`, `dataset_open_tasks`, guarded
   `expected_user_labeler_landing_url`, `dataset_queue_state`,
   `dataset_queue_state_code`, `dataset_queue_state_title`,
   `dataset_queue_blocks_labeler_start`, flat `dataset_queue_start_ready`,
   `dataset_queue_start_status`, `dataset_queue_start_operator_action`, and first guarded
   `dataset_queue_preview_url` values so operators can see who has dataset-level
   work waiting or blocked without generating handoff bundles. The same fields
   are present in dashboard-roster JSON/JSONL/CSV rows, status-report rows, and
   the HTML roster table. Dashboard-roster summaries also expose
   `dashboard_warnings` so launch automation can fail on blocked queue states
   without scraping row text.
   `--include-completed` can show completed rows for review, but
   `ready_to_invite` is still based on incomplete/open tasks.
   Use `--format html --output dashboard-roster.html` when you want a browser
   review page with copy buttons for ready-row draft text.
7. Ask labelers to use `/` or `/datasets` for a queue-first landing page, then
   `/work` as the canonical task dashboard. `/me` is also a queue-first labeler
   landing alias. Generated handoff and launch manifests expose
   both `dashboard_path` and `dashboard_url`; per-user handoff manifests,
   indexes, roster CSV rows, and inspection reports also expose
   `expected_user_identity_probe_url`, `expected_user_dataset_queue_url`, and
   `expected_user_dashboard_url`.
   These fields are blank when the handoff was generated without `--base-url`.
   The handoff and launch README and HTML overview files also show the exact
   dashboard URL, or `(missing --base-url)` when it is not ready to send.
8. Run `batch-readiness` and `check-store` before sharing links or announcing a
   new batch; inspect counts for active assigned recordings with no task rows or
   no open tasks, plus handoff no-open-task reason breakdowns.
9. Generate handoff directories for the batch with `export-user-handoffs`, or a
   single labeler directory with `export-user-handoff --user LABELER`.
10. Run handoff/package inspection with `--require-shareable` and confirm
   `labeler_links_safe_to_share=true`; do not share based on `ready_to_send` or
   `ready_to_invite` alone.
11. Review `validation-checklist.json` to identify pending operator-only
   validation gates, then monitor `/admin` for preflight warnings, dashboard-user readiness,
   live waiting-dataset counts, the dataset queue page path, first guarded dataset queue links, invite/no-open
   next actions, copy-ready dashboard draft text using the browser-visible
   dashboard URL, copy buttons for dashboard ready-row drafts, active sessions, stale
   sessions, and failed promotions.
12. Use `Close stale sessions` for expired open sessions.
13. Use failed-promotion retry only after the underlying target-zarr or registry
   issue has been repaired.
14. Mark tasks complete only after the labeler or operator confirms the relevant
    review state is acceptable.
15. Export an audit bundle with `export-audit-bundle` when handing off a repair,
    archiving a completed batch, or investigating partial save/promotion state.
16. Build immutable exported training artifacts only from reviewed mutable
    per-recording workspaces.

For reassignment, update the recording assignment from `/admin`. The service
closes active sessions for that recording and every old tab re-checks assignment
ownership before reaching workflow mutation code.
The admin assignment API and direct `assign` CLI report include
`single_owner_policy` plus `assignment_transition`, so the archived mutation
evidence records both the before/after owner and the one-owner safety rule.

## Rollback And Recovery

Use these rollback paths when an assignment, task state, or browser mutation is
wrong.

Incorrect assignment:

1. Pause the affected recording assignment from `/admin` or the `assign` CLI so
   old browser tabs stop at the session guard.
2. Confirm `/admin` no longer shows active sessions for that recording, or run
   stale/session cleanup if needed.
3. Reassign the recording to the correct user. The assignment table enforces one
   current owner per recording.
4. Export `dashboard-roster --base-url ... --output dashboard-roster.json` and
   confirm the corrected user has a ready-row draft and safe-share inspection
   passes before sending a new link.
5. Keep the assignment events and status report with the batch notes.

Wrong task completion or premature completion:

1. Use `set-task-state --task-id TASK_ID --state pending --user OPERATOR` to
   reopen the task only when more work is required.
2. Regenerate ready-row draft text or resend a guarded dashboard link only after
   `/admin` shows the task as open and assigned to the intended user, and
   safe-share inspection still passes.
3. If the task was completed correctly, do not reopen it just to make old links
   work; completed tasks are intentionally read-only.

Bad browser mutation:

1. Pause the assignment for the affected recording before touching the zarr.
2. Back up the current sidecar store and affected mutable zarr before repair.
3. Inspect `/admin` recent audit summaries, task detail audit events, or an
   exported audit bundle to identify the event, user, task, target, before, and
   after payload.
4. Restore the affected per-recording zarr from the pre-batch backup, or make a
   deliberate corrective edit through the assigned browser workflow.
5. Refresh registry status/quality rows for the repaired dataset when the
   mutation or promotion affected registry-visible metadata.
6. Export a new status report and only then return the assignment to `active`.

## Partial Save Repair

A label save, promotion, and registry refresh are intentionally separate audited
steps. Do not assume a later failure means the earlier label mutation failed.

Use the event stream this way:

1. If the label save event exists, treat the zarr label mutation as the first
   source to inspect.
2. If `promotion_failed` exists after a save, inspect the failed event details
   and repair the promotion target before retrying.
3. If `registry_refresh_failed` exists after a save or promotion, repair the
   registry path/dataset/zarr metadata and refresh the registry again. Do not
   roll back labels only because registry refresh failed.
4. If a retry records `promotion_success`, the original failure is considered
   resolved and is hidden from the normal failed-promotion queue.
5. If a retry records `promotion_failed` or `promotion_retry_abandoned`, repair
   the underlying issue and retry again.
6. If a label mutation itself appears wrong, restore the affected mutable
   per-recording zarr from backup or make a deliberate corrective edit through
   the assigned labeling workflow.

For destructive repair, pause work on the affected recording first:

1. Change the assignment status to `paused` from `/admin`.
2. Confirm active sessions for the recording have been closed.
3. Back up the current sidecar store and affected zarr before repair.
4. Restore or repair the zarr.
5. Refresh registry quality/status rows for the repaired dataset.
6. Return the assignment status to `active`.

## Static Validation

Before operator testing, run the aggregate readiness helper:

```bash
scripts/check_labeling_web_readiness.sh
```

Use generated `validation-checklist.json` to track which gates still require
operator evidence. Record validation results in
`docs/web_labeling_validation_log_template.md` or copy that template into the
batch archive. When assignment ownership or status changes during validation,
copy the `/api/admin/assignments` transition fields into the
`Assignment Transition Evidence` section.
The generated checklist includes `dataset_queue_start_readiness`; this gate must
pass before labeler links are shared, or the operator must resolve the listed
blocked queue state and regenerate or update the archive.
After recording evidence, update the machine-readable checklist gate status:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  update-validation-checklist --path /path/to/validation-checklist.json \
  --gate browser_smoke --status passed \
  --evidence "Opened /work, /admin, session load, failure state, completion, and reopen." \
  --append-log /path/to/validation-log-template.md \
  --operator OPERATOR
```

Use `--output updated-validation-checklist.json` to write a reviewed copy
instead of updating the checklist in place.
`--append-log` appends a structured Markdown entry to the validation log and
records that log path as an evidence file on the same checklist gate.
The command response includes `available_gates` with current gate IDs, titles,
statuses, and required flags; if a gate ID is wrong, the error lists available
gate IDs with their current statuses.
Generated validation logs include the paired `validation-checklist.json` path
and an `update-validation-checklist --append-log` command template so operators
can record evidence from the log file itself.

It checks the production decision record, runs the focused compile helper, runs
the focused non-zarr unit tests, and runs the real-zarr smoke only when
`PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC` is set.

For focused debugging, the individual commands are:

```bash
scripts/py scripts/check_labeling_production_decision_record.py
```

```bash
scripts/check_labeling_web_static.sh
```

```bash
scripts/check_labeling_web_unit.sh
```

The decision-record checker fails until required production fields are filled
and required sign-off fields are set to `yes`.

The static helper uses `scripts/py` and compiles the labeling web implementation
plus the focused unit/integration test files. It does not run tests or touch
real zarrs.

The unit helper runs the focused non-zarr labeling web tests. It does not run
the real-zarr smoke test.

## Browser Smoke Validation

Run browser smoke checks after static validation passes and before inviting
external labelers. Use a disposable sidecar store and a local fixed user first:

```bash
scripts/py -m fisheye.utils.labeling_work --store /tmp/palette-labeling-smoke.sqlite \
  serve --host 127.0.0.1 --port 8795 --user alice --admin-user alice
```

In the browser, check:

1. `/work` loads and reports the signed-in user.
2. A user with no assignments gets the `no_active_assignments` empty state.
3. `/admin` loads the preflight, roster, assignment work states, session
   summaries, audit summaries, and browser workflow contracts.
4. Opening a task creates a guarded session.
5. Failed session/API paths show copyable operator support details.

For a proxy-auth staging deployment, run `preflight` before serving:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  preflight --host 127.0.0.1 --port 8795 --trust-auth-header \
  --auth-header X-Forwarded-User --admin-user OPERATOR --production \
  --output /path/to/preflight.json --overwrite
```

Then start the service behind the trusted proxy:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling.sqlite \
  serve --host 127.0.0.1 --port 8795 --trust-auth-header \
  --auth-header X-Forwarded-User --admin-user OPERATOR --production \
  --link-secret "$PALETTE_LABELING_LINK_SECRET"
```

## Real-Zarr Smoke Validation

The real-zarr web smoke test is skipped by default. To run it on a workstation
with access to representative mutable zarrs, create a JSON spec and point
`PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC` at it.

Start from:

```text
docs/web_labeling_real_zarr_smoke_spec.template.json
```

Example spec shape:

```json
{
  "schema": "palette_labeling_web_real_zarr_smoke.v1",
  "cases": [
    {
      "name": "keypoint save smoke",
      "user": "alice",
      "recording_id": "recording-a",
      "task_id": "smoke-keypoints-a",
      "workflow_kind": "keypoints",
      "dataset_id": "dataset-a",
      "zarr_use": "training",
      "scope": {
        "zarr_path": "/path/to/training.zarr",
        "include_all": true,
        "target_roi_indices": [0]
      },
      "requests": [
        {"method": "GET", "path": "/state"},
        {"method": "POST", "path": "/nav", "body": {"position": 0}},
        {
          "method": "POST",
          "path": "/save",
          "body": {"points": [[10.0, 10.0], [12.0, 12.0]]}
        }
      ]
    }
  ]
}
```

Run outside the Codex sandbox when real zarr paths or CUDA-backed paths are
involved:

```bash
PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC=/path/to/web_labeling_smoke.json \
PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
scripts/py -m pytest -p no:cacheprovider \
  tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py -q
```

Use disposable or backed-up mutable zarrs for save smokes. The smoke test opens
real assigned sessions and sends the request bodies from the spec to the actual
web workflow routes.

## Deployed Dry Runs

For a one-operator/one-labeler dry run:

1. Create or choose one disposable assigned recording.
2. Back up the sidecar store and mutable zarr.
3. Assign the recording to the labeler and generate one first-rollout browser
   workflow task.
4. Export a dashboard roster and confirm `status_report.ok=true`.
5. Have the labeler open the guarded `/work?expected_user=LABELER` URL.
6. Have the labeler open, save, and complete one task.
7. Confirm `/admin` shows the task complete, no active sessions for that task,
   and the expected recent audit event.
8. Export an audit bundle or launch bundle and archive it with the smoke notes.

For a multi-user dry run:

1. Use at least two labelers and at least two recordings.
2. Confirm `recording_assignments` has exactly one active owner per recording.
3. Export `dashboard-roster --base-url ... --output dashboard-roster.json`.
4. Confirm every ready-row draft includes an `expected_user_identity_probe_url`,
   an `expected_user_dataset_queue_url`, and an `expected_user_dashboard_url`.
5. Confirm each browser sees only that user's assigned recordings.
6. Reassign one recording during the dry run and confirm the old browser tab
   stops with an assignment/session guard instead of mutating labels.
7. Complete at least one task and confirm completed work is read-only until an
   operator reopens it.
8. Archive the status report, audit summary, and rollback notes.

## Registry Refresh

When task scope includes `registry_path`, `dataset_id`, and `zarr_path`, the
service attempts best-effort registry refresh after successful browser
mutations.

Refresh events are audited separately:

```text
registry_refresh_success
registry_refresh_failed
```

A label save can still be valid even if registry refresh fails. Treat refresh
failures as operational repair items, not label rollback signals.

## Admin Assignment Transition Evidence

When an operator assigns or reassigns a recording through `/admin` or
`/api/admin/assignments`, the response includes `previous_assignment`,
`assignment_transition`, `closed_session_count`, and `closed_session_ids`.
Record those fields in the validation or rollback log whenever ownership or
assignment status changes during a batch.
