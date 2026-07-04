<!-- ARCHIVED 2026-07-04: consolidated — current pointer is docs/web_labeling_implementation_status.md. -->

# Web Labeling Assignment Implementation Checklist

<!-- design-meta
status: active implementation checklist
last_updated: 2026-06-23
scope: multi-user browser labeling without requiring local Palette/Crimson installs
-->

## Purpose

This document summarizes the implementation path for web-based Palette labeling
work. The goal is to let collaborators complete assigned labeling/review work in
a browser without installing Palette or Crimson.

The workflow needs two safe entry points:

- A personalized "my labeling work" page for the authenticated user.
- Opaque signed task links that can be shared with a labeler.

Both entry points must still enforce server-side authentication, recording
assignment, task state, and audited zarr mutation through Palette backend code.

## Core Decision

One user is assigned to a given recording.

That makes the recording the authorization root:

```text
recording_id -> assignee_user
```

Everything below the recording inherits that ownership boundary:

- analysis zarrs
- per-recording training zarrs
- detection tasks
- keypoint tasks
- subject-mask component tasks
- registry refreshes
- optional promotion from reviewed analysis edits into training zarrs

This intentionally avoids per-frame, per-ROI, and per-zarr ownership for the
first multi-user workflow. It keeps assignment, auditing, reassignment, and
operator recovery simple.

## Safety Invariants

- Browser URLs must not contain authoritative raw zarr paths.
- Browser requests must not be trusted for recording ownership, run names, zarr
  locations, or task scope.
- A user can only see and mutate tasks under recordings assigned to that user.
- Assignment must be checked before opening a session and before mutation.
- Completed tasks must reject new sessions and further mutation.
- Assignment changes must close active sessions for the affected recording.
- Save, review, promotion, retry, and registry refresh operations must be
  auditable as separate events.
- Per-recording training zarrs are mutable curated workspaces.
- Unified/exported training artifacts are immutable build outputs.
- Review proxy videos are convenience views, not canonical label truth.

## Implemented Components

Primary implementation files:

- `src/fisheye/labeling/assignment_store.py`
- `src/fisheye/labeling/web.py`
- `src/fisheye/labeling/__init__.py`
- `src/fisheye/utils/labeling_work.py`

Supporting operator docs and helpers:

- `docs/web_labeling_implementation_checklist_clean.md`
- `docs/web_labeling_multi_user_workflow_checklist.md`
- `docs/web_labeling_first_batch_operator_checklist.md`
- `docs/web_labeling_deployment_runbook.md`
- `docs/web_labeling_operator_handoff.md`
- `docs/web_labeling_first_operator_test_plan.md`
- `docs/web_labeling_production_decision_record.md`
- `docs/web_labeling_deployment_examples.md`
- `docs/web_labeling_real_zarr_smoke_spec.template.json`
- `docs/web_labeling_implementation_manifest.json`
- `scripts/check_labeling_production_decision_record.py`
- `scripts/check_labeling_web_static.sh`
- `scripts/check_labeling_web_unit.sh`
- `scripts/check_labeling_web_readiness.sh`
- `scripts/setup_labeling_web_local_smoke_store.sh`
- `scripts/start_labeling_web_local_smoke.sh`

## Data Model Checklist

### Recording assignments

- [x] Store one owner per recording.
- [x] Track assignment status, notes, assigned-by, and assigned-at metadata.
- [x] Make unchanged assignment updates idempotent.
- [x] Close active sessions only when owner/status changes require it.
- [x] Audit assignment creation and assignment changes.
- [x] Export assignment audit events with filters and no-overwrite protection.

### Labeling tasks

- [x] Store stable task IDs under a recording authorization root.
- [x] Route tasks by workflow kind.
- [x] Store dataset, zarr use, stage group, run name, component, scope, state,
  priority, notes, and timestamps.
- [x] Make unchanged task upserts idempotent.
- [x] Audit task definition creation and definition changes.
- [x] Export task-definition audit events with filters and no-overwrite
  protection.
- [x] Reject new sessions for completed tasks.
- [x] Close active sessions when a task is completed.
- [x] Support audited task reopen.

Implemented workflow kinds:

- [x] `keypoints`
- [x] `detect_training`
- [x] `detect_analysis`
- [x] `subject_mask_component`

### Browser sessions

- [x] Use short-lived session IDs instead of raw paths.
- [x] Keep one current writer per task.
- [x] Supersede older open sessions when a newer session is opened.
- [x] Clean up stale sessions.
- [x] Re-check assignment before session use.
- [x] Re-check completion state before mutation.
- [x] Provide archived operator session listing and stale-session cleanup
  commands.

### Task events

- [x] Audit session open, close, supersede, force-close, and stale-close events.
- [x] Audit save events for each supported workflow.
- [x] Audit review status changes.
- [x] Audit task completion and task reopen.
- [x] Audit promotion success and failure.
- [x] Audit failed-promotion retry claim and abandon events.
- [x] Audit registry refresh success and failure.
- [x] Export task events as JSON or JSONL.
- [x] Filter event exports by task, recording, event type, current assignee,
  actor, and UTC time window.
- [x] Export a combined audit bundle for archive or handoff.

## Browser Product Checklist

### Personalized work page

- [x] Serve the user dashboard at `/`.
- [x] Show only work assigned to the authenticated user.
- [x] Show active assigned recordings even when no open task rows are currently
  available, so task-generation gaps do not look like missing assignments.
- [x] Distinguish assigned recordings with no generated tasks from assigned
  recordings whose tasks are already complete.
- [x] Include labeler-facing no-open-task reason codes and messages in
  personalized work summaries.
- [x] Group work by recording and workflow.
- [x] Show assignment notes as user-facing instructions.
- [x] Show task notes as task-specific labeler instructions.
- [x] Show open, total, and completed task counts.
- [x] Show task priority and order visible tasks by descending priority within
  each recording.
- [x] Provide client-side search and workflow filtering.
- [x] Provide a visible refresh control to reload the authenticated user's
  current assignments/tasks from the server.
- [x] Show browser-only safety guidance: no local Palette/Crimson install, no
  direct zarr edits, and no forwarding links/handoffs.
- [x] Expose the same payload through `GET /api/me/tasks`.

### Signed task links

- [x] Resolve opaque signed links at `/t/<token>`.
- [x] Include issued-at and expiration in link tokens.
- [x] Support a link revocation floor.
- [x] Require authenticated identity even with a valid signed link.
- [x] Re-check live assignment before resolving a link to a session.
- [x] Generate one signed link with `sign-link`.
- [x] Generate signed-link manifests with `sign-links`.

### Admin/operator surface

- [x] Serve an admin dashboard at `/admin`.
- [x] Expose assignment and task summary APIs.
- [x] Allow admin assignment edits.
- [x] Close affected sessions after assignment changes.
- [x] Provide read-only task detail pages.
- [x] Show task metadata, scope, and recent audit events.
- [x] Link unresolved failed-promotion rows to task detail.

## Workflow Mutation Checklist

### Keypoints

- [x] Load task state.
- [x] Load current ROI image.
- [x] Navigate within task scope.
- [x] Save keypoint edits.
- [x] Set review status.
- [x] Record save and review audit events.

### Detection training zarrs

- [x] Load task state.
- [x] Load current frame.
- [x] Navigate within task scope.
- [x] Save bounding-box edits.
- [x] Record save audit events.

### Detection analysis zarrs

- [x] Load task state.
- [x] Serve scoped review media with range support.
- [x] Load current frame.
- [x] Navigate within task scope.
- [x] Save bounding-box edits.
- [x] Optionally promote curated analysis edits to training zarrs.
- [x] Record save and promotion audit events separately.
- [x] Support idempotent retry of failed promotions.

### Subject-mask components

- [x] Route manual review/editing through subject-mask tooling, not new
  eye-mask-first paths.
- [x] Load task state.
- [x] Load current ROI image.
- [x] Navigate within task scope.
- [x] Save subject-mask ROI edits.
- [x] Set review status.
- [x] Record save and review audit events.

## CLI Checklist

### Store setup and safety

- [x] `init` creates the assignment store.
- [x] `backup-store` creates a SQLite backup.
- [x] `check-store` reports hard consistency issues and operator warnings.
- [x] `check-store --output` archives JSON consistency reports with
  no-overwrite protection unless `--overwrite` is supplied.
- [x] `batch-readiness` reports active assignees, launchable open work, empty
  active assignments, and store consistency before announcing a batch.
- [x] `batch-readiness` reports no-open-task reason breakdowns for active
  assigned recordings with no launchable work.
- [x] `batch-readiness --warnings-as-errors` and
  `export-launch-bundle --warnings-as-errors` can make readiness warnings
  launch-blocking for automation.
- [x] strict readiness reports include `blocking_warning_count` and
  `blocking_warning_codes` for automation-friendly failure triage.
- [x] `preflight` checks launch safety without starting the server.
- [x] `preflight --output` archives JSON server launch-safety reports with
  no-overwrite protection unless `--overwrite` is supplied.
- [x] `serve` starts the web service.

### Assignment and task management

- [x] `assign` assigns or reassigns one recording.
- [x] `assign --output` archives JSON single-assignment reports with
  no-overwrite protection before mutating assignments unless `--overwrite` is
  supplied.
- [x] `import-assignments` dry-runs or applies JSON/JSONL assignment batches.
- [x] `import-assignments` reports warning metadata for duplicate assignment
  rows and owner changes that will close active sessions if applied.
- [x] `import-assignments --apply` and `import-batch-plan --apply` keep
  duplicate assignment input rows visible in results but mutate only the final
  row for each recording.
- [x] Earlier duplicate assignment rows skipped during apply carry
  `duplicate_assignment_row_skipped_for_apply` row-level warnings.
- [x] Assignment import apply payloads report aggregate skipped duplicate apply
  counts for automation.
- [x] Batch-plan HTML reports show duplicate-apply skipped rows and row-warning
  codes in the assignment table.
- [x] `import-assignments --warnings-as-errors` blocks apply when assignment
  import warnings are present.
- [x] `import-assignments` reports generated-at and blocked-by-warnings metadata
  for archived dry-run/apply payloads.
- [x] `import-assignments --output` archives JSON import reports with
  no-overwrite protection unless `--overwrite` is supplied.
- [x] `import-assignments` accepts CSV assignment manifests for
  spreadsheet-driven recording ownership plans.
- [x] `export-assignments` archives the current recording assignment snapshot as
  JSON, JSONL, or CSV.
- [x] `add-task` adds one labeling task.
- [x] `add-task` reports assignment visibility warnings and supports
  `--warnings-as-errors` to block creating tasks that labelers cannot see.
- [x] `add-task --output` archives JSON single-task reports with no-overwrite
  protection unless `--overwrite` is supplied.
- [x] `import-tasks` dry-runs or applies JSON/JSONL task batches.
- [x] `import-tasks` reports warning metadata for tasks whose recordings are
  missing assignments, inactive assignments, or duplicate logical task scopes.
- [x] `import-tasks --warnings-as-errors` blocks apply when task import warnings
  are present.
- [x] `import-tasks` reports generated-at and blocked-by-warnings metadata for
  archived dry-run/apply payloads.
- [x] `import-tasks --output` archives JSON import reports with no-overwrite
  protection unless `--overwrite` is supplied.
- [x] `import-tasks` accepts CSV task manifests with JSON `scope` or
  `scope_json` cells for spreadsheet-driven task plans.
- [x] `export-tasks` archives the current labeling task snapshot as JSON, JSONL,
  or CSV.
- [x] `write-manifest-templates` writes starter assignment/task CSV templates
  plus a README with dry-run/apply commands and CSV rules.
- [x] `import-batch-plan` dry-runs or applies assignment and task manifests
  together, validates task recordings have assignments after the plan, and
  applies assignments before tasks.
- [x] `import-batch-plan` warns when a task recording will have a non-active
  assignment after the plan, because the task will not be available to labelers.
- [x] `import-batch-plan` warns when an assignment manifest contains multiple
  rows for the same recording, so the one-user-per-recording owner is explicit.
- [x] `import-batch-plan` warns when an imported assignment changes an existing
  recording owner, because applying it closes active sessions for that recording.
- [x] `import-batch-plan` warns when one recording has tasks across multiple
  workflow kinds, so mixed labeler workload is explicit before launch.
- [x] `import-batch-plan` warns when multiple task IDs point at the same
  recording/workflow/component/run/scope logical work item.
- [x] `import-batch-plan --warnings-as-errors` blocks apply when review warnings
  are present, before assignments or tasks are modified.
- [x] batch-plan reports include compact `issue_codes`, `warning_codes`, and
  blocking warning fields for automation-friendly triage.
- [x] `import-batch-plan --output ...` archives the combined dry-run/apply
  report with no-overwrite protection.
- [x] `import-batch-plan --html-output ...` writes a human-readable review report
  summarizing cross-file issues, assignment rows, closed sessions, task rows, and
  source lines.
- [x] batch-plan HTML reports include summary counts for assignment changes, task
  changes, issues, warnings, and closed sessions.
- [x] batch-plan HTML reports show compact issue, warning, and blocking-warning
  code summaries above the row tables.
- [x] `export-launch-bundle` writes assignment/task snapshots, readiness report,
  all-user handoffs, launch README, top-level `index.html`, manifest, and
  optional ZIP packaging in one operator command.
- [x] `export-launch-bundle --dry-run` reports the planned bundle users, files,
  output state, and readiness without writing files or requiring a link secret.
- [x] launch bundles include `checksums.json` with SHA-256 hashes for generated
  files so copied packages can be audited.
- [x] launch bundles include `inspect-command.txt` with the exact read-only
  inspection command to run after copying or before re-sharing.
- [x] launch bundles include `inspection-targets.json` with machine-readable
  directory/ZIP inspection targets and commands.
- [x] `export-launch-bundle --include-audit-events` adds task, assignment, and
  task-definition audit JSONL files to the launch archive.
- [x] launch-bundle overwrite refuses stale user handoff directories instead of
  silently carrying old labeler files into a regenerated package.
- [x] launch-bundle overwrite refuses stale audit artifacts unless the new
  export explicitly includes audit capture.
- [x] `set-task-state` completes or reopens tasks through audited state changes.
- [x] `set-task-state --output` archives JSON task-state reports with
  no-overwrite protection unless `--overwrite` is supplied.
- [x] `list` lists assignments and tasks.
- [x] `list --output` archives JSON assignment/task listing reports with
  no-overwrite protection unless `--overwrite` is supplied.
- [x] `work-summary` previews the personalized dashboard payload for a user.

### Sessions and handoff

- [x] `list-sessions` lists browser sessions.
- [x] `list-sessions --output` archives JSON session listing reports with
  no-overwrite protection unless `--overwrite` is supplied.
- [x] `cleanup-stale-sessions` closes expired open sessions.
- [x] `cleanup-stale-sessions --output` archives JSON cleanup reports with
  no-overwrite protection before closing sessions unless `--overwrite` is
  supplied.
- [x] `sign-link` emits one signed task link.
- [x] `sign-link --output` archives JSON single-link reports with no-overwrite
  protection unless `--overwrite` is supplied.
- [x] `sign-links` emits active-assignment signed-link manifests.
- [x] `sign-link` and `sign-links` report whether generated URLs are absolute
  and ready to share, with `missing_base_url` warnings for service-relative
  paths.
- [x] `sign-links` reports top-level ready-to-share and not-ready-to-share
  counts for quick operator review of link manifests.
- [x] `sign-link` reports whether the target task is currently launchable under
  active assignment and task state, with warnings for inactive assignments and
  completed tasks.
- [x] `sign-links` reports per-row task launchability, including completed-task
  warnings when `--include-completed` is used.
- [x] `export-user-handoff` writes one labeler's work summary, signed links,
  store-check report, manifest, labeler-facing `index.html`, shareable
  `message.txt`, and browser-only `labeler-quickstart.txt`.
- [x] `message.txt` reflects `ready_to_send`: ready handoffs say work is ready,
  while not-ready handoffs warn labelers to wait for operator review and list
  sendability reasons.
- [x] `labeler-quickstart.txt` reflects `ready_to_send`: not-ready handoffs
  switch from start instructions to preview/wait instructions with sendability
  reasons.
- [x] User-facing task summaries redact server-only task scope, raw zarr paths,
  registry paths, and related filesystem path fields before they are returned
  from `/api/me/tasks`, `work-summary`, or handoff `work-summary.json`.
- [x] User-facing unresolved failed-promotion rows avoid raw training-zarr path
  display and point labelers back to operator/admin repair context before retry.
- [x] labeler-facing handoff HTML mirrors the dashboard priority/task-note
  guidance by showing task priority, task notes, and priority-sorted work.
- [x] labeler-facing handoff HTML reflects `ready_to_send`: not-ready handoffs
  warn labelers to wait for operator review and suppress clickable task links
  unless link rows are ready to share.
- [x] labeler-facing handoff HTML shows assigned recordings with no open task
  rows and distinguishes not-yet-generated work from completed work.
- [x] `export-user-handoff` reports `ready_to_send`, compact
  `sendability_reasons`, and verbose `sendability_warnings` in the per-user
  manifest and CLI payload.
- [x] All-user handoff indexes and rosters include per-labeler
  `sendability_reasons` so operators can triage not-ready handoffs without
  opening individual manifests.
- [x] All-user handoff indexes use top-level `ok` for safe-to-send status and
  `store_checks_ok` for structural store-check success.
- [x] Operator handoff HTML labels structurally valid but unsendable handoffs as
  `not ready` rather than implying the row is OK.
- [x] All-user handoff and launch summaries include aggregate not-ready
  sendability reason counts.
- [x] handoff `signed-links.jsonl` rows report absolute-URL/shareability and
  task-launchability metadata, including completed-task warnings when completed
  tasks are included.
- [x] Handoff manifests, link rows, HTML, messages, and quickstarts include
  generated-at and link-expiration timestamps.
- [x] Handoff HTML, messages, and quickstarts warn when generated without a
  service `--base-url`, so local preview pages do not imply task links are
  directly clickable.
- [x] `sign-link`, `sign-links`, and handoff link rows report exact
  token-derived issued-at and expires-at timestamps plus effective TTL.
- [x] `export-user-handoffs` writes per-labeler handoff directories for all
  assigned users plus top-level `index.json`, `index.html`, and
  `handoff-readme.txt` files.
- [x] batch handoff and launch bundles include `labeler-roster.csv` for
  spreadsheet-friendly operator tracking of users, task counts, messages,
  quickstarts, dashboard URLs, link expiration, ready-to-send status, and
  manifests.
- [x] batch handoff indexes and rosters summarize assigned recordings without
  open task links, no-open-task reason breakdowns, and redacted
  user-summary field counts for operator audit.
- [x] Operator handoff and launch HTML indexes show no-open-task reason
  breakdowns next to the no-open-task totals.
- [x] all-user handoff indexes and launch manifests summarize ready-to-send and
  not-ready-to-send counts with per-user sendability warnings.
- [x] launch bundle manifests, readmes, and HTML indexes include readiness
  no-open-task reason breakdowns from `batch-readiness`.
- [x] launch bundle manifests, readmes, and HTML indexes distinguish
  safe-to-send `handoffs_ok` from structural `handoff_store_checks_ok`.
- [x] `export-user-handoff --zip-output ...` and
  `export-user-handoffs --zip-output ...` package generated handoff directories
  into protected ZIP archives.
- [x] `inspect-handoff` checks exported handoff or launch-bundle directories and
  ZIPs for status, counts, readiness, link expiration, assigned recordings with
  no open task links, no-open-task reason breakdowns, per-labeler and aggregate
  sendability reasons, and redacted user-summary fields before re-sharing.
- [x] `inspect-handoff` returns not-ready packages as `needs_review` with
  `handoff_not_ready` before re-sharing.
- [x] `inspect-handoff` reports launch-bundle `handoff_store_checks_ok`
  separately from safe-to-send `handoffs_ok`.
- [x] `inspect-handoff` verifies launch bundle `checksums.json` when present and
  fails on missing or modified files.
- [x] `inspect-handoff` reports compact top-level `status` and
  `failure_reasons` fields for automation and quick operator triage.

### Task generation

- [x] `generate-keypoint-tasks` creates keypoint review tasks.
- [x] `generate-detect-training-tasks` creates detection training tasks.
- [x] `generate-detect-analysis-tasks` creates detection analysis tasks.
- [x] `generate-subject-mask-tasks` creates subject-mask component tasks.
- [x] task-generation commands report generated-at timestamps plus standardized
  warning counts/codes for skipped registry rows.
- [x] task-generation commands support `--warnings-as-errors` to return nonzero
  when registry rows are skipped, so partial generation runs require review.
- [x] task-generation commands support `--output` and `--overwrite` for archived
  JSON generation reports with no-overwrite protection.

### Audit export

- [x] `export-events` exports task/session/mutation audit events.
- [x] `export-assignment-events` exports recording assignment audit events.
- [x] `export-task-definition-events` exports task definition audit events.
- [x] `export-audit-bundle` writes task, assignment, and task-definition JSONL
  files plus a bundle manifest.

## Operator Workflow Checklist

### Prepare work

- [ ] Select recordings for browser labeling.
- [ ] Assign each recording to exactly one user.
- [ ] Generate the needed task types for those recordings.
- [ ] Archive the current assignment/task plan with `export-assignments` and
  `export-tasks`.
- [ ] Run `batch-readiness` before announcing a multi-user batch and inspect
  active assigned recordings with no task rows or no open tasks.
- [ ] Prefer `export-launch-bundle` when preparing a full launch package.
- [ ] Run `check-store` before sharing links or launching a user test.
- [ ] Treat `ready_to_send` and `ready_to_invite` as work/readiness signals only;
  do not send labeler links until inspection reports
  `labeler_links_safe_to_share=true`.
- [ ] Confirm the preferred labeler entry is the guarded personalized dataset
  queue `/my-datasets?expected_user=<assignment-user>`, and confirm browser saves
  target assigned task/training Zarrs rather than CSV, roster, manifest, HTML,
  JSON, or intermediate handoff artifacts.
- [ ] Create a backup with `backup-store`.

### Share work

- [ ] Prefer the personalized dashboard when users can reach the web service.
- [ ] Use `export-user-handoff` when a labeler needs a simple offline handoff
  bundle containing a work summary and signed links.
- [ ] Use `export-user-handoffs` when preparing handoff bundles for a full
  multi-labeler batch.
- [ ] Use signed links as convenience entry points, not as the only access
  control.
- [ ] Keep link secrets out of email, docs, and shell history where practical.

### Monitor work

- [ ] Use admin task detail pages for task-level inspection.
- [ ] Use event exports for audit review.
- [ ] Use `list-sessions` to inspect active browser sessions.
- [ ] Use `cleanup-stale-sessions` to close abandoned expired sessions.
- [ ] Review failed promotion and registry refresh events separately from saves.

### Recover or change assignments

- [ ] Reassign at the recording level, not at individual frame/ROI level.
- [ ] Expect reassignment to close affected active sessions.
- [ ] Reopen completed tasks only through `set-task-state`.
- [ ] Archive audit events before and after major reassignment batches.

## Auth and Deployment Checklist

Implemented safeguards:

- [x] Local development identity with `--user`.
- [x] Proxy-provided identity with `--auth-header`.
- [x] Header identity ignored unless `--trust-auth-header` is set.
- [x] Admin users configured with repeatable `--admin-user`.
- [x] `--production` rejects fixed-user auth.
- [x] `--production` requires trusted proxy auth.
- [x] `--production` requires at least one admin user.
- [x] Same-origin POST protection enabled by default.
- [x] Controlled `--disable-csrf-check` escape hatch.
- [x] Non-loopback binds require `--allow-non-loopback`.
- [x] Optional JSON access logging with `--access-log`.

Production decisions still required:

- [ ] Choose the production auth boundary.
- [ ] Decide which reverse proxy sets the trusted user header.
- [ ] Decide the canonical header name and allowed user identifier format.
- [ ] Choose host, service account, filesystem mounts, and data permissions.
- [ ] Decide TLS and network restrictions.
- [ ] Decide SQLite backup location and retention.
- [ ] Decide who owns assignment changes and failed-promotion repair.
- [ ] Fill and approve `docs/web_labeling_production_decision_record.md`.

## Rollout Checklist

### Phase 1: local smoke

- [x] Create local assignment store helpers.
- [x] Create local smoke launch helper.
- [ ] Run static checks.
- [ ] Run focused unit checks.
- [ ] Run a local browser smoke with a non-production store.
- [ ] Confirm dashboard, task open, save, completion, and audit events.

### Phase 2: first operator test

- [x] Write first-operator test plan.
- [ ] Select one real recording.
- [ ] Assign exactly one user to that recording.
- [ ] Generate only the needed task type for that recording.
- [ ] Launch behind the intended auth boundary or a controlled substitute.
- [ ] Have the operator complete one task.
- [ ] Inspect audit events and the resulting zarr mutation.
- [ ] Confirm another user cannot access the task.

### Phase 3: production readiness

- [ ] Fill the production decision record.
- [ ] Confirm auth header trust boundary.
- [ ] Confirm host, service account, mounts, and permissions.
- [ ] Confirm TLS/network restrictions.
- [ ] Confirm SQLite backup and restore process.
- [ ] Confirm failed-promotion retry ownership.
- [ ] Run readiness checks outside the Codex sandbox.
- [ ] Run real-zarr smoke using the production-like mount layout.

### Phase 4: broader rollout

- [ ] Add more recordings only after the first-operator test passes.
- [ ] Assign each recording to exactly one user.
- [ ] Monitor access logs and audit events.
- [ ] Periodically back up the assignment store.
- [ ] Review failed promotions and registry refresh failures.
- [ ] Archive audit bundles for milestone handoffs.

## Validation Status

Validation helpers and tests exist, but this document does not claim they have
been run in the current turn.

Available validation helpers:

```bash
scripts/check_labeling_web_static.sh
scripts/check_labeling_web_unit.sh
scripts/check_labeling_web_readiness.sh
```

Relevant test files:

- `tests/unit/fisheye/test_labeling_assignment_store.py`
- `tests/unit/fisheye/test_labeling_signed_links.py`
- `tests/unit/fisheye/test_labeling_web_security.py`
- `tests/unit/fisheye/test_labeling_promotion_retry.py`
- `tests/unit/fisheye/test_labeling_web_routes.py`
- `tests/unit/fisheye/test_labeling_web_config.py`
- `tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py`

Real-zarr and production-like validation should run outside the Codex sandbox per
repository policy.

## Immediate Next Items

- [ ] Run static and unit validation helpers when explicitly approved.
- [ ] Complete the production decision record before non-loopback production use.
- [ ] Run the first-operator test on one real recording.
- [ ] Use `export-user-handoffs` for the first multi-labeler batch after
  assignments and tasks are current.
