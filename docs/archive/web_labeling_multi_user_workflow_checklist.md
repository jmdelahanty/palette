<!-- ARCHIVED 2026-07-04: consolidated — current pointer is docs/web_labeling_implementation_status.md. -->

# Multi-User Web Labeling Workflow Implementation Checklist

<!-- design-meta
status: planning/implementation checklist
last_updated: 2026-06-23
scope: assigned browser labeling for users without local Palette/Crimson installs
-->

## Purpose

This document summarizes the web-labeling workflow direction discussed during
implementation planning. The goal is to let multiple collaborators label or
mutate assigned training/analysis zarrs from a browser, without requiring each
labeler to install Palette or Crimson.

The desired operator experience is:

- An operator assigns recordings and labeling tasks to users.
- Each labeler opens a personalized "my labeling work" page, or receives a
  signed task/handoff link.
- The server enforces assignment, task state, and safe zarr mutation.
- Operators can audit, package, inspect, and recover each batch.

## Core Product Decision

Only one user is assigned to a given recording.

That makes `recording_id` the authorization boundary for the first multi-user
workflow:

```text
recording_id -> assignee_user -> allowed tasks/sessions/mutations
```

This avoids fragile per-frame or per-zarr ownership rules while still supporting
many users in parallel. Different users can work on different recordings, but
not on the same recording at the same time unless the recording is explicitly
reassigned.

## Safety Model

The browser should never be the source of truth for what may be edited.

Implementation requirements:

- Do not trust browser-supplied zarr paths, run names, recording IDs, or task
  scopes as authorization facts.
- Resolve browser requests through server-side task definitions.
- Require authentication for dashboards and signed links.
- Check current recording assignment before opening a session.
- Re-check assignment and task state before every mutation.
- Reject new sessions and saves for completed tasks.
- Close active sessions when a recording is reassigned, paused, or deactivated.
- Audit assignment changes, session lifecycle, saves, reviews, completions,
  promotions, retries, and registry refreshes.

## User Entry Points

### Personalized work page

The main labeler-facing page should answer:

- What recordings are assigned to me?
- What tasks are waiting?
- What workflow does each task use?
- What instructions or notes did the operator give me?
- Which tasks are open, completed, or blocked?

Checklist:

- [x] Serve a personalized dashboard at `/`.
- [x] Expose the same data through `GET /api/me/tasks`.
- [x] Group work by recording and workflow kind.
- [x] Keep active assigned recordings visible even when they currently have no
  open browser-labeling tasks.
- [x] Tell labelers whether an empty assigned recording is waiting on task
  generation or already complete.
- [x] Show assignment notes as labeler-facing instructions.
- [x] Show task notes as task-specific labeler instructions.
- [x] Show priority and put higher-priority visible tasks first within each
  recording.
- [x] Hide tasks not assigned to the authenticated user.
- [x] Provide workflow/search filtering for larger batches.
- [x] Provide a refresh control so labelers can reload current server-side
  assignments after an operator change.
- [x] Tell labelers directly on the dashboard that no local Palette/Crimson
  installation is needed and that zarrs must not be edited directly.

### Signed links and handoffs

Signed links are convenience routing, not authorization by themselves. A valid
link should only resolve if the authenticated user is still the current assignee
for the linked task's recording.

Checklist:

- [x] Generate opaque task links.
- [x] Include issued-at and expiration metadata.
- [x] Support revocation through a link-revocation floor.
- [x] Require live authenticated identity after link validation.
- [x] Re-check current assignment before opening the linked task.
- [x] Generate per-user and all-user handoff bundles.
- [x] Include labeler-facing HTML, message text, and browser-only quickstarts in
  handoff packages.
- [x] Include priority and task-note guidance in labeler handoff previews.
- [x] Include generated-at and expiration timestamps so stale links are visible.

## Operator Workflow

### 1. Prepare assignments

Operator creates a recording assignment plan.

Checklist:

- [x] Store one active assignee per recording.
- [x] Track assigned-by, assigned-at, status, and notes.
- [x] Support JSON, JSONL, and CSV assignment import.
- [x] Support CSV template generation for spreadsheet planning.
- [x] Make unchanged assignment imports idempotent.
- [x] Close affected sessions when assignment ownership/status changes.
- [x] Export current assignment snapshots for archive.
- [x] Export assignment audit history.

### 2. Prepare task definitions

Operator creates task definitions under assigned recordings.

Checklist:

- [x] Store stable task IDs.
- [x] Store recording, workflow kind, dataset/zarr references, run/component
  metadata, scope, priority, state, and notes.
- [x] Support JSON, JSONL, and CSV task import.
- [x] Support CSV task templates with JSON `scope`/`scope_json` cells.
- [x] Make unchanged task imports idempotent.
- [x] Export current task snapshots for archive.
- [x] Export task-definition audit history.

Supported workflow kinds:

- [x] `keypoints`
- [x] `detect_training`
- [x] `detect_analysis`
- [x] `subject_mask_component`

### 3. Review the batch plan before launch

The batch plan should be checked before labelers receive links.

Checklist:

- [x] Dry-run assignment and task imports together.
- [x] Validate that every task recording has an assignment after the plan.
- [x] Warn when task recordings are assigned but inactive.
- [x] Warn when one recording mixes workflow kinds.
- [x] Warn when multiple task IDs appear to represent duplicate logical work.
- [x] Support `--warnings-as-errors` for stricter launch automation.
- [x] Archive JSON and HTML batch-plan reports.
- [x] Include compact issue and warning codes for triage.

### 4. Launch the batch

Operator packages the batch for users.

Checklist:

- [x] Generate a readiness report.
- [x] Generate one-command launch bundles.
- [x] Include assignment/task snapshots.
- [x] Include per-user handoffs.
- [x] Include a top-level HTML index.
- [x] Include a labeler roster CSV.
- [x] Include package checksums.
- [x] Include inspect commands and machine-readable inspection targets.
- [x] Support optional ZIP packaging.
- [x] Refuse stale handoff/audit overwrite cases unless explicitly safe.

### 5. Labelers complete work

Labelers use browser sessions rather than local installations.

Checklist:

- [x] Open tasks from dashboard or signed link.
- [x] Use short-lived server-side session IDs.
- [x] Keep one current writer per task.
- [x] Supersede older open sessions when a newer session opens.
- [x] Navigate only within server-defined task scope.
- [x] Save edits through workflow-specific backend handlers.
- [x] Record audit events for every mutation.
- [x] Mark tasks complete when finished.

### 6. Operators monitor and recover

Operators need visibility and recovery tools after launch.

Checklist:

- [x] Admin dashboard at `/admin`.
- [x] Read-only task detail pages.
- [x] Session listing and stale-session cleanup.
- [x] Store consistency checks.
- [x] Failed-promotion retry support.
- [x] Audit bundle export.
- [x] Handoff/launch package freshness inspection.
- [x] SQLite sidecar store backup.

## Zarr Mutation Policy

Mutable and immutable artifacts should be kept distinct.

Checklist:

- [x] Treat per-recording training zarrs as mutable curated workspaces.
- [x] Treat unified/exported training artifacts as immutable build outputs.
- [x] Route detection-analysis curation through audited save events.
- [x] Record promotion from analysis edits to training zarrs separately.
- [x] Support idempotent retry for failed promotions.
- [x] Route subject-mask editing through unified subject-mask tooling rather
  than adding new eye-mask-first surfaces.

## Authentication and Deployment Boundary

The service should normally run behind an authenticated proxy in production.

Checklist:

- [x] Support fixed-user auth only for local loopback testing.
- [x] Support trusted-header auth behind a proxy.
- [x] Require explicit `--trust-auth-header` before trusting auth headers.
- [x] Require production launches to use trusted-header auth and admin users.
- [x] Require explicit opt-in for non-loopback binds.
- [x] Keep same-origin POST protection enabled by default.
- [x] Document proxy requirements for stripping inbound auth headers.

Open deployment decisions:

- [ ] Choose production authentication provider and proxy boundary.
- [ ] Choose production host, service account, registry path, and zarr mounts.
- [ ] Decide where the sidecar assignment SQLite store is backed up.
- [ ] Decide where mutable per-recording training zarr backups live.
- [ ] Complete a first-operator test with real labeler identity behavior.

## Minimum End-to-End Acceptance Criteria

Before considering the workflow production-ready:

- [ ] An operator can import assignments and tasks from spreadsheet-friendly
  CSV manifests.
- [ ] A dry-run batch plan catches missing assignments and duplicate logical
  tasks before mutation.
- [ ] A labeler can open `/` and see only their assigned recordings.
- [ ] A labeler can open a signed task link only when they are still assigned to
  that recording.
- [ ] Reassigning a recording closes the previous user's active sessions.
- [ ] A completed task cannot be reopened for mutation without an audited reopen.
- [ ] A workflow save writes through the intended Palette backend path and emits
  an audit event.
- [ ] The operator can export the current batch state, audit bundle, and
  launch/handoff package.
- [ ] A copied handoff or launch bundle can be inspected for staleness and
  checksum mismatches.
- [ ] A real zarr smoke test has been run outside the Codex sandbox for the
  workflows included in the launch.

## Related Files

- `docs/web_labeling_assignment_implementation_checklist.md`
- `docs/web_labeling_deployment_runbook.md`
- `docs/web_labeling_operator_handoff.md`
- `docs/web_labeling_first_operator_test_plan.md`
- `docs/web_labeling_production_decision_record.md`
- `docs/web_labeling_implementation_manifest.json`
- `src/fisheye/labeling/assignment_store.py`
- `src/fisheye/labeling/web.py`
- `src/fisheye/utils/labeling_work.py`
