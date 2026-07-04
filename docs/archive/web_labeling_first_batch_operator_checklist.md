<!-- ARCHIVED 2026-07-04: consolidated — current pointer is docs/web_labeling_implementation_status.md. -->

# Web Labeling First Multi-User Batch Checklist

<!-- design-meta
status: operator checklist
last_updated: 2026-06-23
scope: first real assigned multi-user browser-labeling batch
-->

## Purpose

Use this checklist for the first real batch where multiple labelers receive
assigned browser-labeling work. It assumes the local first-operator pass has
already been completed and the production auth/network decisions are recorded.

This checklist is intentionally operational. It is the shortest path from "I
have recordings and labelers" to "labelers can safely work in the browser and I
can audit/recover the batch."

## Preconditions

- The service host can read the source registry/zarrs needed for display.
- The service host can write only the mutable zarrs intended for curation.
- Mutable per-recording training zarrs have backups or disposable copies.
- The production auth boundary is recorded in
  `docs/web_labeling_production_decision_record.md`.
- The service is not launched in fixed-user mode for real labelers.
- Every recording in the first batch has exactly one intended assignee.

## 1. Create Spreadsheet-Friendly Manifests

Generate starter templates:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  write-manifest-templates --output-dir /path/to/first-batch/templates
```

Fill in:

- `assignments-template.csv`
- `tasks-template.csv`

Rules:

- One active assignee per `recording_id`.
- Use assignment notes for labeler-facing instructions.
- Use stable `task_id` values that are meaningful to operators.
- Put complex task scope into `scope` or `scope_json` as JSON.
- Leave no partially filled rows in the CSV files.

## 2. Dry-Run the Batch Plan

Run the combined import in dry-run mode first:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  import-batch-plan \
  --assignments /path/to/first-batch/assignments.csv \
  --tasks /path/to/first-batch/tasks.csv \
  --output /path/to/first-batch/batch-plan-dry-run.json \
  --html-output /path/to/first-batch/batch-plan-dry-run.html \
  --warnings-as-errors
```

Review the HTML report before applying.

Block launch if the report shows:

- missing recording assignments for tasks
- inactive assignments for task recordings
- duplicate logical tasks
- unexpected mixed workflow kinds on the same recording
- unexpected reassignment session closures

## 3. Apply the Batch Plan

Apply only after the dry-run report is reviewed:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  import-batch-plan \
  --assignments /path/to/first-batch/assignments.csv \
  --tasks /path/to/first-batch/tasks.csv \
  --apply \
  --assigned-by OPERATOR \
  --actor OPERATOR \
  --output /path/to/first-batch/batch-plan-applied.json \
  --html-output /path/to/first-batch/batch-plan-applied.html \
  --warnings-as-errors
```

Expected result:

- assignments are applied before tasks
- unchanged rows are idempotent
- changed assignments close affected sessions
- task definitions are audit logged

## 4. Check Batch Readiness

Archive a readiness report:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  batch-readiness \
  --output /path/to/first-batch/batch-readiness.json \
  --warnings-as-errors
```

Resolve hard issues before sharing links.

Do not share labeler links from readiness alone. `ready_to_invite=true` and
`ready_to_send=true` mean the row or handoff has launchable work, but they are
not the final safe-share decision. Before any labeler receives a link, the batch
must pass handoff inspection with `labeler_links_safe_to_share=true`.

Readiness warnings should be intentional, especially:

- active assignments with no open tasks
- users with no launchable work
- active sessions from earlier operator testing
- incomplete tasks under paused assignments

## 5. Export the Launch Bundle

Generate the complete operator package:

```bash
PALETTE_LABELING_LINK_SECRET='<secret-from-secret-store>' \
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-launch-bundle \
  --base-url https://labeling.example.org \
  --output-dir /path/to/first-batch/launch-bundle \
  --zip-output /path/to/first-batch/launch-bundle.zip \
  --include-audit-events \
  --warnings-as-errors
```

The bundle should contain:

- `assignments.json`
- `tasks.json`
- `batch-readiness.json`
- `handoffs/`
- `handoffs/labeler-roster.csv`
- per-user `index.html`
- per-user `message.txt`
- per-user `labeler-quickstart.txt`
- top-level `index.html`
- `launch-readme.txt`
- `checksums.json`
- `inspect-command.txt`
- `inspection-targets.json`
- optional `audit/*.jsonl`

## 6. Inspect the Bundle Before Sharing

Run the generated inspection command, or inspect directly:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  inspect-handoff --path /path/to/first-batch/launch-bundle --require-shareable
```

Do not share the package if inspection reports:

- checksum mismatch
- missing expected files
- stale generated-at timestamps
- expired or near-expired signed links
- launch metadata that does not match the current batch

## 7. Send Labeler Messages

Use each generated per-user `message.txt` as the starting point, and include
`labeler-quickstart.txt` for first-time labelers.

If this batch only needs the authenticated personalized dashboard and not
per-task signed links, generate and review a dashboard-only roster instead:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  dashboard-roster --base-url https://labeling.example.org \
  --output /path/to/first-batch/dashboard-roster.json
```

Use `--format html --output dashboard-roster.html` for a browser-readable review
page with copyable ready-row draft text.

Before sending:

- confirm the recipient matches the assignee in `labeler-roster.csv`
- for dashboard-only rollout, confirm `ready_to_invite` is true in
  `dashboard-roster.json`
- for dashboard-only rollout, confirm the roster or status report exposes no
  remaining safe-share blockers, and do not send if
  `safe_share_next_action_summary` lists missing or unsatisfied evidence
- for dashboard-only rollout, inspect `invite_reasons` and do not send rows with
  `missing_base_url`, `no_active_recordings`, or `no_open_tasks`; if the roster
  has `no_users`, fix assignments or filters before contacting labelers
- for dashboard-only rollout, use the roster's per-user `invitation_message`
  field only as ready-row draft text after confirming the recipient matches the
  assigned user and safe-share inspection passes
- confirm `ready_to_send` is true in `labeler-roster.csv`, then treat it only as
  handoff row readiness
- run package inspection with `--require-shareable` and confirm
  `labeler_links_safe_to_share=true`
- confirm the handoff `sendability_warnings` list is empty
- confirm all safe-share evidence gates are passed: mutable Zarr backup,
  browser response-security headers, identity-source verification,
  representative browser smoke, disposable-Zarr mutation smoke, and
  operator-recovery contract evidence
- confirm the preferred link is the guarded personalized dataset queue
  `/my-datasets?expected_user=<assignment-user>`; use `/datasets` and `/work`
  only as fallbacks
- confirm the dashboard URL uses the production host
- regenerate with `--base-url` before sending if the dashboard URL is blank
- confirm the signed-link expiration is acceptable
- confirm the quickstart says no Palette/Crimson install and no direct zarr edits
- confirm browser saves target the assigned task/training Zarr and not any CSV,
  roster, manifest, HTML, JSON, or intermediate handoff artifact
- do not send another labeler's handoff file or link

Recommended first-batch rollout:

- start with one friendly labeler
- confirm they can open `/` and see only their assigned recording
- confirm they can open one task
- confirm a save/review path works on a backed-up mutable zarr
- then send the remaining labeler messages

## 8. Monitor During Labeling

Use the admin dashboard and CLI reports during the batch:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  list-sessions
```

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  check-store
```

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-audit-bundle --output-dir /path/to/first-batch/audit-snapshot
```

Watch for:

- stale sessions
- failed promotions
- unexpected reassignment
- task saves by the wrong authenticated user
- incomplete work under paused assignments

## 9. Recover or Reassign Safely

If a labeler should stop working on a recording:

- pause or reassign the recording
- confirm affected sessions close
- regenerate links or handoffs for the new assignee
- archive the audit bundle before and after the change

If a promotion fails:

- inspect the failed-promotion event from `/admin`
- retry only after the underlying zarr/registry issue is understood
- keep the retry event audit trail

## 10. Close and Archive the Batch

At batch close:

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-assignments --format json \
  --output /path/to/first-batch/final-assignments.json
```

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-tasks --format json \
  --output /path/to/first-batch/final-tasks.json
```

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  export-audit-bundle --output-dir /path/to/first-batch/final-audit
```

```bash
scripts/py -m fisheye.utils.labeling_work --store /path/to/labeling_work.sqlite \
  backup-store --output /path/to/first-batch/labeling-work.sqlite.backup
```

Archive together:

- original CSV manifests
- dry-run and applied batch-plan reports
- launch bundle and ZIP
- final assignment/task snapshots
- final audit bundle
- sidecar SQLite backup
- notes about any manual recovery/reassignment

## First-Batch Success Criteria

- [ ] Every labeler sees only their assigned recording(s) on `/`.
- [ ] Signed links resolve only for the current assigned user.
- [ ] At least one real save/review path is confirmed on backed-up mutable zarrs.
- [ ] Reassignment closes the previous user's active sessions.
- [ ] Completed tasks reject new mutation.
- [ ] Operator can inspect recent audit events for saves and completions.
- [ ] Operator can export a final audit bundle and sidecar backup.
- [ ] No raw zarr paths or direct filesystem instructions are sent to labelers;
  user-facing work summaries show redaction markers instead of server-only task
  scopes or filesystem paths.

## Related Documents

- `docs/web_labeling_multi_user_workflow_checklist.md`
- `docs/web_labeling_deployment_runbook.md`
- `docs/web_labeling_first_operator_test_plan.md`
- `docs/web_labeling_production_decision_record.md`
- `docs/web_labeling_operator_handoff.md`
