# Web Labeling Validation Log Template

Use this template for a validation run before inviting labelers to browser-only
work. Copy the headings into the batch notes or archive this file with the
launch bundle.

## Batch

- Date:
- Operator:
- Store path:
- Base URL:
- Validation environment:
- Link secret source:
- Auth mode:
- Auth header, if proxy-backed:
- Admin users:

## Backups

- Sidecar store backup:
- Mutable zarr backup location:
- Backup manifest:
- Restore command or procedure:

## Focused Static And Unit Validation

Command:

```bash
scripts/check_labeling_web_readiness.sh
```

Result:

- Exit code:
- Report path:
- Warnings:
- Failures:
- Follow-up:

Individual checks, if run separately:

```bash
scripts/py scripts/check_labeling_production_decision_record.py
scripts/check_labeling_web_static.sh
scripts/check_labeling_web_unit.sh
```

## Browser Smoke Validation

Local service command:

```bash
scripts/py -m fisheye.utils.labeling_work --store /tmp/palette-labeling-smoke.sqlite \
  serve --host 127.0.0.1 --port 8795 --user alice --admin-user alice
```

Checks:

- `/work` shows the expected user:
- No-assignment empty state appears:
- `/admin` preflight loads:
- `/admin` assignment work states load:
- `/admin` session summaries load:
- `/admin` audit summaries load:
- Browser workflow contracts load:
- Task open creates a guarded session:
- Failed API state shows copyable support details:

Result:

- Pass/fail:
- Screenshots or notes:
- Follow-up:

## Browser Response Security Headers

Capture response headers from the deployed browser/proxy path as an authenticated
test labeler. Use `/datasets` or `/api/me/tasks`; do not rely only on local
service output if labelers will use a reverse proxy.

Captured URL:

Authenticated test labeler:

Capture method:

Checks:

- `Cache-Control: no-store, no-cache, must-revalidate, max-age=0`:
- `Pragma: no-cache`:
- `Expires: 0`:
- `X-Frame-Options: DENY`:
- `X-Content-Type-Options: nosniff`:
- `Referrer-Policy: no-referrer`:
- `Content-Security-Policy` preserves frame/base/form/object restrictions:
- `Permissions-Policy` disables camera, microphone, and geolocation:
- Proxy strips or weakens none of the above:

Result:

- Pass/fail:
- Evidence file or screenshot:
- Follow-up:

## Dataset Queue Start Readiness

Use the generated `validation-checklist.json`, handoff roster, or
`inspect-handoff` output to confirm queue-state launchability before invitations
are sent.

Checks:

- `dataset_queue_start_readiness` gate status:
- `dataset_queue_blocked_start_users` is empty:
- Every invite-ready labeler has `dataset_queue_state.blocks_labeler_start=false`:
- If any queue is blocked, operator repair or stop-labeling decision:

Result:

- Pass/fail:
- Evidence file or screenshot:
- Follow-up:

## Real-Zarr Smoke Validation

Spec path:

```bash
PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC=/path/to/web_labeling_smoke.json \
PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
scripts/py -m pytest -p no:cacheprovider \
  tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py -q
```

Cases:

| Case | Recording | User | Workflow | Zarr | Result |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

Result:

- Exit code:
- Mutation event IDs:
- Registry refresh events:
- Zarr backup used:
- Follow-up:

## One-Operator / One-Labeler Dry Run

- Operator:
- Labeler:
- Recording:
- Assignment status:
- Task ID:
- Workflow:
- Dashboard URL:
- Expected-user dashboard URL:

Checks:

- Labeler sees only assigned recording:
- Labeler opens task:
- Labeler saves one edit:
- Labeler completes task:
- `/admin` shows task complete:
- Old task/session link cannot mutate completed work:
- Audit summary shows expected event:
- Status report archived:

Result:

- Pass/fail:
- Event IDs:
- Follow-up:

## Multi-User Dry Run

Users and recordings:

| User | Recording | Workflow | Expected state |
| --- | --- | --- | --- |
|  |  |  |  |

Checks:

- Each recording has exactly one active owner:
- Each user sees only assigned recordings:
- Ready-row draft text uses `expected_user_dashboard_url`:
- Reassignment closes/supersedes old sessions:
- Old tab is blocked by assignment/session guard:
- Completed task is read-only until operator reopen:
- Status report archived:
- Audit bundle archived:

Result:

- Pass/fail:
- Event IDs:
- Follow-up:

## Assignment Transition Evidence

Use this section when assigning, reassigning, pausing, or reactivating a
recording during validation or rollback. Copy the relevant fields from the
`/api/admin/assignments` response.

- Assignment API response archived:
- Recording:
- Previous assignee:
- Previous status:
- New assignee:
- New status:
- Owner changed:
- Status changed:
- Closed session IDs:
- Old browser tab rejected after transition:
- Follow-up:

## Rollback Drill

Incorrect assignment drill:

- Recording:
- Original assignee:
- Corrected assignee:
- Assignment event IDs:
- Old sessions closed:

Bad mutation drill:

- Recording:
- Task:
- Event ID:
- Backup restored or corrective edit applied:
- Registry refresh result:

Result:

- Pass/fail:
- Follow-up:

## Final Sign-Off

- Operator confirms identity source matches `assignee_user` values:
- Operator confirms one active owner per recording:
- Operator confirms browser workflows are in first-rollout scope:
- Operator confirms status report has no launch-blocking warnings:
- Operator confirms rollback path is documented and backups exist:
- Operator confirms `ready_to_invite` / `ready_to_send` were treated as
  readiness signals only, not final link-sharing approval:
- Operator confirms handoff inspection with `--require-shareable` reported
  `labeler_links_safe_to_share=true`:
- Safe-share inspection command:
- Safe-share inspection report path:
- Operator confirms preferred labeler entry is guarded
  `/my-datasets?expected_user=<assignment-user>`:
- Operator confirms browser saves target assigned task/training Zarrs and not
  CSV, roster, manifest, HTML, JSON, or intermediate handoff artifacts:
- Safe-share approved to contact labelers (`labeler_links_safe_to_share=true`):
