# Web Labeling First Operator Test Plan

<!-- design-meta
status: test-plan
last_updated: 2026-06-23
scope: first local/operator evaluation before production deployment
-->

## Purpose

Use this plan for the first hands-on evaluation of the assigned browser labeling
workflow before production deployment decisions are finalized.

This is not a production launch plan. It uses loopback fixed-user mode so one
operator can confirm the dashboard, assignment store, task opening, session
locking, and admin surfaces before connecting real SSO or sharing links with
labelers.

## Preconditions

- Work from the Palette repository root.
- Use `scripts/py`; do not activate conda manually.
- Use a temporary sidecar store for the first pass.
- Use backed-up or disposable mutable zarrs for any save-path test.
- Do not expose the local fixed-user server beyond loopback.

## Static and Focused Non-Zarr Checks

Run these before opening the browser:

```bash
scripts/check_labeling_web_static.sh
```

```bash
scripts/check_labeling_web_unit.sh
```

These commands have not been run by the implementation agent.

## Local Loopback Smoke

Create a temporary store with sample assignments and placeholder tasks:

```bash
store=/tmp/palette_labeling_first_operator_test.sqlite
scripts/setup_labeling_web_local_smoke_store.sh "$store"
```

The helper creates:

- `recording-a` assigned to `alice`.
- `recording-b` assigned to `bob`.
- One keypoint placeholder task for Alice.
- One detection placeholder task for Bob.

The placeholder tasks exercise dashboard/admin/session behavior. Replace their
scope paths or use the real-zarr smoke spec before testing workflow save paths.

Start Alice's local service:

```bash
scripts/start_labeling_web_local_smoke.sh "$store" alice 8795
```

Browser checks:

- Open `http://127.0.0.1:8795/`.
- Confirm only `recording-a` appears.
- Confirm assignment notes are visible.
- Confirm dashboard search/workflow filters work.
- Open `/admin`.
- Confirm assignments, preflight, and task counts are visible.
- Reassign `recording-a` to `bob` from `/admin`.
- Confirm Alice's old session/dashboard no longer has access to that recording
  after refresh.

Stop the service, then repeat with `--user bob`:

```bash
scripts/start_labeling_web_local_smoke.sh "$store" bob 8795
```

Confirm Bob sees `recording-a` after reassignment and `recording-b` if both are
active assignments for Bob.

## Signed Link Local Check

Generate a signed link for a task assigned to the current fixed user:

```bash
scripts/py -m fisheye.utils.labeling_work --store "$store" sign-link \
  --task-id task-alice-keypoints \
  --link-secret local-test-secret \
  --base-url http://127.0.0.1:8795
```

Expected behavior:

- The link opens only for the user currently assigned to the recording.
- Reassignment changes link behavior because session creation is assignment
  gated.
- Completed tasks cannot be reopened through signed links.

## Real-Zarr Save Smoke

After the local dashboard/admin behavior is acceptable, copy and fill in:

```text
docs/web_labeling_real_zarr_smoke_spec.template.json
```

Then run outside the Codex sandbox:

```bash
PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC=/path/to/copied_web_labeling_real_zarr_smoke_spec.json \
PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
scripts/py -m pytest -p no:cacheprovider \
  tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py -q
```

Use only disposable or backed-up mutable zarrs.

## Production Readiness

Do not use the local fixed-user launch shape for production.

Before production, fill out:

```text
docs/web_labeling_production_decision_record.md
```

Then run:

```bash
scripts/check_labeling_web_readiness.sh
```
