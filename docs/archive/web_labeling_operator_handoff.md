<!-- ARCHIVED 2026-07-04: consolidated — current pointer is docs/web_labeling_implementation_status.md. -->

# Web Labeling Operator Handoff

<!-- design-meta
status: handoff
last_updated: 2026-06-23
scope: current readiness state for assigned browser labeling workflow
-->

## Current State

The web-labeling implementation now has the core in-repo pieces needed for
assigned multi-user browser labeling:

- Recording-level assignment ownership.
- Personalized labeler dashboard.
- Admin dashboard and browser-based assignment editing.
- Assignment-gated short-lived sessions.
- Signed task links with expiry and revocation floor.
- Labeler-facing handoff pages, messages, and browser-only quickstarts.
- Same-origin POST protection.
- Explicit trusted-proxy auth and production launch checks.
- Single active writer per task via session supersession.
- Task completion locks.
- Audit events for saves, session changes, retries, promotions, and registry
  refresh outcomes.
- Workflow routes for keypoints, detection training, detection analysis, and
  subject-mask components.
- Failed-promotion retry/idempotency paths for labelers and admins.
- Deployment runbook, production decision record, proxy/systemd examples, smoke
  spec template, and validation helper scripts.

For a machine-readable inventory of implementation files, routes, scripts,
tests, and validation commands, see:

```text
docs/web_labeling_implementation_manifest.json
```

For the first real multi-user batch, use:

```text
docs/web_labeling_first_batch_operator_checklist.md
```

## Remaining Required Decisions

These are intentionally not decided by the codebase:

1. Production auth boundary.

   Decide the real SSO/proxy/gateway and the exact trusted user header.

2. Production host, service account, filesystem mounts.

   Decide where the Palette-capable service runs, which account owns it, and
   which registry/zarr paths are readable or writable.

3. TLS and network restrictions.

   Decide TLS termination, allowed clients/networks, access-log retention, and
   whether Palette remains bound to loopback.

Record these in:

```text
docs/web_labeling_production_decision_record.md
```

## Required Validation Before Operator Testing

These commands have not been run as part of implementation.

Production decision record:

```bash
scripts/py scripts/check_labeling_production_decision_record.py
```

Static compile check:

```bash
scripts/check_labeling_web_static.sh
```

Focused non-zarr unit tests:

```bash
scripts/check_labeling_web_unit.sh
```

Aggregate readiness helper:

```bash
scripts/check_labeling_web_readiness.sh
```

Real-zarr smoke, only after preparing a copied spec from
`docs/web_labeling_real_zarr_smoke_spec.template.json`:

```bash
PALETTE_LABELING_WEB_REAL_ZARR_SMOKE_SPEC=/path/to/copied_web_labeling_real_zarr_smoke_spec.json \
PYTHONPYCACHEPREFIX=/tmp/palette-pycache \
scripts/py -m pytest -p no:cacheprovider \
  tests/integration/fisheye/test_labeling_web_real_zarr_smoke.py -q
```

## Suggested Next Human Step

For a local fixed-user operator pass before production, use:

```text
docs/web_labeling_first_operator_test_plan.md
```

For the first real multi-user handoff after local testing and production
decision-record completion, use:

```text
docs/web_labeling_first_batch_operator_checklist.md
```

For production, fill out the production decision record for the first intended
deployment environment, then run:

```bash
scripts/check_labeling_web_readiness.sh
```

If the decision record is not ready, run the static and unit helpers separately:

```bash
scripts/check_labeling_web_static.sh
scripts/check_labeling_web_unit.sh
```
