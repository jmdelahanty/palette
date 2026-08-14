# Registered-dish geometry review web operations

## Purpose and authority boundaries

`apps/marimo/geometry_review.py` is an operator-facing, read-only evidence
viewer. The Palette SQLite registry supplies the queue and canonical analysis
Zarr paths. The selected canonical Zarr supplies immutable scientific evidence.
Campaign staging receipts are not read.

The app has no controls or code paths for recording review decisions. It does
not select geometry, publish candidates, change registry stage state,
materialize comparisons or gates, trigger refinement, or write a Zarr. A
reviewer copies the exact run IDs and digests from the handoff panel into the
separate pipeline operation appropriate to the reviewed decision.

The registry contract remains unchanged. `recording_step_status.status` is one
of `ok`, `missing`, `absent`, `na`, or `error`. Human review state is read from
`review_status_json`; the viewer neither expects nor accepts `status="review"`.

## Launching the viewer

Registry mode is the normal operator mode. It queries SQLite in read-only mode,
does not recursively scan `/groups`, and opens only the Zarr selected in the
recording dropdown.

```bash
scripts/run_geometry_review.sh \
  --registry /nvme1/palette_registry.sqlite
```

An exact dataset and immutable run can be selected at launch:

```bash
scripts/run_geometry_review.sh \
  --registry /nvme1/palette_registry.sqlite \
  --dataset-id DATASET_ID \
  --run-id arena-geometry-fit-review-EXACT_ID
```

Direct mode is for one explicit development or diagnostic archive:

```bash
scripts/run_geometry_review.sh \
  --zarr-path /path/to/recording_analysis.zarr \
  --run-id arena-geometry-fit-review-EXACT_ID
```

The app defaults to `127.0.0.1:8772`. Configure binding and Marimo access with:

```bash
export PALETTE_GEOMETRY_REVIEW_HOST=127.0.0.1
export PALETTE_GEOMETRY_REVIEW_PORT=8772
export PALETTE_GEOMETRY_REVIEW_TOKEN='use-an-operator-managed-secret'
```

Published archives are opened with consolidated metadata explicitly enabled.
Missing or stale consolidated metadata is reported as a publication defect;
the app does not silently fall back to unconsolidated reads. An active-campaign
unconsolidated diagnostic mode is intentionally not included in this version.

If multiple complete pending fit-review runs exist, the viewer displays an
ambiguity warning and requires an exact choice. It never selects the newest run.
Every artifact is resolved through `review_record.artifacts`, then checked for
safe path, media type, size, byte length, node metadata, SHA-256, and PNG/JSON
structure before rendering.

## SSH tunnel

Keep the service bound to loopback unless the deployment environment supplies
an authenticated reverse proxy. From an operator laptop with campus/VPN SSH
access:

```bash
ssh -N -L 8772:127.0.0.1:8772 USER@PALETTE_WORKSTATION
```

Keep that shell open and visit `http://127.0.0.1:8772`. Provide the Marimo token
through the browser prompt when one is configured.

## One-shot notification scanner

Email is never sent by loading or refreshing the web page. The independent
scanner performs one registry read and exits:

```bash
export PALETTE_GEOMETRY_REVIEW_NOTIFICATION_TO='operator1@example.org,operator2@example.org'
export PALETTE_GEOMETRY_REVIEW_NOTIFICATION_STATE_DB="$HOME/.palette/geometry_review_notifications.sqlite"
export PALETTE_LABELING_NOTIFICATION_MODE=outbox
export PALETTE_LABELING_NOTIFICATION_OUTBOX="$HOME/.palette/geometry_review_outbox"
export PALETTE_LABELING_BASE_URL='http://127.0.0.1:8772'

scripts/py -m fisheye.utils.scan_geometry_review_notifications \
  --registry /nvme1/palette_registry.sqlite
```

The state database is operational data and must stay outside the canonical
registry and all analysis Zarrs. Its event key binds dataset ID, exact run,
available scientific digest (or an exact registry-state fingerprint), stage,
and semantic state. Successfully queued or sent events are
durably deduplicated. Disabled, dry-run, skipped, and failed deliveries are
recorded but remain eligible for a later successful send. Multiple new
recordings are combined into one digest. Ordinary missing upstream data and
still-running work do not trigger notifications.

Test a scan without transport or consuming deduplication state:

```bash
scripts/py -m fisheye.utils.scan_geometry_review_notifications \
  --registry /nvme1/palette_registry.sqlite \
  --dry-run
```

Disable transport explicitly:

```bash
export PALETTE_LABELING_NOTIFICATION_MODE=disabled
```

For SMTP, configure the existing Palette notification transport; do not place
credentials or recipients in the repository:

```bash
export PALETTE_LABELING_NOTIFICATION_MODE=smtp
export PALETTE_LABELING_NOTIFICATION_FROM='Palette Geometry <palette@example.org>'
export PALETTE_LABELING_SMTP_HOST='smtp.example.org'
export PALETTE_LABELING_SMTP_PORT=587
export PALETTE_LABELING_SMTP_STARTTLS=true
export PALETTE_LABELING_SMTP_USERNAME='managed-account'
export PALETTE_LABELING_SMTP_PASSWORD='managed-secret'
```

For cron, use an operator-owned environment file and absolute paths. For
example, every five minutes:

```cron
*/5 * * * * cd /path/to/palette && scripts/py -m fisheye.utils.scan_geometry_review_notifications --registry /nvme1/palette_registry.sqlite >>/var/tmp/palette-geometry-review-notifications.log 2>&1
```

A systemd service should use `Type=oneshot`, an `EnvironmentFile` readable only
by the service account, and a timer unit. Set `WorkingDirectory` to the pinned
Palette checkout and execute the same `scripts/py -m ...` command. Scanner
delivery results are operational audit records only and never mark scientific
work complete.

## Validation and read-only smoke

Static and deterministic checks:

```bash
scripts/py -m py_compile <changed-python-files>
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_geometry_review_evidence.py \
  tests/unit/fisheye/test_geometry_review_registry_notifications.py -q
scripts/py -m marimo check apps/marimo/geometry_review.py
git diff --check
```

The production smoke fixture is inspected read-only. A smoke should record the
exact run, review-record digest, montage and panel bindings, and confirm that
the root and `zarr.json` modification times did not change. Never use an append
or update mode for this smoke.
