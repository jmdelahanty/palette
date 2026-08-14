# Registered-dish geometry review web operations

## Purpose and authority boundaries

`apps/marimo/geometry_review.py` is an operator-facing evidence viewer with an
explicitly opt-in approval launcher. The Palette SQLite registry supplies the
queue and canonical analysis Zarr paths. The selected canonical Zarr supplies
immutable scientific evidence. Campaign staging receipts are not read.

The default server mode is read-only. In `dry-run` mode, a confirmed browser
decision can write only a content-addressed approval request and LSF plan to a
durable operations directory. In `submit` mode, the browser persists that same
request and submits a four-job, commit-pinned LSF workflow. The browser process
never opens the canonical Zarr for writing and never writes registry status.

The workflow is:

1. revalidate the exact registry item, fit-review digest, acquisition
   candidate, raw-detection source, recording/camera/arena identity, and Palette
   commit;
2. publish the reviewed Palette candidate, immutable comparison,
   comparison-bound operator selection, and exact keyed centroid gate;
3. run detection quality and required-gate refinement over the unchanged raw
   detections;
4. rescan the canonical Zarr against a node-local registry shadow, validate the
   complete SQLite database, and atomically publish it only if the canonical
   registry has not changed.

Crop publication is intentionally deferred. A later crop workflow can consume
the exact finalized refined-detection run without rerunning raw detection,
geometry review, quality, or refinement.

The small publication job requests one CPU and 8 GB. Quality and refinement use
the existing production resource envelopes. The final registry job
never performs in-place SQLite writes on the multi-host shared filesystem. It
preserves an immutable pre-write registry backup below the approval run,
requires complete SQLite integrity and foreign-key checks before and after the
local mutation, rejects any concurrent canonical change, and publishes one
fully validated database by atomic rename. A different active
selection, stale source digest, incomplete/reordered gate, registry binding
mismatch, dirty deployment, or missing required-CI assertion fails closed.

The registry contract remains unchanged. `recording_step_status.status` is one
of `ok`, `missing`, `absent`, `na`, or `error`. Human review state is read from
`review_status_json`; the viewer neither expects nor accepts `status="review"`.

## Launching the viewer

Registry mode is the normal operator mode. It queries SQLite in read-only mode,
does not recursively scan `/groups`, and opens only the Zarr selected in the
recording dropdown. By default it contains only actionable registry rows—those
waiting for review or carrying a geometry error. Completed and merely running
recordings are not mixed into the operator queue.

```bash
scripts/run_geometry_review.sh \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite
```

For diagnostics only, include inactive geometry states explicitly:

```bash
scripts/run_geometry_review.sh \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --include-inactive true
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

Direct mode never exposes approval controls because it has no registry-backed
actionability or dataset identity.

## Enabling approval

Launch `dry-run` first to validate the exact decision request and dependency
plan without submitting LSF or changing the canonical Zarr/registry:

```bash
export PALETTE_GEOMETRY_REVIEW_TOKEN='use-an-operator-managed-secret'

scripts/run_geometry_review.sh \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --approval-mode dry-run \
  --palette-repo /groups/johnson/johnsonlab/jeremy/palette-deployments/EXACT_COMMIT \
  --approval-root /groups/johnson/johnsonlab/jeremy/operations/palette_geometry_review_approvals \
  --reviewer operator@example.org
```

The approval root is durable operational state. Do not place it under a
campaign staging tree or inside an analysis Zarr. It contains immutable
requests, exact workflow plans, submission receipts, status receipts, logs,
publication results, and final registry-refresh results.
Each submitted run also contains
`registry_backups/palette_registry_before_<request-id>.sqlite`; retain that
backup with the approval audit record.

Only after every required CI job is successful for the exact deployed commit,
launch submit mode:

```bash
scripts/run_geometry_review.sh \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --approval-mode submit \
  --required-ci-success true \
  --palette-repo /groups/johnson/johnsonlab/jeremy/palette-deployments/EXACT_COMMIT \
  --approval-root /groups/johnson/johnsonlab/jeremy/operations/palette_geometry_review_approvals \
  --submit-host login1-citrus-poller \
  --reviewer operator@example.org
```

The server launch refuses either approval mode without a Marimo token. Submit
mode also refuses a dirty Palette deployment or an absent explicit required-CI
assertion. The Palette deployment, approval root, authoritative registry, and
analysis Zarr must all be visible from the LSF environment.

The form requires an exact acquisition candidate and raw detection run, one of
the two operational choices (Palette or acquisition), a reviewed edge-identity
classification, reviewer, reason, and typed `SELECT PALETTE` or
`SELECT ACQUISITION` confirmation. The choice controls only the
bounding-box-centroid gate. It does not change or reinterpret the physical
inner-rim raster-mask authority.

Requests are content-addressed. Repeating the identical submitted request
returns its existing submission receipt instead of launching duplicate jobs.
An incomplete or failed prior submission is not automatically resubmitted;
inspect its exact status and outputs before an explicit recovery. A recording
with an existing downstream candidate/comparison/selection/gate chain is
disabled in the browser so this interface cannot become a correction or
override path.

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

## Validation and safe smoke

Static and deterministic checks:

```bash
scripts/py -m py_compile <changed-python-files>
scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_geometry_review_approval.py \
  tests/unit/fisheye/test_geometry_review_evidence.py \
  tests/unit/fisheye/test_geometry_review_registry_notifications.py \
  tests/unit/fisheye/test_registry_shadow_publish.py \
  tests/unit/fisheye/test_registry_rescan.py \
  tests/unit/fisheye/test_arena_geometry_campaign.py -q
scripts/py -m marimo check apps/marimo/geometry_review.py
git diff --check
```

The first production-data smoke remains read-only: use a copied registry under
`/tmp`, reconcile one canonical Zarr into that copy, load the actionable queue,
and build a `dry-run` request/plan under `/tmp`. Record the exact run,
review-record digest, acquisition candidate, raw-detection binding, montage and
panel bindings, and confirm that the canonical root and `zarr.json`
modification times did not change. Never use submit/apply mode for this smoke.

Canonical submission is permitted only after that exact commit's required CI
is green. After the final registry-reconciliation job succeeds, the recording
must leave the default actionable queue and appear as
`gate_and_refinement_consumed` only under `--include-inactive true`.
