# Registry design assessment

**Date:** 2026-06-18
**Author:** Claude (read-only review), at the maintainer's request.
**Scope caveat:** This assessment is based on a partial view — the model-metadata
tables, the status ledger, the stage catalog, the query/TUI/web surfaces, and the
metadata-gap investigation (see `model_input_shapes_metadata_gaps_2026-06-17.md`).
It has **not** deeply read the extractors, the zarr scan logic, or the training
pipeline. Calibrate accordingly; some apparent gaps may be solved elsewhere.

## Summary

The registry's static modeling is strong and uses several established patterns
(correctly, if independently reinvented). The growth area is the *dynamics*: how
captured state stays convergent with on-disk reality over time. The clearest
symptom is a proliferation of one-off sync/backfill scripts where a single
idempotent reconciliation path would do.

## What is well-designed (with the names for the patterns)

- **Metadata catalog in SQLite.** Single source of truth, rebuildable, inspectable.
  The right call for a solo project — not over-engineered into Postgres/cloud.
- **Append-only event log + latest projection.** `recording_step_status_history`
  (events) + `recording_step_status` (materialized latest) gives auditability and
  idempotent upserts. This is solid event-sourcing-lite that many systems get wrong.
- **Explicit workflow DAG.** `stage_catalog` with dependency + invalidation maps
  models the pipeline graph *as data*, which is what enables "what is stale
  downstream" reasoning. Sophisticated and correct.
- **Disk-as-truth, DB-as-index.** Sidecar manifests on disk with registry rows
  mirroring them — the registry is a derived index that could be rebuilt from disk.
- **Disciplined schema evolution** via numbered migrations.

## Weaknesses (ranked; each notes the underlying principle)

### 1. Capture is scattered and not idempotent-by-design  *(highest leverage)*
Symptom: a drawer of artifact-specific repair scripts —
`backfill_pose_onnx_registry_metadata.py`, `sync_detection_profile_registry.py`,
`sync_keypoint_profile_registry.py`, `sync_eye_mask_profile_registry.py`,
`dish_mask_registry_sync.py`, `audit_registry_dataset_paths.py`,
`registry_rescan.py`, `scripts/backfill_protocol_json.py`, and more. The v004-pose
gap (shape present in the on-disk manifest but never recorded in the table) is this
problem in miniature: capture ran once, imperfectly, and now needs a bespoke fix.

**Principle — reconciliation/convergence.** A mature catalog has *one* idempotent
"re-derive registry state from disk truth" operation that converges, not N repair
scripts. Build systems, search indexers, and Kubernetes controllers all work this
way: desired state is a pure function of source, repeatedly reconciled. The pieces
exist here (`scan_zarr`, migration-time backfills) but fragmented per artifact.

### 2. Value-level provenance is solved once and should be generalized
`input_shape_status` (`explicit` / `inferred_from_imgsz` / `export_backfill` /
`unknown`) records *why* a value is what it is. That's excellent — a bare `NULL`
conflates "not applicable," "not captured yet," and "genuinely absent." The
best-practice move is to make "value + how-derived + confidence" a general shape so
every derived field can report its origin, instead of solving it field-by-field.

### 3. Schema sprawl without consolidation
86 tables/views, 55 migrations, multiple near-duplicate `*_wide_view`s,
`recording_overview` defined twice. Normal organic growth, but a maintenance tax for
a solo dev. **Principle — treat the schema like code:** migrations are append-only
history, but views can be consolidated freely. A pruning pass lowers cognitive load
more than most new features.

### 4. Leaky unification in `model_input_shapes`
A view papering over three tables with divergent schemas (no `dtype` in
onnx/tensorrt, no `precision` in training/onnx), forcing hardcoded `NULL`s. The
abstraction implies a uniformity that does not exist. Either give the underlying
tables a real shared core or stop presenting them as one thing.

### 5. Three readers, no shared read API
The TUI, the status page, and `registry/query.py` each issue their own SQL against
the DB. Drift risk; a shared read model would prevent three copies of query logic.

## Highest-leverage move

Build the single idempotent `reconcile`/`reindex` command (per-recording and
whole-registry) that re-derives every registry field from disk truth and converges.
It directly resolves #1, exposes #4, and would have prevented the v004 gap.
Everything else is real but secondary.

## Meta-lesson

The hard part of a catalog is not capturing state once — it is keeping it
**convergent with reality over time**. The static modeling here (ledger, DAG,
lineage) is genuinely good. The investment that pays off next is the dynamics: one
reconciliation path instead of a drawer of repair scripts.

## Next step (in progress)

A read-only audit of the sync/backfill/scan scripts to determine how many could
collapse into one reconciliation path — what each reads, what it writes, whether it
is idempotent, and where the boundaries genuinely differ. Findings to be recorded
separately.
