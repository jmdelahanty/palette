# Brief: registry reconcile orchestrator + subject_mask data profile

**From:** commander session, 2026-07-05
**Status: READY.**
**Read first (these are the specification; this brief only scopes the slice):**
`docs/archive/registry_reconcile_collapse_audit_2026-06-18.md` (the collapse map,
boundaries, and "smallest first step" you are executing),
`docs/diagnostics/subject_mask_profile_design_2026-06-18.md` (the profile-path design),
`docs/archive/provenance_enforcement_roadmap.md` Slice 4, `docs/registry_schema_reference.md`
(schema conventions). Where a design doc and this brief disagree, the design doc wins —
report the disagreement.

## Maintainer answers to the profile design's open questions

1. **Yes, build the distribution profile** — detection/keypoint precedent holds;
   subject_mask is the strategic stage and inherits the eye-mask stack's role.
2. **Wide columns for the common labels + `profile_json` for the rest** (the design
   doc's own recommendation). No long-form companion table in this slice.
3. **Build trigger: mirror the detection profile's trigger convention exactly** —
   find where/when `build_detection_profile_summary` actually runs today and match it.
   Report what that convention turned out to be; if detection's own trigger is ad-hoc
   or surprising, that's a finding, not something to replicate blindly — report and
   default to on-demand + reconcile pickup.

## Scope — two checkpoints

### Checkpoint 1: the orchestrator (assembly, not invention)

1. **`Registry.reconcile_dataset_from_root(root, zarr_path, *, include_step_status=...)`**
   in `registry/db.py`, per the audit's proposal: loop the EXISTING extractors +
   `replace_*` + step-status refresh for one dataset in a single idempotent pass.
   Compose existing primitives; add no new write paths (the audit's idempotency note
   is a constraint, not an observation).
2. **Re-express the detection and keypoint profile syncs as extractors** behind the
   orchestrator (`_extract_detection_data_profile_rows` / `_extract_keypoint_...`,
   matching the uniform extractor contract). Then DELETE
   `utils/sync_detection_profile_registry.py` and `utils/sync_keypoint_profile_registry.py`
   (solo-use repo: no deprecation wrappers) and update any docs/operator-guide
   references to them in the same pass (standing rule: no doc landmines).
   **Do NOT touch `sync_eye_mask_profile_registry.py`** — it is a subtraction target on
   the eye-mask deprecation path, not a fold-in candidate (audit's Category B
   correction). Leave it and all eye-mask profile code alone.
3. **CLI exposure** following the existing maintenance-CLI pattern (however
   `register_from_root` is invoked today — extend that, don't invent a new entry
   point). `register_from_root` itself stays; the orchestrator builds on it.
4. **STOP and report** (commander verifies + merges before checkpoint 2).

### Checkpoint 2: the subject_mask profile path

Per `subject_mask_profile_design_2026-06-18.md` §"Proposed shape" exactly:
5. Builder `build_subject_mask_profile_summary(...)` (schema
   `subject_mask_dataset_profile` v1; sections dataset/source/coverage/components/
   composition; reuse the shared percentile/stats helpers).
6. Zarr convention: `analysis/subject_mask_profile_runs/<run>/profile_summary` +
   `latest` pointer, identical to detection/keypoint.
7. Table `subject_mask_data_profile` keyed `(dataset_id, profile_run)` with the design
   doc's columns (wide common-label columns + `profile_json`), following the existing
   schema-migration pattern in `db.py`, plus whatever `*_latest` /
   `recording_*_latest` view conventions the sibling profile tables have
   (check `registry_schema_reference.md` and match).
8. Extractor `_extract_subject_mask_data_profile_rows(...)` + `replace_subject_mask_data_profile(...)`
   registered under the orchestrator — explicitly NOT a standalone sync script.
9. Update `docs/registry_schema_reference.md` for the new table (match how it documents
   the sibling profile tables; if the doc is generated, run the generator).

## Explicitly OUT of scope

- `reconcile_model_from_manifests()` / model-artifact reconcile (audit Category C).
- Folding `dish_mask_registry_sync` or `backfill_pose_onnx_*` (later, same interface).
- The zarr-attr backfills and read-only audit scripts (audit says keep separate —
  they repair disk truth / report, respectively).
- Deleting the eye-mask profile stack (deprecation slice, not this one).
- Any `maintenance.py` `_backfill_*` deletion beyond what the orchestrator directly
  subsumes — if ambiguous, keep and report rather than delete.
- Multi-dataset sweep orchestration (`reconcile all`) — one dataset per call is the
  contract here; sweeps stay in whatever drives `scan_paths` today.

## Constraints

- Registry writes stay inside the existing `replace_*`/upsert primitives — DELETE-then-
  INSERT idempotency and the payload-signature short-circuits must survive. New table
  follows the same pattern.
- SQLite discipline: busy_timeout, no WAL (multi-host NFS — deliberate; do not
  "improve" this).
- The orchestrator must be a strict superset of `register_from_root`'s effect on one
  dataset: running reconcile after register must be a no-op (assert this in a test).
- Zero behavior change to the extractors you re-express — the detection/keypoint
  profile rows produced via reconcile must be identical to what the sync scripts
  produced (test: run old extraction logic vs new on the same fixture, compare rows,
  BEFORE deleting the scripts).

## Validation bar (each checkpoint)

- Focused tests incl.: reconcile idempotency (run twice → identical registry state),
  reconcile-after-register no-op, sync-vs-extractor row equality (checkpoint 1),
  profile builder → zarr → extractor → table round-trip on a synthetic fixture
  (checkpoint 2).
- Full non-GPU suite: `PYTHONPATH=src ~/miniconda3/envs/palette-py311/bin/python -m
  pytest tests -m "not gpu" -q -n 16`. Baseline 3364 passed / 2 skipped as of
  edca55c — recount, it grows.
- `git diff --check` + `py_compile` clean.
- Grep proof at checkpoint 1: no remaining references to the two deleted sync scripts
  anywhere in src/, tests/, docs/, or the registry browser/TUI tooling.

## Reporting

Branch `agent/registry-reconcile` from current `sun`. One commit per concern.
Checkpoint 1 report: orchestrator shape, which inline-refresh/backfill functions it
subsumes (and which you left, with reasons), row-equality proof, deleted-script grep
proof, detection-profile trigger finding. Checkpoint 2 report: schema DDL, builder
trigger wiring, round-trip test results, schema-reference doc update.
Co-author trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
