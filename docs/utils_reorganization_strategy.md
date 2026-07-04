# utils/ reorganization strategy

<!-- contract-meta
status: proposed
created: 2026-07-04
owner: jeremy
audience: maintainer (strategy; execution phased and mostly deferred)
related: docs/interface_and_execution_strategy.md,
         docs/palette_cli_narrow_waist_design.md,
         docs/diagnostics/codebase_review_2026-07-01.md
-->

## Purpose & method

Four read-only opus agents analyzed `src/fisheye/utils/` (274 files / ~149k lines, 257
with `__main__`) along four axes: import graph/layering, functional inventory, target
structure + forcing function, and duplication/dead-code. This is the synthesis and the
phased plan. **Execution is mostly deferred** (see phasing); the drift fixes and helper
relocations are the exceptions that can proceed sooner.

## The reframe: concentrated, not total

`utils/` is the application layer wearing a library's name — **257 of 274 files carry
`__main__`**. On paper the tangle is total: the package sits in a single 19-node
strongly-connected component, in a 2-cycle with 13 subpackages. But the problems are
**concentrated into three separable layers**, and the load-bearing part is small:

1. **Structural tangle** = ~8 misfiled library modules (not a design knot).
2. **Drift** = ~120 copy-pasted helpers, 3 of them *live correctness hazards*.
3. **Sediment** = ~50 backfills + ~15 migrations/one-offs (accretion, mostly spent).

Fixing it is not a heroic reorg. It is: move ~8 files down, finish half-done helper
migrations onto homes that already exist, delete the confirmed-dead, and **forbid the
name `fisheye.utils`** so it cannot re-accrete.

---

## Layer 1 — the structural tangle is ~8 files

The import graph *looks* fully connected, but ~120 of the 185 "into-utils" edges collapse
onto a handful of genuinely misfiled library modules (no `__main__`, no upward imports of
their own):

| Module | upward edges | destination |
|---|---|---|
| `utils/system.py` | **75** (imported even by `shared/zarr/schema.py`) | `shared/` |
| `utils/zarr_io.py` | 23 | `shared/` |
| `utils/metadata.py` | 9 | `shared/` or `io/` (domain-flavored — decide per-file) |
| `utils/zarr_metadata.py` | 2 | `shared/` |
| `calibration`, `encoder_tags`, `recording_preflight`, `import_video_metadata`, `zarr_recording_context` | 1–3 each | `shared/` |
| `resolve_detect_model` / `resolve_subject_mask_model` resolvers | — | `registry/model_resolution.py` (promote `_load_candidates`/`_load_target_profile`/`_resolve_recording_id` to public) |

The worst single edge is `shared/zarr/schema.py:21` importing six provenance helpers from
`utils/system.py` — the bottom layer reaching to the top. **Moving these ~8 files down,
plus promoting ~10 borrowed `_private` symbols to public APIs in their owning layers,
severs ~120 of the upward edges and is the entire structural fix.** The remaining
utils→app and app↔utils laterals are cosmetic cycles (both endpoints land in `apps/`),
non-blocking for a `shared < apps` cut.

---

## Layer 2 — the drift is a correctness problem, not style

~120 copy-pasted private helpers, and **three have drifted into live hazards** (this is
what elevates helper-consolidation from tidiness to a correctness fix, same class as the
silent-wrong-data work):

- **`_iter_zarr` — 64 files, 6 behavior families that discover different recording
  sets.** The families differ in glob pattern, dedup, and file-vs-dir matching, so
  "find the recordings" returns a *different set* depending on which tool you run — a
  tool on a narrower family can silently omit recordings from a backfill/audit. The
  shared canonical (`shared/zarr_discovery.py::iter_filesystem_zarrs`) is yet another
  slightly-different behavior. **Real data-coverage hazard.**
- **`_utc_now` — 35 copies emit 4 timestamp formats** (isoformat, two strftime variants,
  and one **date-only** in `run_subject_mask_batch_pipeline.py`). Any consumer that
  parses/compares these provenance timestamps disagrees across tools.
- **`_write_json` — 17 of 27 writer copies are non-atomic** (no temp + `os.replace`):
  partial-write/clobber risk on crash for provenance/report JSON.

Plus ~110 more style-level copies (`_resolve_roots`, `_normalize_text`, coercions, …).

**Key finding: the homes mostly already exist and are half-adopted, not missing.**
`shared/zarr_discovery.py::iter_filesystem_zarrs`, `shared/batch_logging.py::utc_now`
(~33 files import it, 35 still define a local copy — half-migrated),
`shared/environment.py::resolve_recording_roots`, `shared/type_conversions.py`. Only two
need new homes: `infer_zarr_use()` (reconcile 6 variants; canonical precedence is
`zarr_purpose → zarr_use → filename-suffix → default` — `zarr_purpose` is the
canonical store attr current producers write, `zarr_use` is the registry/vocabulary
alias checked as a fallback) and `write_json_atomic()` (standardize on atomic).
Finishing these migrations removes ~120 copies and kills the 3
drifts. **Do not blind-sed** — `_iter_zarr` needs a generalized signature
(glob-pattern/dedupe/is_file) decided per call-site so Family B callers don't silently
change which recordings they see.

---

## Layer 3 — the sediment (backfills, migrations, one-offs)

~50 `backfill_*` + ~15 `migrate_*`/`fix_*`/one-offs are the accretion fingerprint: one
throwaway migration committed to the permanent surface per schema change, never removed.

- **Eye-mask deletes are already done** (severance executed) — the prior review's
  ~4,500-LOC eye-mask delete list is gone; do not re-recommend it.
- **High-confidence deletes (7, read-and-confirmed, zero refs, oldest in tree):** the
  raw-H5 debug cluster — `read_h5_data`, `check_h5_tracking_data`,
  `check_h5_subject_metadata`, `fix_stimulus_mode_mappings`, `backfill_h5_metadata`,
  `patch_legacy_h5`, `inspect_zarr_events`.
- **Verify-then-delete migrations:** `rename_recording_zarrs_to_training`,
  `repair_keypoint_offset_corruption`, `migrate_legacy_detect_labels`, the
  subject-mask-bridge backfills — safe to remove **only once the store is confirmed
  migrated**.

**Guardrail (critical): orphan-in-code ≠ dead.** ~44 files have zero src/test import refs
but are legitimate human-invoked `python -m` CLIs (e.g. `validate_keypoint_training_zarr`
has 8 doc refs, `registry_tui` 7). "No fisheye import" (66 files) is likewise a weak dead
signal — many are active CLIs using only stdlib+zarr+numpy. **Gate deletion on docs +
operator knowledge, not the import graph.** The directory has zero `tmp_*`/`debug_*`
scratch scripts — the dead surface is small (~1–1.5k LOC of historical tooling); the real
prize is Layer 2 duplication.

### Near-duplicate file pairs worth merging (measured)
`validate_detect_/keypoint_training_zarr` (8 diff lines → trivial `--stage` merge);
`review_subject_body_/swim_bladder_masks_batch` (~830 LOC, biggest win → `ReviewBatchDriver`);
`set_crop_/keypoint_review_status`; `backfill_detection_/keypoint_profiles`;
`finalize_refinement_/keypoint_refinement_artifacts`. Also
`_register_merged_dataset_in_registry` still copy-pasted 3× across the training-zarr
exporters (provenance code — drift is correctness-sensitive) → hoist to
`shared/training_export.py`.

---

## Target structure

Five layers, imports flow strictly downward. **The name `utils` is retired** — it is the
category error that licensed the accretion.

```
fisheye/
├── apps/        L4  operational entry points — the ONLY place __main__ lives
│   ├── runners/     run_* the waist shims to (internal impl of verbs)
│   ├── ingest/      import_*/export_*/prepare_* pipeline-data builders
│   ├── verify/      audit_*/check_*/validate_* (graduate a few to `palette verify`)
│   ├── experiments/ goodcopbadcop_*, orange_style_* — study-specific
│   ├── tools/       inspect_*/list_*/report_* diagnostics
│   └── migrations/  backfill_*/migrate_* with mandatory RETIRES_AFTER markers
├── cli/         L3  the waist — palette verbs, envelope, plan oracle, Recording accessor
├── <domain>/    L2  detection, pose, segmentation, refinement, analysis, tracking,
│                    registry, io, preprocessing, inference, training
├── shared/      L1  pure library helpers — no __main__
└── (storage specs) L0  zarr layout, RLE store, registry schema
```

The load-bearing predicate: **`apps/` = has a `__main__`, is an operation;
`shared`+domain = no `__main__`, is a capability.** That single distinction makes the
split mechanical, not judgment-based. (Note: reconcile with the existing top-level
`apps/` used for marimo/manim — recommend `fisheye/apps/` as the package.)

---

## The forcing function (or it re-accretes)

The review's lesson: utils grew because no rule said an operation couldn't be filed as a
utility. Four CI-enforced gates make that structurally impossible:

1. **Forbid the name (keystone).** import-linter `forbidden` contract:
   `forbidden_modules = fisheye.utils`. Once helpers and runners have moved, nothing may
   import `fisheye.utils` — the dumping ground has no address. Highest-leverage rule.
2. **Layered contract.** `shared < registry < stages < cli < apps`; domain packages
   cannot import `cli` or `apps`; `shared` imports nothing above it. Stops a helper from
   depending upward on a runner (the inversion that tangled utils).
3. **The `__main__` predicate** (custom ~20-line CI check import-linter can't do): a
   module with `__main__` may only live under `apps/` (or be `cli/palette.py`); a module
   under `shared/` may not contain `__main__`.
4. **Ceilings & expiry:** per-file line ceiling (flag the next 4,000-line monster);
   file-count caps on `apps/tools` and `apps/experiments` (the ad-hoc buckets accretion
   targets next); `apps/migrations/*` require a `RETIRES_AFTER` marker with a CI job that
   flags any past its date — backfills born with an expiry can't become permanent surface.

---

## Phased execution plan

Big-bang is unsafe (90 files import cross-utils; `cli/palette.py` edited daily; the
executor doesn't exist). Ordered so each step is independently revertible and never blocks
waist work. **Phases 1–3 can proceed now; runner moves (Phase 5) are gated on Slice D.**

- **Phase 0 — frame, no moves.** Create `fisheye/apps/`; land import-linter in CI
  **report-only** with the layered contract. Visibility before migration.
- **Phase 1 — fix the drifts / finish helper migration (NOW; correctness win).**
  Migrate `_iter_zarr`→`iter_filesystem_zarrs` (generalize signature first, per-call-site,
  reconciling the 6 discovery behavior families deliberately), `_utc_now`→`utc_now`,
  `_resolve_roots`/coercions onto existing homes; add `infer_zarr_use()` and
  `write_json_atomic()` homes. Removes ~120 copies + kills the 3 live drifts. Touches
  `shared/` + utils bodies, **not** `cli/palette.py` — low conflict with the API arc.
- **Phase 2 — move the ~8 misfiled libraries down (NOW-ish; structural win).**
  `system`, `zarr_io`, `metadata`, `zarr_metadata` (+ the 1–3-edge modules) → `shared/`;
  the model-resolvers → `registry/`; promote the ~10 borrowed privates to public first.
  Use one-release re-export shims (`utils/system.py` re-exports from `shared/`) then
  delete. This severs ~120 upward edges and the worst `shared→utils` edge — the
  prerequisite for forbidding `fisheye.utils`.
- **Phase 3 — retire confirmed-dead (NOW).** Delete the 7 H5-debug files; move surviving
  migrations to `apps/migrations/` with expiry markers; verify-then-delete the spent ones.
- **Phase 4 — trivial merges.** `validate_*` `--stage` merge; the `review_*_batch`
  driver; hoist `_register_merged_dataset_in_registry` to `shared/training_export.py`.
- **Phase 5 — move runners (GATED on Slice D `verb(request)->Envelope`).** Do **not**
  relocate `run_detect_with_registry_model` et al. while the waist imports them by path
  and that import is being rewritten. After the verb decoupling: move each runner to
  `apps/runners/` in the same PR that repoints its verb import. One runner per PR.
- **Phase 6 — move standalone tools** to `apps/{verify,tools,experiments}`; graduate 2–3
  contract-checkers into `palette verify`.
- **Phase 7 — flip the gate.** Add `forbidden: fisheye.utils`, flip import-linter to
  blocking, add the `__main__`-predicate check + ceilings; delete the empty `utils/`.
- **Phase 8 — executor convergence.** As the plan-driven executor is built, the
  `apps/runners/` batch-drivers + `run_recording_analysis_pipeline` retire as standalone
  entry points. Rides on the executor work, not this reorg.

**Hard dependencies:** Phase 1+2 (helpers/libraries out) gate Phase 7 (forbid utils);
Slice D (verb decoupling) gates Phase 5 (runner moves). Everything else parallelizes.

---

## What NOT to do

- **No blind sed** on `_iter_zarr` — the discovery-set drift means a mechanical replace
  silently changes which recordings a tool sees. Reconcile per call-site.
- **Don't delete on the import graph alone** — orphan-in-code ≠ dead; ~44 code-orphans
  are live human CLIs. Gate on docs + operator knowledge + confirmed store-migration.
- **Don't move runners before Slice D** — path-imports being rewritten by the waist.
- **Don't couple the reorg to the executor** — Phases 0–7 are valuable and safe before the
  executor exists; Phase 8 just cleans up afterward.
- **Don't do this as one big-bang PR** — 90 cross-utils imports; go phase by phase, each
  revertible.
