# Five-Lens Architecture Review — 2026-09-01

> Disposition added 2026-09-06: historical evidence, not an executable checklist.
> The [second opinion](review_wave_second_opinion_2026-09-04.md) corrects the universal gate/resolver diagnosis and rejects
> blanket selector backfills and weaker universal receipts. A default-off
> registry gate alone does not close admission/publication. Recheck surviving
> findings against exact current code; see the [September 6 reconciliation](review_docs_rules_landing_handoff_2026-09-05.md#september-6-audit-reconciliation).

<!-- contract-meta
version: 1
status: active
last_verified: 2026-09-01
implementation: specified-only
-->

**Date:** 2026-09-01
**Snapshot:** `dcdcc081` on branch `agent/palette/refined-assignment-rebinding-gaze-20260831`
**Method:** five parallel read-only review agents (orchestration, provenance/authority, storage/performance, code structure/CI, governance) followed by a synthesis pass. No code was executed beyond `grep`/`sed`/`wc`; no tests were run. Findings marked **[VERIFIED]** were re-read by the synthesizer at the cited line; everything else is from agent code reading and **must be re-verified against current code before implementation**.
**Question asked:** does this codebase deliver the stated goal of an auditable, performant, distributed, stage-based data processing pipeline?

**Queue disposition:** this document is audit evidence and a checklist source. It does not open a fourth parallel fix queue. Items that overlap authority, admission, resolver, or enforcement work are tracked only in [`authority_consolidation_work_queue_2026-08-25.md`](authority_consolidation_work_queue_2026-08-25.md); items that overlap deletion are tracked in [`subtraction_queue_2026-08-21.md`](subtraction_queue_2026-08-21.md). New items introduced here (runtime commit assertion, digest-bound authority history, store-opener lint, findings ledger, ratchet widening, error-budget measurement) should be adopted into one of those queues, not tracked here.

---

## 1. Verdict

The architecture of an auditable pipeline exists and is being run as a convention-based one.

The primitives are genuinely strong: parent-scoped completion epochs with fail-closed `MissingCompletionEpochError`, lease/generation selector activation, atomic publish-by-rename on NFS, digest-bound bundle manifests, commit-pinned detached worktrees, an exhaustive import linter, and a 9,201-test suite that uses real zarr rather than mocks. Most scientific pipelines never acquire one of these.

Nearly all of them are optional at the point of use. Every lens independently found the same shape: a well-designed spine plus N bypasses that carry most of the traffic.

| Concern | The spine | The bypasses |
|---|---|---|
| Sequencing | typed `cluster/lsf` workflow framework | 69 hand-written `scripts/submit_*_bsub.sh`; legacy `core/pipeline.py` still wired to `python -m fisheye`; three other runners |
| Which run is current | `zarr_run_completion` resolvers (eligibility + matching-pair rule) | 9 independent resolvers, one falling back to lexical sort with no eligibility check; 229 files read `attrs["latest"]` directly |
| Store access | `open_zarr_root` with explicit consolidated policy | 839 direct `zarr.open_group` calls in 352 files, 219 with no consolidated flag |
| Code identity | `run_provenance.git_sha` | 4 incompatible attr grammars; no job-start assertion that the checkout matches the plan's recorded commit |
| Contracts | 122 `docs/*contract*.md` | roughly 20 have a named validator; the freshness check sees 49 of 122 |
| Layering | import-linter, `exhaustive = true` | two declared layers; 850k LOC above `shared` has no dependency direction; `utils` at 244k LOC is the real application |
| Enforcement gate | `emit_stage_completion` runs array-spec, provenance, eligibility validation | wrapped in `except Exception` → warning → `return False`, after the run is already `latest` |

**The one finding all five lenses converged on** is the last row. `src/fisheye/registry/stage_complete.py:488-496` catches every exception, prints a yellow warning, and returns `False` **[VERIFIED]**. The zarr run has already been promoted to `latest` by `mark_run_complete` before this executes. `docs/zarr_run_completion_contract.md:91` describes the same helper as fail-closed **[VERIFIED]**. This was flagged as F1 in the 2026-08-21 enforcement review and is still present. It is the load-bearing defect; everything else in this document is downstream of it.

**Auditability today, concretely.** For `track_kinematics_runs` and `subject_shape_runs`, which bind `zarr_payload_receipt`, an auditor can mechanically verify inputs, code, and outputs. For most other families the auditor gets a run name, a model *path* rather than a hash, one of four git attr grammars, and a "latest" resolved by whichever of nine resolvers the reader chose. That is reconstructible by a careful human. It is not mechanically verifiable.

**Calibration.** The ratio of enforcement machinery to enforced behavior is unusual in the right direction. Closing the gap is mostly connecting things that already exist, plus deletion. That is a far better position than the reverse.

---

## 2. Cross-cutting priorities (in order)

1. **Flip the gate.** Narrow the catch at `stage_complete.py:488` to registry I/O errors; let contract validation raise or write an `error` row; run it before promotion to `latest`. 1-2 days plus a canary. Until this lands, do not author another contract document.
2. **One resolver, one store opener, one code-identity block.** Delete the eight other resolvers; ban bare `zarr.open_group` outside `fisheye.shared` via the import linter that already exists; write one provenance helper every writer calls; persist `selection_branch` in `recording_step_status.details_json`. ~3 weeks, mostly deletion and codemod.
3. **Runtime commit assertion and one scheduled reconcile.** Half a day for the assertion in `lsf/runtime.py`; ~2 days for a nightly job diffing registry rows against resolver output and publishing counts to the status page. This is the first thing that makes the error budget measurable.
4. **Findings ledger and widened ratchet.** One index for `docs/diagnostics/` with `open | landed | waived | superseded`; CI refuses unlisted docs. Extend the file-size ratchet from 4 files to the 153 over 1,500 lines. ~1 day each.

Everything below is the per-component evidence and checklists behind those four.

---

## 3. Orchestration

### 3.1 How it works today

There is no single orchestrator. Sequencing lives in five parallel mechanisms:

| Mechanism | Location | Notes |
|---|---|---|
| Hand-written bsub submitters | 69 × `scripts/submit_*_bsub.sh`, ~22k lines bash, largest `submit_chaser_analytics_bsub.sh` at 1,448 lines | build `bsub` argv by hand; ~10 of 69 use `-w done()` chaining |
| Typed LSF framework | `src/fisheye/cluster/lsf/` (`models.py` → `submission.py` → `runtime.py` → `task_group.py`) | ~40 planner modules; immutable `palette.lsf_workflow.v1` JSON plan; topological submit |
| Declarative analysis DAG | `docs/analysis_workflow_dag.md`, `execute_analysis_workflow` | submitted as one monolithic job by `scripts/submit_analysis_workflow_bsub.sh:464` |
| Sequential subprocess runner | `fisheye.utils.run_recording_analysis_pipeline` | one recording at a time |
| Legacy orchestrator | `src/fisheye/core/pipeline.py` (1,511 lines) | still the `python -m fisheye` entrypoint (`src/fisheye/__main__.py:8`); writes legacy `refined_runs`/`keypoints_refined_runs` groups (`pipeline.py:41-44`) |

Completion is signaled in-band: `mark_run_complete` stamps `palette_run_completion_status=complete`, then flips the parent's `latest_complete` and `latest` (`src/fisheye/shared/zarr_run_completion.py:258-305`). Registry rows are written afterward by `emit_stage_completion` (`registry/stage_complete.py`), the only place array-spec/provenance/eligibility gates execute. Deployments are commit-pinned by `scripts/deploy_palette_cluster_worktree.sh` (detached locked worktree `<branch>-<sha8>`, `:321-333`); `scripts/py` prepends the `src` beside itself.

### 3.2 Strengths

- `cluster/lsf` is well designed: incrementally persisted submission record (`lsf/submission.py:99-166`), per-job status JSON that fails the job when expected outputs are missing even on rc=0 (`lsf/runtime.py:283-286`), bundles mark tail tasks `skipped_after_failure` (`task_group.py:192-213`), backend refuses anything but `bsub` over SSH (`backend.py:217`).
- Plans record `palette_commit` (`clipped_inference.py:2041`, `lsf/models.py:256`); the deploy script checks clean status and verifies from the submit host.
- Completion markers are two-phase and fail-closed post-cutover (`zarr_run_completion.py:90-95`), with `latest_pending` for in-flight runs.
- Resume validates rather than trusts (`_validate_existing_detection_for_resume`, `clipped_inference.py:1748`).

### 3.3 Risks

| # | Finding | Evidence | Why it matters |
|---|---|---|---|
| O1 | Gates are advisory | `stage_complete.py:488-497` **[VERIFIED]**; 20 call sites, return value mostly discarded | A failing contract changes nothing scientifically visible; the run is already `latest` |
| O2 | Five sequencing mechanisms, no shared stage definition | `stage_catalog.py` names stages but binds no runnable/inputs/outputs; 69 bash submitters each re-implement resources, logs, force/skip (`submit_subject_mask_batches_bsub.sh:190-191`) | Ordering depends on humans running scripts in sequence |
| O3 | No retry or failure policy | zero `requeue`/`-r`/retry in `scripts/*.sh` or `cluster/lsf`; `LsfJob` has no retry field (`models.py:220-270`); recovery is bespoke per stage (`clipped_inference_*_recovery.py` ×3) | Every stage owns its own recovery code |
| O4 | Completion-marker split-brain | 453 `use_consolidated=False` sites show the workaround is manual; two-attr activation window `zarr_run_completion.py:298-300` is non-atomic on NFS | Readers that forget see stale attrs; fail-closed epoch gate becomes fail-open (884 groups known) |
| O5 | Commit pinning ends at submission | `lsf/runtime.py` and `lsf/models.py` contain no commit verification at job start **[VERIFIED: grep empty]** | A moved or re-deployed worktree silently runs different code under pinned provenance |
| O6 | Nothing is scheduled | no `schedule:` in `.github/workflows`, no cron | Reconcilers, sweeps, epoch backfill are human-remembered CLIs |
| O7 | Legacy orchestrator is the package entrypoint | `__main__.py:8` → `core/pipeline.py` | A new user's first command uses the deprecated path |

### 3.4 Checklist

- [ ] **O-1** Narrow `stage_complete.py:488` catch to registry I/O; contract raises propagate or write an `error` status row. Add a test that a failing array-spec validation produces a non-`ok` row.
- [ ] **O-2** Move `emit_stage_completion` (or its validation core) before `mark_run_complete` so a run cannot be `latest` without passing gates. Confirm behavior for all 20 callers.
- [ ] **O-3** Add `assert_executing_commit(plan)` to `lsf/runtime.run_with_status`: compare `git rev-parse HEAD` of the executing checkout to `plan.palette_commit`; refuse on mismatch; record the check in the status JSON.
- [ ] **O-4** Add `retry: {max_attempts, on_exit_codes}` to `LsfJob`; honor via `bsub -r` or resubmit-from-status. Document which stages are safe to retry (idempotent writers only).
- [ ] **O-5** Declare a stage registry (YAML or Python table): `stage_id → module, input run parents, output run parent, resources, idempotency check`. Make `stage_catalog.py` consume it.
- [ ] **O-6** Migrate the 10 highest-traffic bash submitters onto `cluster/lsf` planners (most already have Python planners); delete the shell once parity is tested.
- [ ] **O-7** Add one scheduled job (cron on the submit host or CI `schedule:`) running `check_zarr_run_completion --fail-on-unsafe` and `reconcile_sweep --dry-run`, publishing counts to the status page. Requires the dry-run write bug (2026-08-20 audit) to be closed first.
- [ ] **O-8** Point `__main__.py` at the CLI; delete `core/pipeline.py` after confirming no bsub script invokes it.
- [ ] **O-9** Make the `latest_complete`/`latest` flip a single attrs write (or a single envelope attr) so there is no observable half-activated window.

---

## 4. Provenance and authority

### 4.1 How it works today

Every stage writes `<family>_runs/<run>` groups; lineage lives in attrs (`provenance` via `stage_provenance.py`, `run_provenance` via `run_provenance.py`, family-specific `source_*` via `provenance_attrs.py`). Completion is the parent-scoped epoch contract. Selection is the `latest`/`latest_complete` matching pair, optionally overridden by `authoritative_run` + `authoritative_run_provenance`; modern bundles (subject masks, keypoint v2) use an atomically written root envelope (`subject_mask_authority`) binding member run IDs and manifest digests through `selector_activation.py` lease/generation mechanics. Content identity is digest-based where the modern path applies (`zarr_payload_receipt.py`, bundle manifests, compact array references); older families identify inputs by run name only. The SQLite registry (`registry/maintenance.py`, 9.6k lines) is a projection that re-walks archives and re-resolves "latest" with its own resolver.

### 4.2 Strengths

- Completion-epoch design is correct fail-closed engineering (`zarr_run_completion.py:84-127`).
- `set_authoritative_run` refuses ineligible/incomplete runs and rolls back the attr pair on failure (`zarr_run_completion.py:705-763`).
- `selector_activation.py` and `subject_mask_bundle_publication.py` (manifest digests, `_require_unselected`, receipt-bound live validation at `:1213`, `:1686`) are real atomic-publication primitives.
- `docs/compact_array_backed_provenance_contract.md` is the right shape: readable subrecord + typed arrays + `content_sha256` references.
- `run_provenance.git_identity` captures `git_dirty`; `recording_step_status_history` exists.
- Terminology (`canonical` / `complete` / `selector_eligible` / `authoritative` / `accepted`) is nailed down in AGENTS.md and the 2026-08-27 checklist.

### 4.3 Risks

| # | Finding | Evidence | Why it matters |
|---|---|---|---|
| P1 | Registry resolver bypasses eligibility, falls back to lexical order | `maintenance.py:4113-4149` (`_resolve_latest_group`, `sorted_fallback`) **[VERIFIED]**; `:4180-4211`; `inline_refresh.py:28-36` takes `latest` else `names[-1]` with no completion check **[VERIFIED]** | Registry can project a run as "the" run that no zarr reader would select |
| P2 | `authoritative_run` is a name pointer with no history and no digest | `authoritative_run_history`: 0 hits in `src` **[VERIFIED]**; `set_authoritative_run` stores approver/time/sha but not manifest or payload digest | "What was authoritative when this figure was made" is unanswerable after re-approval |
| P3 | Split-brain normalized in code, contrary to AGENTS.md | `zarr_helpers.resolve_zarr_run:660-730` silently falls through to the filesystem path when consolidated view lacks the `latest` child ("Preserve the stale-consolidated-metadata fallback") | Same archive, three answers to "which run is latest" |
| P4 | Four incompatible code-identity grammars | `run_provenance.git_sha` (`run_provenance.py:22`); `provenance.git.commit` + top-level `git_commit` (`stage_provenance.py:113-199`); `system_metadata.git` (`:800`); `producer.git_sha` (`recording_import_receipt.py:122`); `palette_git_sha` (`recording_geometry_recovery.py:423`). `run_lineage_fingerprint.py:220-226` probes three of them | No repo-wide answer to "what commit produced this group" |
| P5 | Model identity is a path, not a hash, on the detection candidate | `detection_model_provenance.py:24-40` writes `model_resolution_selected_model_path`, `run_id`, `set_id`; `model_sha256` only in registry (`model_resolution.py:75`) | If registry is rebuilt or a path reused, the zarr cannot prove which weights ran |
| P6 | Registry write failures are warnings | `stage_complete.py:488-496`; 77 `except Exception` in `maintenance.py`; `details_json` does not record selection branch | Projection diverges silently and nothing records that it did |
| P7 | Validation evidence evaporates | `zarr_payload_receipt` has 4 importers; provenance-enforcement bypass allowed at 6 sites with free-text reason only (`zarr_run_completion.py:184-195`); `validate_run_provenance` accepts `git_dirty=True` silently | Every consumer re-scans or trusts attrs |

**Resolver census:** 9 independent "latest/authoritative" resolvers: `resolve_latest_complete_run_name`, `resolve_authoritative_run_name`, `zarr_helpers.resolve_zarr_run` (11 callers pass `fallback_to_latest=True`), `maintenance._resolve_latest_group`, `_resolve_latest_group_nested`, `run_resolution` `INVENTORY_LATEST` (SQL `ORDER BY updated_utc`), `run_resolution` `SOURCE_MATCH` (no completion check), `inline_refresh._select_latest_profile_run`, `recording_artifact_inventory._resolved_authoritative`. Plus root envelopes, `*_review_status_latest` pointers, and 229 files reading `attrs["latest"]` directly.

### 4.4 Checklist

- [ ] **P-1** Make `zarr_run_completion.resolve_*` the only resolver implementation. Return `(name, group, selection_branch, eligible, complete)`. Delete `maintenance._resolve_latest_group`, `_resolve_latest_group_nested`, `inline_refresh._select_latest_profile_run`, and the `zarr_helpers.resolve_zarr_run` fallbacks; route `run_resolution` through it.
- [ ] **P-2** Make `sorted_fallback` a diagnostic-only flag that can never populate an `ok` row.
- [ ] **P-3** Persist `selection_branch` into `recording_step_status.details_json` for every row.
- [ ] **P-4** `set_authoritative_run` records `manifest_sha256` (or payload receipt digest) and appends the prior `(run, provenance)` to an `authoritative_run_history` list attr. Add a reader that answers "authoritative as of timestamp T".
- [ ] **P-5** Add `model_sha256` to `detection_model_provenance` attrs; make cohort release refuse candidates lacking it after a cutover date.
- [ ] **P-6** Define one `palette_code_identity` attr block (`git_sha`, `git_dirty`, `fisheye_version`, `config_hash`) written by one helper; migrate the four existing grammars behind a reader shim; add a freshness check that no new writer emits the old keys.
- [ ] **P-7** `validate_run_provenance` warns (and records) on `git_dirty=True`; production selectors refuse dirty runs after cutover.
- [ ] **P-8** Provenance-enforcement bypass sites (6) require a structured reason code and are counted by the nightly reconcile.
- [ ] **P-9** Extend `zarr_payload_receipt` binding to the next two highest-consumed families (candidates: `swim_bout_runs`, `bout_kinematics_runs`); see the 2026-08-17 receipt audit for the sibling-evidence-group design.
- [ ] **P-10** Nightly reconcile diffs registry `run_name` vs resolver output per family and publishes mismatch counts; a non-zero count is a Tier-1 error-budget event.
- [ ] **P-11** Remove the stale-consolidated fallback in `zarr_helpers.resolve_zarr_run:660-730`; a consolidated view that cannot see `latest` is a publication defect and must raise.

---

## 5. Storage and performance

### 5.1 How it works today

Each recording owns Zarr-v3 archives (`training.zarr`, `analysis.zarr`) on NFS or NVMe, containing ~30 `*_runs` parents plus an `analysis/` subtree of derived families. Runs are timestamp-named append-only groups. Analysis families publish via `atomic_run_publisher.py` (node-local compute → verified copy → atomic rename). Masks have three encodings (dense `masks_roi`, component RLE, bitpacked) with coherence tracked by stale-flag attrs (`mask_store.py:378-470`). Analysis families use `compact_tabular_v2` / `compact_dense_v2` layouts with byte-planned chunks/shards (`shared/zarr/storage_planner.py`, `storage_profiles.py`). Cross-recording analytics are immutable Parquet generations with flock + manifest compare-and-swap (`analytics_exports/publication.py:128,178-201`), consumed via Polars `scan_parquet`. Video decode fans out through `crop_image_source.py` (2.4k LOC) to cv2, decord, PyNvVideoCodec, ffmpeg-NVDEC, and kvikio/GDS.

### 5.2 Strengths

- Dask write-safety rule (`docs/dask_zarr_write_safety.md`) is correct and learned from a real bug; `zarr_sharded_copy.py` and `detection_storage.py:25-40` enforce whole-row-axis chunk ownership.
- Atomic publisher pattern is the right transactional model on NFS; analytics exports mirror it with generation dirs and a single manifest rename.
- Analytics families are consumer-facing through `*_io` resolvers; physical layout is explicitly not a contract (`docs/analytics_storage_schema_matrix.md`).
- `open_zarr_root` fails closed on v2 metadata and forces callers to name a consolidated policy.
- Compact-v2 layouts and codec/storage profiles are named, versioned, and stamped as provenance.

### 5.3 Risks

| # | Finding | Evidence | Why it matters |
|---|---|---|---|
| S1 | No single store abstraction; 61% of opens bypass it | 839 direct `zarr.open_group`/`open_consolidated` in 352 files **[VERIFIED: 352 files]** vs 537 `open_zarr_root`; 219 pass no consolidated flag (e.g. `analysis/analyze_goodcopbadcop_escape.py:38`) | Under zarr 3.1.3 the default uses consolidated metadata if present; science scripts read through the split-brain path |
| S2 | Three competing direct-open helpers | `zarr_io.open_zarr_root`; `zarr_helpers.open_zarr_group_direct:413` (no v2 guard); `registry/zarr_open.py:26`; plus ~29 module-local `open_*`/`get_*_group` wrappers | Different guarantees per caller |
| S3 | Root schema versioning vestigial; family versioning sprawled | `ZARR_SCHEMA_VERSION = "3.0.0"` at `shared/zarr/schema.py:31`, self-labelled `legacy-metadata`, 2 refs; 2,640 `"schema_version"` literals, 194 distinct `schema_id`s, ~30 `*_schema_version` keys; 10 ad-hoc `utils/migrate_*.py`; registry by contrast has `PRAGMA user_version` + 83 migrations | No zarr migration engine; `hierarchical_v1` compat lives in 11 readers forever |
| S4 | Mask cache coherence by mutable attrs, not content identity | `mask_store.py:378-470` stale flags | Any writer touching `masks_roi` without the marker leaves derived caches "fresh"; same class as H1 hardlink-as-identity |
| S5 | Run-family naming drift baked into readers | `keypoints_refined_runs` and `refined_keypoints_runs` both exist (`zarr_metadata.py:120`); `refined_runs` aliases `refined_detect_runs` (`stage_run_groups.py:14-15`) | Every new reader must know aliases |
| S6 | Dask writers align by convention only | `refine_subject_masks.py:469-537` takes user `chunk_size`, records it, never asserts against the physical chunk grid | The exact failure the doc describes is one flag away |
| S7 | NFS locking assumptions | `archive_metadata_publication_lock` (`zarr_helpers.py:69-131`) flocks a sidecar; no lease or staleness detection | Only as good as lockd across LSF hosts; registry WAL is already off for the same reason |
| S8 | Monoliths on the hot path | `analysis/track_kinematics.py` 14,520 lines; `tracking/crop.py` 5,190; 17 files > 200 KB | Chunk decisions for track kinematics live near line 10,015 |

### 5.4 Checklist

- [ ] **S-1** Fold `open_zarr_group_direct` and `registry/zarr_open.py` into `zarr_io.open_zarr_root`; delete the ~29 module-local wrappers or make them thin delegates.
- [ ] **S-2** Add an import-linter forbidden-import rule (or a ratchet like `check_zarr_open_mode`) banning bare `zarr.open_group` / `zarr.open_consolidated` outside `fisheye.shared.zarr_io`. Baseline the 839 sites; require the count to decrease.
- [ ] **S-3** Codemod the 219 flag-less opens to `open_zarr_root(..., use_consolidated=<explicit>)`. Delete rather than convert the sites in `utils/` and `diagnostics/` that the subtraction queue already marks dead.
- [ ] **S-4** Introduce a family version registry: one module mapping `schema_id → (current version, reader, migrator)`; add a `palette_layout_epoch` root attr; retire `ZARR_SCHEMA_VERSION`; time-box `hierarchical_v1` readers via that registry.
- [ ] **S-5** Replace mask stale-flag attrs with a digest of `masks_roi` chunk payloads stamped on each derived encoding; `open_mask_store` compares digests and refuses stale caches.
- [ ] **S-6** In every Dask apply path, assert worker row ranges are multiples of each written array's physical row chunk; fail before writing.
- [ ] **S-7** Collapse `keypoints_refined_runs`/`refined_keypoints_runs` and `refined_runs`/`refined_detect_runs` behind one canonical name in `stage_run_groups.py`; readers accept aliases only through that table.
- [ ] **S-8** Add a lease timestamp to `archive_metadata_publication_lock` sidecars and a staleness check; log lock waits over a threshold.
- [ ] **S-9** Split `track_kinematics.py` along its existing section boundaries (planning, compute, storage, publication) once the file-size ratchet covers it.

---

## 6. Code structure and CI

### 6.1 Numbers

| Metric | Value |
|---|---|
| LOC in `src/fisheye` | 1,084,079 (~36 subpackages) |
| LOC under `src/` outside `fisheye` | 41,103 (`src/*.py` 44 files / 18,067; `src/chaser_analysis` 48 files / 22,799) |
| Modules > 1,500 LOC | 153 |
| Test files / test functions | 1,045 / 9,201 (1,044 in `tests/unit`, 1 in `tests/integration`) |
| CI workflows / jobs | 1 (`ci.yml`) / 8 jobs, `tests` fans to 16 shards |
| `argparse.ArgumentParser(` modules | 806 (728 in `fisheye`: `utils` 401, `diagnostics` 124, `analysis` 67); 3 `[project.scripts]` |
| Largest subpackages | `utils` 243,670 (420 modules); `shared` 233,409 (189); `analysis` 127,581 |
| Helper clones | `_sha`/`_sha256` in 102 modules; `_utc_now` 73; `_read_json` 57; `_digest` 35; `_open_root` 33 |

Fifteen largest modules: `utils/audit_coordinate_contracts.py` 16,856; `analysis/track_kinematics.py` 14,520; `labeling/web.py` 12,756; `registry/maintenance.py` 9,635; `registry/migration_bodies.py` 9,120; `refinement/finalize_subject_masks.py` 8,769; `registry/db.py` 8,542; `labeling/admin_dashboard.py` 7,728; `analysis/eye_angle_analysis.py` 6,897; `utils/export_cross_recording_analytics.py` 6,470; `shared/subject_shape_coordinate_publication.py` 6,164; `shared/refined_subject_mask_coordinate_publication.py` 6,164; `shared/pixel_frame_authority.py` 5,458; `tracking/crop.py` 5,190; `utils/export_keypoint_training_zarr.py` 4,974.

### 6.2 Strengths

- Enforcement CI is real and layered: import-linter with `exhaustive = true` and `unmatched_ignore_imports_alerting = "error"` (`pyproject.toml:107-150`); file-size ratchet; zarr-open-mode ratchet; observed-metadata-literal and contract-freshness checks; a `quality` job that builds a non-editable wheel, imports from `/tmp`, and smoke-runs the `palette` entrypoint.
- Ratchet ignores are annotated with `# RATCHET:` reasons (`pyproject.toml:124-137`).
- Tests exercise real zarr: 418 test files import `zarr`, 741 use `tmp_path`, 22 use `MemoryStore`, zero mock zarr.
- Sharded, fixture-cached CI with per-shard ownership of expensive fixtures.

### 6.3 Risks

| # | Finding | Evidence | Why it matters |
|---|---|---|---|
| C1 | Layering contract is two layers | `pyproject.toml:118-121` declares one flat mega-layer of 36 packages above `shared`; 57 files do `utils → analysis` | Checker proves only "shared is at the bottom" |
| C2 | `utils/` is the application | 420 modules, 244k LOC, 401 argparse mains; larger than `analysis`+`registry`+`tracking`+`detection`+`segmentation` | Ownership undiscoverable; name inverts meaning |
| C3 | File-size ratchet guards 4 files | `scripts/file_size_ratchet_baseline.json` lists `cli/palette.py`, `labeling/web.py`, `registry/db.py`, `registry/maintenance.py`; +200 allowance (`check_file_size_ratchet.py:12`) | 149 other modules > 1,500 LOC unratcheted, including the top two |
| C4 | 806 parsers, 3 installed scripts | `runnables.yaml` is 30 lines; `def main` 728×; `_build_parser`/variants 366× | Every ops action has bespoke argument grammar; the 35 apply/dry-run bugs are this |
| C5 | Micro-duplicated primitives | `_sha256` ×102, `_utc_now` ×73, `_read_json` ×57 (e.g. `tracking/run_manifest.py:56`, `analysis/chaser_component_publication.py:233`) | A digest canonicalization change is a 100-site edit with no agreement test |
| C6 | `chaser_analysis` is an unlayered second package | `src/chaser_analysis` imported by 9 `fisheye` modules (`visualization/chaser_analysis_figures.py`, `analysis/chaser_profiles.py`, `cluster/provider_chaser_position_suite.py`, …); outside import-linter root | Not dead; invisible to the boundary check |
| C7 | Dead trees and manifest sprawl | `src/*.py` 44 legacy scripts incl. `src/test_fisheye.py`; 16 `*v2*` modules with no v1 retirement; 4 dependency manifests (`pyproject` 30 deps, `pixi.toml` 21, `environment.yml` 45, stub `setup.py`); no ruff/mypy/pyright, only `[tool.black]` | No single source of truth for env or style |
| C8 | Test suite has no fast lane | unit:integration 1044:1 nominal; largest test files 19.7k and 11.6k lines; 37 tests shell out; zero `slow` markers despite `pytest.ini` declaring one | Cannot run a 2-minute subset locally |

### 6.4 Checklist

- [ ] **C-1** Split the flat layer in `pyproject.toml:118` into at least `cli/utils/diagnostics → analysis_workflows → analysis/labeling/refinement/tracking/… → registry → shared`; absorb current violations as `# RATCHET:` entries; require the count to decrease.
- [ ] **C-2** Add `chaser_analysis` as an import-linter root package, or fold it into `fisheye`.
- [ ] **C-3** Regenerate `file_size_ratchet_baseline.json` from all modules > 1,500 LOC (153 entries); keep the +200 tolerance.
- [ ] **C-4** Add ruff (`F`, `E`, `I`, `B`) in zero-new-violations mode, ratcheted by count.
- [ ] **C-5** Create `fisheye.shared.primitives` (sha256 canonicalization, utc_now, read_json, write_json_atomic); replace the ~270 clones by codemod; add one test asserting digest agreement on a fixture.
- [ ] **C-6** Delete `src/*.py` and `setup.py`; pick `pyproject.toml` as dependency truth and derive `environment.yml` / `pixi.toml` from it (or delete one).
- [ ] **C-7** Mark the 50 slowest tests `slow`; add a `-m "not slow"` lane to CI and document it as the local default.
- [ ] **C-8** Move `utils/` modules that are actually pipeline stages into their owning subpackage; leave `utils/` for true leaf helpers. Sequence behind the subtraction queue so dead modules are deleted, not moved.
- [ ] **C-9** Register the ~20 ops entry points that matter in `[project.scripts]` or one `palette <verb>` CLI; retire bespoke parsers as they are subsumed.

---

## 7. Governance: contracts, enforcement, process

### 7.1 Sampled contracts

| Contract doc | Runtime validator | Test | CI-gated |
|---|---|---|---|
| `zarr_run_completion_contract.md` | yes; gate in `stage_complete.py:277` (`_ENFORCE_STAGE_ARRAY_VALIDATION_FOR`, 7 of 25 stages) | yes | tests only; doc line 91 says fail-closed, code at 488 is fail-open **[VERIFIED]** |
| `zarr_payload_validation_receipt_contract.md` | yes (`track_kinematics.py`, `subject_shape_storage.py`) | `test_zarr_payload_receipt.py` | tests only; 1 producer family |
| `subject_mask_runs_contract.md` | yes (`finalize_subject_masks.py`, `publish_recording_bundle.py`) | yes (3) | tests only |
| `recording_manifest_contract.md` | yes (`validate_recording_manifest`) | yes | tests only |
| `training_quality_gate_contract.md` | yes (`refine_detect.py`, `run_detect_training_pipeline.py`) | yes (3) | tests only |
| `body_frame_contract.md` | yes (`pose/body_frame.py`, `body_frame_source_handle.py`) | yes | tests only |
| `chaser_escape_events_contract.md` | partial: materializer only; no `validate_*escape*`; known resolver bypass H2 | materializer tests | no |
| `keypoint_storage_contract_v2.md` | no dedicated validator | none by name | no |
| `mutable_review_runs_contract.md` | no; doc cites zero `.py` files | no | no |

### 7.2 Counts

- `docs/`: 385 files; 122 match `*_contract*.md`. `docs/diagnostics/`: 95 files, 2026-06-05 → 2026-09-01 (~1 per day).
- Contract-meta header present on 49 / 122 contract docs (103 / 385 overall). Status: 19 active, 27 draft, 2 superseded. Implementation: 28 implemented, 16 partial, 4 specified-only.
- `check_contract_freshness.py:44` scans only `docs/*contract*.md`, skips files without the header, ages only `active` ones. 73 contract docs are invisible to the one CI gate about contracts.
- Contract filename stem appears verbatim in code: 16 / 122 in `src`+`scripts`, 2 / 122 in `tests`.
- Diagnostics: 65 / 95 carry a status line across ~25 free-text vocabularies; 11 superseded; 13 no status; 31 not cross-referenced from any other doc.
- `grep -ri "error_budget|error budget" src scripts tests .github` → nothing. None of the commands named in the error budget policy's indicator columns exist under `scripts/`.

### 7.3 Strengths

- CI is real and green (recent runs mostly success including `main`) with seven mechanical gates.
- AGENTS.md's Required CI rule is versioned (`required-ci-integration-contract:v1`) and explicit that skipped checks are unrun evidence.
- `design_review_findings_2026-08-09.md` tracks every wave item to a landing commit.
- The error-budget policy's core idea (buy detection latency, not prevention; Tier 2 deliberately no-SLO) is correct engineering judgment.

### 7.4 Risks

| # | Finding | Evidence | Why it matters |
|---|---|---|---|
| G1 | Contracts outrun validators ~5:1 | 122 docs vs ~20-25 named validators; 3 of 9 samples have none | Unexecuted documents drift silently |
| G2 | Central gate is fail-open while its contract says fail-closed | `stage_complete.py:488` vs `zarr_run_completion_contract.md:91` **[VERIFIED]** | Docs describe the intended system; code ships the lenient one |
| G3 | Error budget has zero measurement | no code references; `implementation: specified-only` since 2026-08-11; no calendar hook for quarterly review | A wish, not a budget |
| G4 | Review output is unbounded and unindexed | ~1 diagnostic/day; 25+ status vocabularies; 31 orphans; three parallel queues (`design_review_findings`, `subtraction_queue`, `authority_consolidation_work_queue`), no roll-up | Reviews are cheap for agents; closure is not |
| G5 | AGENTS.md enforced by convention | no `.pre-commit-config.yaml`; `.claude/settings.local.json` is an allowlist incl. `Bash(git *)`; ~4 of ~211 rule lines have a mechanism | Only CI-visible rules bite |
| G6 | Process debt in the root | `agents_todo/` 20 briefs all `READY`/`HOLD`, newest Aug 10, landed briefs never closed; `playgrounds/` 14 GB on disk; 160/164 remote branches `agent/*`; 4 stale `.claude/worktrees/`; root `diagnostics/` and `docs/diagnostics/` coexist | Signal-to-noise for the next agent |
| G7 | Three environment systems | `environment.yml`, `pixi.toml`, `conda-packages-explicit.txt`/`pip-packages-exact.txt`; README shows `scripts/py` and `pixi run`; AGENTS.md says only `scripts/py` | Docs disagree with docs |

### 7.5 Checklist

- [ ] **G-1** Land O-1 before any new contract document is authored. Record the policy in AGENTS.md.
- [ ] **G-2** Make contract-meta mandatory: `check_contract_freshness.py` errors on any `docs/*contract*.md` without a header; add a required `validator:` field naming an importable symbol, checked by `importlib`. Contracts without one must declare `implementation: specified-only`.
- [ ] **G-3** Create `docs/diagnostics/INDEX.md` (or a registry table) with `id, doc, status ∈ {open, landed, waived, superseded}, commit`; CI fails if a new diagnostics doc is unlisted. Normalize the 25 status vocabularies to those four.
- [ ] **G-4** Collapse the three fix queues into the index; mark `design_review_findings_2026-08-09.md` and `subtraction_queue_2026-08-21.md` items by landing commit or `waived`.
- [ ] **G-5** Implement the first two error-budget indicators as scripts the nightly job runs (registry/resolver mismatch count from P-10; completion-epoch unsafe count from O-7); publish to the status page; write the quarterly review date into the index.
- [ ] **G-6** Add a pre-commit or Claude Code hook enforcing the two AGENTS.md rules that are cheapest to mechanize: block `pip install`/`conda install` in tool calls, and block plain `git push` in favor of the tracked helper.
- [ ] **G-7** Archive `agents_todo/` briefs that landed (cross-reference `HANDOFF_2026-07-05.md`); delete the 4 stale worktrees; prune merged `agent/*` branches; move root `diagnostics/` under `docs/diagnostics/` or delete.
- [ ] **G-8** Pick one environment manifest; delete or generate the others; make README agree with AGENTS.md.

---

## 8. What was not assessed

- No tests were run and no live zarr or registry was opened; counts are from static grep and may include dead code the subtraction queue already sequences for deletion.
- Scientific correctness of analyses was out of scope (see the 2026-08 five-lens design review and the GoodCopBadCop synthesis for that).
- Labeling web app, TensorRT realtime path, and acquisition/capture were not reviewed beyond their import edges.
- Performance was assessed by design reading only; no benchmarks were compared against `docs/*benchmark*` results.
