# Palette Full Docs Pass — Master Report (2026-05-20)

Six-agent parallel audit covering 217+ active docs against current code, plus
an engineer-level critique of the codebase as a distributed computational
pipeline. Today is 2026-05-20.

## Per-Shard Reports

- [Detection / refinement](docs_audit_2026-05-20_detection.md) — 18 docs
- [Keypoints / pose / masks](docs_audit_2026-05-20_keypoints_masks.md) — 73 docs
- [Analysis / zarr / storage](docs_audit_2026-05-20_analysis_storage.md) — 47 docs
- [Cluster / registry / provenance](docs_audit_2026-05-20_cluster_registry.md) — 36 docs
- [Review / training / clipped](docs_audit_2026-05-20_review_training.md) — 28 docs (overlap with detection on a few)
- [Codebase engineering review](codebase_engineering_review_2026-05-20.md) — code, not docs

## Headline

Docs are in **better shape than the codebase**. Across ~200 audited docs the
distribution is roughly:

- **CURRENT:** ~85% — contracts and runbooks largely match the code
- **STALE-EDIT:** ~10% — partial drift (status labels, line numbers, a few
  checkboxes that should be ticked, two contracts in `draft` status that
  are shipped and adopted)
- **CHECKLIST-COMPLETE (archive candidates):** ~5%
- **ARCHIVE outright (now-historical coordination plans):** a small handful

There are **no severe doc/code contradictions** in this pass. The big
contradiction surfaced is in the *code*, not the docs: idempotence and
distributed-execution behavior are weaker than the contracts suggest.

Follow-up cleanup applied 2026-05-20:

- archived `detection_training_zarr_edit_todo.md`,
  `detect_quality_parallel_agents_contract.md`, and
  `subject_mask_component_provenance_followthrough_checklist.md`
- refreshed the highest-impact stale claims in
  `analysis_post_detection_workflow_status.md` and
  `experiment_types_reference.md`
- bumped `refined_detect_sparse_instances_schema.md` and
  `provenance_contract_draft.md` from draft to active

## Action List — Archive Moves (CHECKLIST-COMPLETE)

Move these to `docs/archive/` per `docs/legacy_archive_migration_policy.md`:

1. `detection_training_zarr_edit_todo.md` — Phase 0-3 all done; fold Phase 4
   remainder into `detection_review_web_todo.md` first.
2. `detect_quality_parallel_agents_contract.md` — companion TODO already
   archived; the view rename + module set shipped.
3. `eye_mask_parity_parallel_agents_contract.md` — wave 0-3 work delivered.
4. `pose_detect_parity_parallel_agents_contract.md` — same pattern, delivered.
5. `subject_mask_component_provenance_followthrough_checklist.md` — every
   item `[x]`; Done Definition met.
6. `provenance_multi_agent_handoff.md` — T1-T7 all completed; extract the
   "canonical stage strings" reference (L82-94) into a thin survivor doc
   before moving.
7. `recording_step_status_parallel_agents_contract.md` — multi-agent execution
   done; survivor should be a "stage ledger write contract" reference.
8. `refined_detect_downstream_adoption_checklist.md` (Palette portion) — split
   the cross-repo residue into a small tracker and archive the rest.

Conditional archives (good candidates once one item resolves):

- `subject_mask_stage_unification_todo.md` — Immediate TODO 1-6 all `[x]`;
  Open Questions belong in `subject_mask_refinement_todo.md`.
- `track_assignment_id_status.md` — self-marked historical; either archive or
  rename to `_history.md` with a banner pointing at
  `tracking_runs_contract_status.md`.
- `recording_status_page_todo.md` — Phase 0-4 done; tick acceptance criteria
  and archive once Phase 5 smoke tests are added.
- `keypoint_heading_validity_todo.md` — Phase 1/2 done; archive when Phase 3
  production backfill runs.

## Action List — STALE-EDIT Fixes (in-place edits, no archive)

Ordered by impact:

1. **`analysis_post_detection_workflow_status.md`** (highest drift) — Executive
   summary calls `stimulus_response` "design, not current workflow," but
   `stimulus_response.py` + 4 sibling modules are shipped and the compact-v2
   default switched on 2026-05-11. Rewrite Executive Summary + "Multi-Stimulus
   Readiness," or archive in favor of `current_pipeline_contract.md` +
   `stimulus_response_implementation_plan.md`.

2. **`experiment_types_reference.md`** — still claims H5 protocol key-name
   mismatch blocks extraction. `import_stimulus_to_zarr.py:756-759` handles
   both keys. Remove the blocker note.

3. **`detection_merged_export_contract.md`** — `status: draft` for behavior
   that's implemented in `utils/export_detect_training_zarr.py`. Bump to
   `active`; reconcile `prepare_detect_training` cross-reference (active
   module is `prepare_detect_training_from_registry.py`).

4. **`refined_detect_sparse_instances_schema.md`** — `status: draft` for the
   shipped `sparse_instances_v1`. Bump to `active`/`implemented`.

5. **`provenance_contract_draft.md`** — `status: draft` for an adopted
   contract. Bump to `active` v1.

6. **`registry_schema_reference.md`** — auto-generated, header dated
   2026-05-10, predates migration 049 and current 94 tables/views. Regenerate
   via `scripts/generate_registry_schema_reference.py`. Consider hooking into
   CI.

7. **`registry_tui_todo.md`** — "Current State (2026-02-28)" snapshot says
   413 lines; actual file is 754. Refresh snapshot; Phase 1 done claim stays.

8. **`detection_training_plan.md`** — module names drifted
   (`prepare_detect_training` vs `prepare_detect_training_from_registry`),
   default registry path drifted (`runs/registry/...` vs `/nvme1/...`). Either
   rewrite as a current pipeline index or mark historical.

9. **`detection_chunking_findings.md`** — bottom half ("Decord vs Native Decode
   Benchmark Plan") duplicates `detect_decode_backend_benchmark_todo.md`.
   Trim; verify loader line citations.

10. **`detection_review_web_todo.md`** — corrected today, but the "MVP TODO"
    framing has outgrown its scope (now covers two web reviewers + proxy
    videos + promotion). Rename to `detection_review_web_status.md` or split.

11. **`sam3_subject_mask_canary_plan.md`** — Phases 1-3 shipped per
    `paintera_palette_subject_mask_workflow.md` but checkboxes still `[ ]`.
    Tick them; refocus on Phase 4/5.

12. **`crop_review_workflow.md`** — broken link to non-existent
    `docs/crop_review_registry_todo.md`. Drop or fix. Add one line on
    geometry-only viewer behavior.

13. **`crop_distributed_tradeoffs.md`, `crop_persistence_tradeoff.md`** —
    predate multi-instance refined-detect and geometry-only crops. Either
    archive or add prominent redirect to
    `geometry_only_crop_workflow_cache_design.md`.

14. **`keypoints_pipeline_inline_registry_report.md`** — no `last_verified`;
    table likely stale post-parity contracts. Re-audit or mark historic.

15. **`subject_body_mask_qc_design.md`** — one item open ("Surface
    `requires_review` in mask review and overlay tooling"). Close or note.

16. **`subject_mask_refinement_todo.md`** (50KB) — heavy with completed
    items. Trim "Current State" to a short summary; restrict to open work.

17. **`subject_mask_stage_unification_todo.md`** — all Immediate TODO `[x]`;
    Open Questions remain. Reframe or split.

18. **Various TODOs with shipped Phase 1**: `keypoint_auto_approval_todo.md`,
    `keypoint_multi_skeleton_todo.md`,
    `keypoint_multi_skeleton_training_selection_todo.md`,
    `keypoint_heading_validity_todo.md`. Same pattern: trim done sections,
    refocus on remaining phase.

19. **Bump `last_verified` dates** on stable contracts whose claims still
    hold: `crimson_detect_review_acceptance_contract.md`,
    `detection_data_profile_schema_contract.md`,
    `review_status_schema_unification_contract.md`,
    `registry_metadata_ownership_refactor_design.md`.

## Documentation Gaps

1. **Clipped refined-detect resolver contract.** Multiple docs reference
   `fisheye.utils.resolve_clipped_refined_detect_collection`,
   `experiment_index/finalized_runs/`, and `recording_frame_index.parquet`.
   Confirm `clipped_recording_consumer_mapping_contract.md` is the anchor or
   write a dedicated resolver contract.

2. **`detect_training_promotion_backend` contract.** The post-save promotion
   hook in `src/fisheye/tune/detect_training_promotion_backend.py` is
   untracked in git. `analysis_to_training_promotion_contract.md` partially
   covers it; needs explicit coverage + cross-link from
   `detection_review_web_todo.md`.

3. **`refine_detect` per-frame top-k and dish-mask gating.** Documented only
   in workflow prose. Needs section in
   `refined_detect_sparse_instances_schema.md` or a small dedicated contract:
   covers `outside_dish_mask` decision label and the
   `quality_filtered_per_frame_top_k_sparse_instances_no_interpolation`
   method string.

4. **`detect_quality` artifact layer** (`detect_runs/<run>/quality_reports/`)
   has no schema doc since `detect_quality_registry_todo.md` was archived.

5. **`detect_review_web` vs `video_detect_review_web` comparison.** Two
   reviewers, two backing-data models, two use cases. Short operator-facing
   comparison would prevent confusion.

6. **`pose_kinematics_runs` contract** — write when implementation lands;
   only a design note exists.

7. **`auto_keypoint_review` operator runbook** — Phase 1 shipped but no
   how-to mirroring `eye_mask_profile_registry_ops_runbook.md`.

8. **`single_subject_per_arena` blocking-threshold enforcement** — when
   blocking activates.

9. **`recording_store_relocation_components.md`** (new untracked file) should
   be cross-linked from `registry_data_governance_policy.md` when it lands.

## Consolidation Opportunities

Multi-doc clusters worth merging (in rough order of payoff):

1. **Subject-mask refinement family** (5 docs): consolidate
   `subject_mask_refinement_todo.md`, `subject_mask_stage_unification_todo.md`,
   `subject_mask_component_provenance_followthrough_checklist.md`,
   `eye_subject_mask_unification_design.md`, `segmentation_stage_split_review.md`
   into one active TODO + archive the rest.

2. **Provenance migration** (3 docs):
   `provenance_todo.md` + `provenance_contract_draft.md` +
   `provenance_multi_agent_handoff.md` → one active contract + archived
   handoff.

3. **Keypoint multi-skeleton family** (5 docs): make
   `keypoint_pose_rollout_status.md` the single status doc; others remain
   focused TODOs.

4. **Tracking ID family** (3 docs): archive `track_assignment_id_status.md`;
   keep `tracking_runs_contract_status.md` (current) and
   `track_identity_target_architecture.md` (future).

5. **Detection web review docs** (3 docs):
   `detection_review_web_todo.md`, `detection_training_zarr_edit_todo.md`,
   `analysis_to_training_promotion_contract.md` describe overlapping scope;
   collapse the editing TODO into the contract.

6. **`recording_registry_normalization_todo.md`** (1113 lines): split
   completed Phase A-F migration runbook into an archived doc; keep open
   items active.

## Engineering Critique (Codebase, not Docs)

Full report: `codebase_engineering_review_2026-05-20.md`. The headline,
unflinching:

### What's strong
- **Stage vocabulary is real.** `registry/stage_catalog.STAGE_SPECS` is a
  declarative contract with `depends_on`/`invalidates`/`artifact_families`
  used by launcher + registry.
- **Single chokepoint for stage completion events** at
  `shared/registry_stage_complete.emit_stage_completion` (~49 call sites).
- **Per-stage signature/fingerprint discipline** exists where used —
  `crop_signature.py`, `run_lineage_fingerprint.py` do content-addressed
  SHA-256 over canonical JSON.
- **Provenance attrs are formalized** and consistent across detect/refine/crop.
- **Broad unit-test surface** (~405 tests) skewed toward leaf utilities,
  which is the honest pragmatic choice for research code.

### Top 3 risks (ranked by leverage)

**1. No completion-sentinel contract on zarr run groups.** Freshness is
decided by `<group>_runs.attrs['latest']`. Killed workers leave half-written
run groups that still read as `latest`. Downstream stages will silently
consume corrupt inputs. *This is the single highest-leverage thing to fix —
it's the failure mode distributed pipelines exist to prevent and yours
doesn't.* Fix: write under `<group>/<name>.partial/`, atomic rename to
`<name>/`, refuse to consider `.partial` as latest, make
`emit_stage_completion` enforce a sentinel.

**2. `registry/db.py` is 15,865 lines with an import cycle to `shared/`.**
`shared/registry_stage_complete.py` imports `..registry.db`; `registry/db.py`
imports `fisheye.shared.batch_logging`. Held together by Python's lazy
function-time imports. Every stage's extraction logic and SQL is in this one
file. Touching the schema is an under-tested edit. Fix: split
`_extract_<stage>_rows` into `registry/extractors/<stage>.py`; move
`registry_stage_complete` *into* `registry/`.

**3. No real DAG executor.** `core/pipeline.py` (1843 lines) is an
`if/elif` ladder with hard-coded `_run_<stage>` methods and three parallel
sources of truth for stage order (vs `stage_catalog.py` vs
`interactive_launcher.py`). `cli/batch_runner.py` is `subprocess.run` in a
loop. "Distributed" in `--scheduler distributed` means local Dask cluster.
Only one bsub submitter exists, for one workflow. Fix: pick one — Snakemake
or `dask-jobqueue` (LSFCluster) — and commit. `STAGE_SPECS` already gives
you 80% of what Snakemake needs.

### Other notable findings
- `tests/integration/` exists but contains **zero .py files**.
- 18 "unit" tests actually `subprocess.run` the CLI.
- **46 loose .py files under `src/` top-level** (debug_*, plot_*, *_analyzer.py)
  + ~16 more at repo root. Recent mtimes (e.g. `video_diagnostic_tool.py`
  2026-04-15) prove they're not all dead — just undeclared. Triage and either
  move under `src/fisheye/diagnostics/` or delete.
- Module-size hotspots: `refined_detect_curation.py` 2943,
  `stimulus_response.py` 2799, `track_kinematics.py` 2787,
  `subject_shape_runs.py` 2644, `train_detection.py` 2133,
  `interactive_launcher.py` 1921, `core/pipeline.py` 1843,
  `registry/maintenance.py` 9666, `registry/db.py` 15865.
- **Compact-v2 zarr migration is half-done**: writers exist with `if/else`
  legacy fallbacks and heuristic `_is_compact_v2_group` detection. No
  `layout_version` attr enforces it. Either commit the migration with a
  required `layout_version` and a one-shot upgrader, or stop calling it
  done.
- **Three competing stage taxonomies**: `stage_catalog.STAGE_SPECS`,
  `core/pipeline.STAGE_ORDER`, `interactive_launcher.STAGE_INFO`. Drift
  patched by `canonical_stage_id` aliases. Derive the latter two from the
  catalog.

### Distributed-pipeline scorecard (summary)

| Practice | Status | Note |
|---|---|---|
| Explicit DAG | Declared, not executed | `stage_catalog` has it; nothing schedules from it |
| Idempotent stages | Convention only | `attrs['latest']`, no sentinel |
| Content-addressed outputs | Partial | Fingerprints exist; not enforced as path/identity |
| Atomic writes | Sidecar JSONs only | Zarr writers don't use temp-then-rename |
| Registry as SoT | Yes, but with caveat | Authoritative for status, but freshness check often re-reads zarr attrs |
| Schema versioning | Partial | Compact-v2 is half-migrated with heuristic detection |
| Failure recovery | Manual | No retry, no resume from sentinel |
| Pipeline-glue test coverage | Weak | tests/integration/ is empty |
| Reproducible env | Partial | `environment.yml` + pinned exports; no lockfile-as-source-of-truth |
| Observability | Strong on registry/status | `recording_status_page` is solid; missing structured worker logs |

### Top 3 moves (dependency order)

1. **Completion-sentinel contract** — add `_complete` sentinel + atomic
   rename + registry-driven freshness. Highest leverage; single most
   important fix.
2. **Break the registry/shared cycle; split `registry/db.py`** — unblocks
   safe schema work.
3. **Derive `STAGE_ORDER` from `stage_catalog.dependency_map()`** — kills
   one of three taxonomies; sets up real DAG execution.

If only one ships, it should be #1.

## Suggested Execution Order

1. Apply the 8 outright archive moves (mechanical, low-risk).
2. Apply the small STALE-EDIT fixes that are one-line status bumps and
   regenerated artifacts (items 3-7 in the STALE-EDIT list).
3. Address engineering risk #1 (completion sentinel) — single biggest
   reliability win.
4. Apply the larger STALE-EDIT rewrites (items 1-2, 8-13).
5. Tackle consolidation opportunities one cluster at a time.
6. Address engineering risks #2 and #3.
7. Loose-script triage under `src/`.
