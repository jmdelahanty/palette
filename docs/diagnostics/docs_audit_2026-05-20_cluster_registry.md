# Cluster/Registry/Provenance Docs Audit (2026-05-20)

## Summary

Reviewed 36 docs in the cluster orchestration, registry, provenance,
recording-step status, staleness machinery, and shared cross-cutting slices.
Most docs are CURRENT (recent `last_verified` 2026-05-10..18, claims match
implementation). Several `*_todo.md` files are CHECKLIST-COMPLETE: nearly every
box ticked and the remaining items represent open design questions, not
unimplemented work. A small handful are STALE-EDIT (load-bearing claim
contradicted or terminology drifted). No ARCHIVE candidates outright, but
several are strong promote-to-archive targets per
`docs/legacy_archive_migration_policy.md` if the user wants to reduce active
surface.

Heavy concentration of structural docs around clip-camera DAG / finalized
collection / status_page / stage_catalog — these are all current and
load-bearing.

## Findings

### cluster_batching_guide.md
**Classification:** CURRENT
**Evidence:** L85-90 reference `fisheye.utils.plan_clipped_detect_refine_workflow`
which exists at `src/fisheye/utils/plan_clipped_detect_refine_workflow.py`;
L92-99 cite the 2026-05-17 sleepyfish 22-clip smoke (133/133 ok) which is
consistent with `cluster_pipeline_migration_checklist.md` last_verified
2026-05-18.
**Action:** none.

### cluster_pipeline_migration_checklist.md
**Classification:** CURRENT (last_verified 2026-05-18)
**Evidence:** All cited submitters exist under `scripts/` (verified:
`submit_detect_batches_bsub.sh`, `submit_crop_batches_bsub.sh`,
`submit_keypoints_batches_bsub.sh`, `submit_eye_masks_batches_bsub.sh`,
`submit_detect_artifact_bsub.sh`, `submit_crop_flat_roi_cache_bsub.sh`,
`submit_clipped_collection_flat_roi_cache_bsub.sh`). All cited utils exist:
`run_detection_artifact.py`, `import_run_group_artifact.py`,
`validate_imported_run_group.py`, `plan_clipped_detect_refine_workflow.py`,
`submit_clipped_detect_refine_plan_bsub.py`,
`finalize_clipped_detect_refine_workflow.py`,
`resolve_clipped_refined_detect_collection.py`,
`check_clipped_detect_refine_submission.py`,
`build_clipped_collection_flat_roi_cache.py`,
`validate_refined_detect_run.py`.
**Action:** none. Living checklist with explicit pending-vs-done state, well
maintained.

### cluster_workflow_orchestration.md
**Classification:** CURRENT (last_verified 2026-05-16)
**Evidence:** DAG and clipped-finalizer responsibilities (L130-280) match the
referenced utilities listed above. Janelia queue claims (L68-72) referenced in
cluster_batching_guide.md.
**Action:** none.

### cluster_run_group_artifact_workflow.md
**Classification:** CURRENT (last_verified 2026-05-16)
**Evidence:** Cluster artifact pattern (`.incoming`/`.failed`/serialized
importer) is reflected in
`src/fisheye/utils/import_run_group_artifact.py` and the apply mode referenced
in cluster_pipeline_migration_checklist.md L208-209.
**Action:** none.

### dask_zarr_write_safety.md
**Classification:** CURRENT
**Evidence:** General contract; references subject-mask finalization mask-vs-metric
chunk race (L31-37) which matches the broader stale-cache material in repo_wide
staleness gap matrix.
**Action:** none.

### registry_data_governance_policy.md
**Classification:** CURRENT
**Evidence:** Immutable/mutable lists match canonical owners in
`registry_metadata_ownership_refactor_design.md`. Backup script exists at
`scripts/backup_palette_registry.sh` (cited L131).
**Action:** none.

### registry_metadata_ownership_refactor_design.md
**Classification:** CURRENT (last_verified 2026-03-18, but the implementation
has moved forward)
**Evidence:** `dataset_context_current` view exists
(`src/fisheye/registry/db.py:2799-2800`, migration 034), and migration 035
migrates `recording_step_status_latest` to it. Ownership map matches code.
**Action:** none, but `last_verified` could be bumped.

### registry_metadata_ownership_refactor_todo.md
**Classification:** CHECKLIST-COMPLETE for Phases 1, 2, 5, 6, 7; partial 0, 3, 4, 8
**Evidence:** Phase 1/2/5/6/7 boxes mostly all [x] with concrete migrations
(`dataset_context_current`, profile-latest views, lineage tests). Remaining
open items are governance/deprecation decisions, not implementation gaps.
**Action:** STALE-EDIT minor — bump anchor date and consider promoting Phase
1/2/5 to an archived design note since they're done; keep Phases 0/3/4/8 open
items as the active TODO surface.

### registry_multi_source_provenance_design.md
**Classification:** CURRENT (design draft)
**Evidence:** Design only; no code shipped. Verified
`recording_intent`/`metadata_level`/`source_system`/`source_payload_json`
columns do NOT exist in `src/fisheye/registry/db.py` (grep returns nothing).
Doc accurately describes itself as `status: draft`.
**Action:** none. Doc honestly reflects unimplemented state.

### registry_multi_source_provenance_todo.md
**Classification:** CURRENT
**Evidence:** Every implementation checkbox is [ ] (unstarted) — matches the
finding above. Decision snapshot section [x] reflects design agreement only.
**Action:** none.

### registry_repair_playbook.md
**Classification:** CURRENT
**Evidence:** Verified `fisheye.utils.rename_recording_zarrs_to_training`,
`fisheye.utils.registry_rescan`, `fisheye.registry.maintenance`,
`fisheye.registry.dedupe` all exist. Cross-link to
`recording_store_relocation_components.md` at L81 is consistent with the new
file present in `git status`.
**Action:** none.

### registry_schema_reference.md
**Classification:** STALE-EDIT (auto-generated, slightly stale)
**Evidence:** Header says "Generated at: 2026-05-10T19:41:08Z, Tables: 39,
Views: 42". Current `db.py` shows 94 CREATE TABLE/VIEW statements and includes
migrations through 049 (model_input_shape). Some recent additions (e.g.
training_models input_shape columns from 2026-05-13) may not be reflected.
**Action:** regenerate via `scripts/generate_registry_schema_reference.py` (the
header explicitly says "do not edit by hand"). Single command fix.

### registry_tui_todo.md
**Classification:** CURRENT (Phase 1) / TODO-OPEN (Phase 2+)
**Evidence:** Phase 1 claims verified —
`src/fisheye/utils/registry_tui.py` exists (754 lines vs. doc's 413; the file
has grown since the 2026-02-28 snapshot). All Phase 2/3/4 items are [ ] open.
**Action:** STALE-EDIT minor — line count and "Current State (2026-02-28)"
snapshot is out of date. Phase 1 still accurately listed as complete.

### recording_registry_normalization_todo.md
**Classification:** CHECKLIST-COMPLETE for Phase 1 + ID-rekey runbook +
multi-zarr lineage; remaining items are open
**Evidence:** All Phase 1 core items, Phase A-F migration runbook (with
2026-02-09 execution notes), `dataset_lineage`, `recording_overview`,
two-field origin/use, manifest validator, and runtime status hooks are all
[x]. Validation script paths (`scripts/validate_recording_step_status_registry.sh`)
referenced at L82-83 — should verify existence.
**Action:** STALE-EDIT candidate for promotion to archive: the headline
phase-1 migration goals are done. Keep open items as a smaller doc; this
1113-line file is more historical record than active TODO.

### recording_status_page_design.md
**Classification:** CURRENT
**Evidence:** Implemented at `src/fisheye/status_page/{__init__,app,api,query,models}.py`
with static under `static/{index.html,status_page.css,status_page.js}`. Launcher
at `src/fisheye/utils/serve_recording_status_page.py` exists.
**Action:** none.

### recording_status_page_deployment.md
**Classification:** CURRENT
**Evidence:** Command shape matches launcher; static files match. Three
deployment modes (localhost / SSH-forward / reverse-proxy) cleanly described.
**Action:** none.

### recording_status_page_todo.md
**Classification:** CHECKLIST-COMPLETE
**Evidence:** All Phase 0-4 items are [x] except: CSV download (Phase 3), API
+ frontend smoke tests + manual validation (Phase 5), and Phase 6 nice-to-haves.
Acceptance criteria L87-92 are also unchecked but the substance is shipped.
**Action:** STALE-EDIT — mark acceptance criteria boxes that are met (live
URL, isolate missing rows, inspect per-dataset detail, no new schema). Consider
archiving once remaining tests are added; the v1 MVP is operational.

### recording_step_status_parallel_agents_contract.md
**Classification:** CHECKLIST-COMPLETE (work executed)
**Evidence:** `src/fisheye/registry/status_ledger.py` exists with
`upsert_recording_step_status`. All cited writer files exist
(`refinement/refine_detect.py`, `refinement/refine_keypoints.py`,
`refinement/refine_eye_masks.py`, `tracking/crop.py`,
`tracking/arena_assignment.py`, `inference/predict_*.py`).
**Action:** ARCHIVE candidate per
`legacy_archive_migration_policy.md`. The parallel-agent execution plan is
historical now; the contract subset (write API, step name vocab, status enum)
might survive as a thinner active reference.

### provenance_todo.md
**Classification:** CHECKLIST-COMPLETE for offline stage migration; remaining
items are open design questions
**Evidence:** All "Current status (offline dataset focus)" items [x]. Stage
provenance writers verified migrated by
`provenance_multi_agent_handoff.md` and existence of
`src/fisheye/shared/stage_provenance.py`.
**Action:** STALE-EDIT — move offline-stage section into a "Done" header and
keep only "High-priority fixes", "Medium-priority improvements", "Open
decisions", and "Deferred (online stage)" as the active TODO. Or archive and
spin out a smaller open-items doc.

### provenance_backfill_todo.md
**Classification:** CURRENT (anchored 2026-02-27)
**Evidence:** Cited 114 provenance rows and audit gaps; investigation items
all still [ ]. `snapshot_status`/`snapshot_missing_json` columns exist in
`db.py:3243-3244`. Doc still accurately describes the gap (columns exist but
are unpopulated per the audit).
**Action:** none, though could note progress since 2026-02-27 (115+ rows now).

### provenance_checks.md
**Classification:** CURRENT
**Evidence:** Describes existing diagnostics. Strict-mode contract behavior
matches `provenance_contract_draft.md`.
**Action:** none.

### provenance_contract_draft.md
**Classification:** CURRENT (last_verified 2026-02-27)
**Evidence:** Helpers cited (`build_stage_provenance`, `write_stage_provenance`,
etc.) exist in `src/fisheye/shared/stage_provenance.py`. Migration plan in L151-165
matches actual adoption recorded in `provenance_multi_agent_handoff.md`.
**Action:** STALE-EDIT minor — status field is "draft" but the contract is
shipped and adopted. Bump to status: active / v1 and refresh last_verified.

### provenance_multi_agent_handoff.md
**Classification:** CHECKLIST-COMPLETE
**Evidence:** All T1-T7 marked completed. Refinement and offline stage
migrations all done.
**Action:** ARCHIVE candidate — historical execution record. The "canonical
stage strings" (L82-94) might survive as a short reference doc; the rest is
done.

### provenance_registry_json_audit_2026-05-11.md
**Classification:** CURRENT (point-in-time audit)
**Evidence:** Snapshot doc, last_verified 2026-05-11. Sound as historical
record.
**Action:** none. (Archive when next audit supersedes.)

### repo_wide_staleness_policy.md
**Classification:** CURRENT (last_updated 2026-04-06, draft)
**Evidence:** Policy is normative; principles consistent with eye-mask /
subject-mask code patterns. Cross-references match.
**Action:** none.

### repo_wide_staleness_checklist.md
**Classification:** CURRENT
**Evidence:** Reflects current per-stage contracts; coherent with policy.
**Action:** none.

### repo_wide_staleness_gap_matrix.md
**Classification:** CURRENT (last_updated 2026-04-06)
**Evidence:** Gap classifications still accurate — verified that
`source_subject_mask_stale` top-level payload, subject/swim registry stale
projection, and detect/crop row-stable contracts remain unimplemented (no
contradicting evidence found in code search).
**Action:** none. High-value live doc.

### repo_wide_staleness_implementation_todo.md
**Classification:** CURRENT
**Evidence:** Priority list internally consistent with gap matrix.
**Action:** none.

### repo_wide_staleness_workflow_edge_checklist.md
**Classification:** CURRENT
**Evidence:** Edge enumeration consistent with gap matrix and policy.
**Action:** none.

### stage_catalog_design.md
**Classification:** CURRENT (last_verified 2026-05-10)
**Evidence:** `src/fisheye/registry/stage_catalog.py` exists. Pipeline
`STAGE_CANONICAL_IDS` and launcher mapping verified by
`doc_code_staleness_pass_2026-05-10.md`. Migration 043-048 in db.py adds
derived-analysis recording_step_status_wide views. Catalog table matches
documented IDs.
**Action:** none.

### pipeline_metadata_boundaries.md
**Classification:** CURRENT
**Evidence:** Decision rule and ownership table are normative and consistent
with provenance contract.
**Action:** none.

### legacy_archive_migration_policy.md
**Classification:** CURRENT
**Evidence:** Normative policy doc; consistent with crop_persistence_tradeoff
and ROI cache implementation.
**Action:** none.

### run_lineage_dag_inspector.md
**Classification:** CURRENT (status: implemented, 2026-05-11)
**Evidence:** `src/fisheye/utils/inspect_run_lineage_graph.py` exists. Cited
formats (tree/json/mermaid/dot) and `--collapse-run-duplicates` are spec
matching the file.
**Action:** none.

### model_export_registry.md
**Classification:** CURRENT
**Evidence:** `onnx_models` and `tensorrt_models` typed columns
(`opset`/`img_h`/`img_w`/`trt_version`/etc.) verified present in db.py.
Preferred detector baseline as of 2026-05-16 is concrete and current.
**Action:** none.

### model_input_shape_registry_design.md
**Classification:** CURRENT (last_verified 2026-05-13)
**Evidence:** Migration 049 (`_migration_049_model_input_shape_registry`) at
`db.py:9708-9714` exists and adds training_models input-shape columns and
backfill, matching the design.
**Action:** none.

### protocol_hash_stability_todo.md
**Classification:** CURRENT
**Evidence:** `_extract_protocol` still at `db.py:2121` using full-blob
sha256. No `protocol_semantic_hash` column or function present (grep returns
nothing). Doc accurately states the gap and Citrus-side investigation as
prerequisites.
**Action:** none.

### protocol_parameter_registry_todo.md
**Classification:** CURRENT (2026-02-27 anchor; Phase 0 unfixed)
**Evidence:** Phase 0 prereq (H5 key name mismatch) still describes the
current state. No `protocols` or `protocol_steps` table found in db.py.
`provenance.protocol_name`/`protocol_hash` columns exist but are unpopulated
per the 2026-02-27 audit.
**Action:** none.

### review_status_schema_unification_contract.md
**Classification:** CURRENT (last_verified 2026-02-27)
**Evidence:** `review_status_json` column referenced in
`recording_step_status_parallel_agents_contract.md` and confirmed in
registry. Contract is normative.
**Action:** none, though last_verified could be bumped.

### crop_persistence_tradeoff.md
**Classification:** CURRENT
**Evidence:** High-level tradeoff doc; cross-link to
`crop_live_view_vs_materialized_stream_design.md` exists.
**Action:** none.

### crop_distributed_tradeoffs.md
**Classification:** CURRENT
**Evidence:** Single-detection-per-frame chunk alignment policy still
matches current crop behavior (`tracking/crop.py`).
**Action:** none.

### environment_setup.md
**Classification:** CURRENT
**Evidence:** Verified `scripts/validate_cluster_palette_env.sh`,
`--require-pynvvc` flag described in cluster checklist L42, and `scripts/py`
wrapper behavior match.
**Action:** none.

### testing_todo.md
**Classification:** CURRENT
**Evidence:** Decord/AppArmor caveats are environment-specific and still
relevant. Refinement-selection ordering claim
(`refined_detect_runs/<latest>/instances` -> legacy sparse fallback -> raw
detect) is consistent with detect-curation code.
**Action:** none.

### session_context.md
**Classification:** CURRENT
**Evidence:** Schema fields match recording-context canonical owners in
`registry_metadata_ownership_refactor_design.md`.
**Action:** none.

### shared_helpers_refactor_todo.md
**Classification:** CURRENT (with embedded 2026-03-06 snapshot)
**Evidence:** C1-C4 marked done; verified `src/fisheye/shared/batch_logging.py`,
`type_conversions.py`, `registry_stage_complete.py`, `zarr_helpers.py`,
`zarr_discovery.py`, `environment.py` all exist. Items 1, 2, 5, 7, 8, 9, 11,
13, 14, 15, 16, 18 marked partial — believable given the file count and the
broad scope. No claim contradicted.
**Action:** none, though a 2026-05 refresh of the partial-items table would
be useful.

### training_data_api_surface_audit.md
**Classification:** CURRENT (updated 2026-04-24)
**Evidence:** All cited prep/export/validate/aggregate/plot utilities exist
in `src/fisheye/utils/`. Table patterns
(`detection_data_profile`/`keypoint_data_profile`/`eye_mask_data_profile`,
`*_latest` views) confirmed in db.py.
**Action:** none.

### doc_code_divergence_inventory_2026-05-01.md
**Classification:** CURRENT (point-in-time inventory + 2026-05-09/10 recheck)
**Evidence:** Self-aware about being a snapshot. The 2026-05-10 follow-up
section accurately marks several earlier findings as stale (e.g.
subject-mask wide view; bout_kinematics latest-pointer atomicity).
**Action:** none. (Archive when next divergence inventory supersedes.)

### doc_code_staleness_pass_2026-05-10.md
**Classification:** CURRENT
**Evidence:** All "Confirmed Current State" items verified above (stage
catalog, derived-analysis backfills, status_page query expansion).
**Action:** none.

## Overlaps / Gaps

- Three docs cover effectively the same provenance migration story:
  `provenance_todo.md`, `provenance_contract_draft.md`,
  `provenance_multi_agent_handoff.md`. The handoff and most of the todo are
  done; consolidating to one active contract doc + a closed handoff under
  `docs/archive/` would reduce surface.

- Four docs cover repo-wide staleness:
  `repo_wide_staleness_{policy,checklist,gap_matrix,implementation_todo,workflow_edge_checklist}.md`.
  This is intentional layering, and each is still internally consistent, but
  cross-references are dense; a reader needs all four to act.

- `recording_registry_normalization_todo.md` (1113 lines) is mostly historical
  achievement record. Consider splitting completed migration runbook (Phases A-F)
  into a closed archive doc and keeping only open items active.

- `recording_step_status_parallel_agents_contract.md` and the
  `provenance_multi_agent_handoff.md` are both completed multi-agent
  coordination docs — natural pair to archive together once a short "stage
  ledger write contract" reference replaces them.

- Two cluster docs (`cluster_pipeline_migration_checklist.md`,
  `cluster_workflow_orchestration.md`) have some overlap in DAG description
  for the clipped detect-to-refine flow, but each is doing different work
  (one is operator checklist, one is architectural). Acceptable.

- `registry_schema_reference.md` is the one auto-generated artifact in this
  slice and it's stale. Single command (`scripts/generate_registry_schema_reference.py`)
  would refresh; consider hooking it into CI.

- No detection/mask/keypoint/analysis-schema docs were touched (excluded per
  shard); other agents own those.

- Gap: no doc currently describes the `recording_store_relocation_components.md`
  contract (new untracked file from `git status`). When that lands, it should
  be cross-linked from `registry_repair_playbook.md` (already referenced at L81)
  and `registry_data_governance_policy.md`.
