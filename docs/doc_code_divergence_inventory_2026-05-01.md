# Doc / Code Divergence Inventory — 2026-05-01

<!-- contract-meta
version: 1
status: inventory
last_verified: 2026-05-01
purpose: One-time snapshot of drift between contract/design docs and the code that should honor them. NOT a contract. Strategy for resolution lives in a separate doc.
-->

This is a snapshot, not infrastructure. Each item names a doc claim and the code reality. No fixes are proposed here beyond a "Resolution direction" hint. The strategy doc that follows this one is where reconciliation happens.

## 2026-05-09 Read-Only Recheck Notes

This section records a later read-only comparison against the current repository
state. It does not rewrite the original 2026-05-01 inventory; it marks which
high-signal findings still look current and which have become partially stale.

2026-05-10 follow-up: the canonical-stage-catalog slice now covers registry
consumers, the runtime pipeline stage map, the interactive launcher stage map,
and the first derived-analysis status rows. Treat the stage-vocabulary and
derived-analysis registry items below as partially addressed. Remaining gaps
are richer source-ref freshness semantics, writer-side status upserts, and
several provenance/status normalizations.

Still confirmed:

- The stage/DAG split is reduced but not gone. `step_cascade.py` now derives
  invalidation from the canonical catalog, `pipeline.py` exposes
  `STAGE_CANONICAL_IDS`, and `interactive_launcher.py` records canonical IDs
  for launcher stages. Runtime command names still remain intentionally
  separate from registry stage IDs.
- Derived analysis runs now have first-pass `recording_step_status` coverage
  for `track_kinematics`, `swim_bouts`, `bout_kinematics`, `eye_angles`,
  `subject_shape`, and `stimulus_response`. Remaining work is not basic
  visibility; it is source-ref freshness, staleness policy, and writer-side
  status emission.
- `stimulus_runs` and `speed_runs` still sit outside the stronger derived-run
  provenance discipline. `import_stimulus_to_zarr.py` has gained richer
  canonical step metadata, but the run itself is still not contract-compliant
  as a derived analysis run.
- Required-attribute drift remains real. Examples still observed:
  `bout_kinematics` lacks top-level `created_at_utc`; `eye_angle` uses flat
  source attrs rather than a unified `source_refs`; `track_kinematics` keeps
  most schema metadata on nested groups rather than at run root.
- Root-level one-time migration logs such as `ENUM_*.md` and
  `CRITICAL_REIMPORT_NEEDED.md` still exist at repo root and can read like
  active references.

Partially stale or changed since the snapshot:

- The "schema_version is checked in exactly one place" finding is now stale.
  `bout_classification_runs.py` also gates on `schema_version`, and export /
  manifest utilities read schema versions. The broader concern remains: most
  schema versions are still informational rather than enforced reader gates.
- The "registry wide status view omits subject-mask stages" finding is stale.
  `subject_masks`, `refined_subject_masks`, and subject-mask tuning columns are
  now in `recording_step_status_wide` via the catalog-driven view refresh.
- Body-frame metadata discipline improved for eye-angle runs. `eye_angle` now
  uses `build_keypoint_body_frame_contract_attrs`, so it writes canonical
  estimator version, coordinate space, angle convention, and source refs.
  `tail_kinematics` still uses non-canonical names such as
  `body_frame_convention` and `body_frame_source`.
- The "latest pointer atomicity is nowhere" theme is now stale for
  `bout_kinematics`: that writer now marks runs `running` / `complete` /
  `failed`, records visualization-artifact failures, and only updates parent
  `latest` after requested artifacts succeed. The global concern remains
  because many other writers still update `latest` ad hoc.
- The open question about cited pipeline modules being missing is resolved for
  the checked module list: all six cited modules existed at the expected paths
  during the 2026-05-09 recheck.

Practical interpretation:

- Treat this document as a useful hypothesis map, not a literal fix list.
- The highest-leverage strategy items now look like: keep runtime command names
  translated through the canonical stage catalog; make derived-analysis
  freshness compare source refs instead of only detecting run presence;
  normalize run-root provenance attrs for new analysis writers; and archive or
  close out stale migration-log docs.

## Summary

| Severity        | Count |
|-----------------|-------|
| load-bearing    | ~26   |
| metadata        | ~41   |
| cosmetic        | ~20   |
| open questions  | ~33   |
| **Total**       | ~120  |

Severity definitions:
- **load-bearing** — affects correctness, staleness propagation, reproducibility, or operator-visible runtime behavior.
- **metadata** — contract claims an attr/array/structure that's missing or extra, but no current behavior depends on it.
- **cosmetic** — naming, ordering, or doc phrasing only.
- **open question** — cannot be classified without your input.

Sections:
1. Pipeline & registry layer
2. Derived analysis run layer
3. Detection / refinement / crop layer
4. Eye mask / keypoint / pose layer
5. Cross-cutting (schema/provenance/body-frame/heading)
6. Writer × required-attr matrix
7. Open questions (consolidated)
8. Patterns & themes

---

## 1. Pipeline & registry layer

### 1.1 Stage DAG defined in five disagreeing places (load-bearing)
- **Code A:** `src/fisheye/core/pipeline.py:155–170` (`Pipeline.STAGE_ORDER`) — 14 stages including `detect_quality`, `refine`, `keypoints_refine`, `eye_masks`, `refined_eye_masks`, `refined_subject_masks`, `assign_ids`.
- **Code B:** `src/fisheye/core/pipeline.py:172–187` (`STAGE_DEPENDENCIES`).
- **Code C:** `src/fisheye/cli/interactive_launcher.py:39` (`STAGE_INFO`).
- **Code D:** `src/fisheye/cli/interactive_launcher.py:88–98` (`STAGE_ORDER`) — only 9 stages; missing `detect_quality`, `eye_masks`, `refined_eye_masks`, `refined_subject_masks`, `keypoints_refine`.
- **Code E:** `src/fisheye/registry/step_cascade.py:35–48` (`STEP_DEPENDENTS`) — different vocabulary again (`refined_detect`, `refined_keypoints`, `arena_assignment`, `tracks`).
- **Resolution direction:** unify (one source of truth + a translation table if vocabularies must remain).

### 1.2 Three independent stage-name vocabularies (load-bearing)
- `pipeline.py` uses: `refine`, `keypoints_refine`, `assign_ids`, `track`.
- `step_cascade.py` / `maintenance.py` use: `refined_detect`, `refined_keypoints`, `arena_assignment`, `tracks`.
- `recording_status_page_design.md:151–152` UX uses: "Refine Detect", "Refined Keypoints", "Arena Assignment", "Track".
- Each pair is naming the same logical stage. No documented translation table.
- **Resolution direction:** unify.

### 1.3 `Pipeline.STAGE_DEPENDENCIES` and `step_cascade.STEP_DEPENDENTS` disagree on `track` parents (load-bearing)
- **Pipeline says:** `track: ['keypoints']` (raw keypoints).
- **Cascade says:** `tracks` reached only via `arena_assignment`, which requires `refined_keypoints` (refined).
- **Doc says:** `current_pipeline_contract.md` does not resolve which is canonical.
- **Resolution direction:** decide whether `track` depends on raw or refined keypoints; update both.

### 1.4 `Pipeline.STAGE_DEPENDENCIES['refined_subject_masks'] = []` (load-bearing)
- **Code:** `src/fisheye/core/pipeline.py:184`.
- **Doc:** `docs/current_pipeline_contract.md:55–58` requires `subject_mask_runs/<run>` upstream.
- **Resolution direction:** update code.

### 1.5 `subject_masks` and `refined_subject_masks` invisible to status-page wide view (load-bearing)
- **Code:** `src/fisheye/registry/db.py:5827–5868` (`recording_step_status_wide` pivot) — no columns for subject-mask family.
- **Doc:** `docs/repo_wide_staleness_implementation_todo.md:42–55` says these must project into registry/query surfaces.
- **Resolution direction:** update code (extend pivot).

### 1.6 `subject_mask_tuning` registered but not pivoted (metadata)
- **Code:** `src/fisheye/registry/maintenance.py:51` includes `subject_mask_tuning` in `RECORDING_TUNING_STEP_NAMES`; `db.py:5864–5868` pivots only `dish_mask`, `detection_tuning`, `keypoint_tuning`, `eye_mask_tuning`, `subdish_mask_tuning`.
- **Resolution direction:** update code.

### 1.7 `interactive_launcher` does not require `detect_quality` before `refine` (load-bearing)
- **Code:** `src/fisheye/cli/interactive_launcher.py:62–69, 76–80` declares `refine.requires=['detect']`.
- **Doc:** `docs/recording_analysis_pipeline_contract.md:69–77` requires order `detect → detect_quality → refine`.
- **Resolution direction:** update code (launcher).

### 1.8 `interactive_launcher` `STAGE_ORDER` missing `eye_masks`, `refined_eye_masks`, `refined_subject_masks` (load-bearing)
- **Code:** `src/fisheye/cli/interactive_launcher.py:88–98`.
- **Doc:** `current_pipeline_contract.md` and pipeline-side `STAGE_ORDER` include them.
- **Resolution direction:** update code.

### 1.9 Derived analysis runs absent from `RECORDING_STEP_NAMES` and cascade (load-bearing)
- **Doc:** `current_pipeline_contract.md:59–61`, `derived_analysis_run_contract.md:319–342` declare `subject_shape_runs`, `tail_kinematics_runs`, `eye_angle_runs`, `bout_kinematics_runs`, `swim_bout_runs` as families.
- **Code:** none of these appear in `registry/maintenance.py:55–72` (`RECORDING_STEP_NAMES`) or `step_cascade.STEP_DEPENDENTS`.
- **Effect:** when `refined_subject_masks` or `refined_keypoints` regenerate, downstream analysis runs are never marked stale.
- **Resolution direction:** update code (decide whether analysis runs belong in registry; if yes, extend) **OR** update doc to say analysis runs are intentionally outside registry coverage.
- **2026-05-10 status:** partially addressed. Presence-level registry/status
  coverage now exists for `track_kinematics`, `swim_bouts`,
  `bout_kinematics`, `eye_angles`, `subject_shape`, and
  `stimulus_response`. Still open: `tail_kinematics_runs`,
  `tail_posture_view_runs`, `bout_classification_runs`, source-ref freshness,
  and writer-side status upserts.

### 1.10 `repo_wide_staleness_workflow_edge_checklist.md` calls `crop → subject_masks` "todo"; code already implements it (metadata)
- **Doc:** `docs/repo_wide_staleness_workflow_edge_checklist.md:64–65`.
- **Code:** `step_cascade.py:39` `"crop": frozenset({"keypoints", "subject_masks"})`.
- **Resolution direction:** update doc.

### 1.11 `detect → subject_masks` cascade "missing" per doc; transitively present in code (metadata)
- **Doc:** `repo_wide_staleness_workflow_edge_checklist.md:57`.
- **Code:** `step_cascade.py:36` reaches subject_masks via `detect → refined_detect → crop → subject_masks`.
- **Resolution direction:** update doc.

### 1.12 `refined_keypoints → tracks` listed as direct in doc; transitive in code (cosmetic)
- **Doc:** `repo_wide_staleness_workflow_edge_checklist.md:69` lists four direct downstream targets.
- **Code:** `step_cascade.py:41` lists only `eye_masks` and `arena_assignment`; `tracks` and `refined_eye_masks` are transitive.
- **Resolution direction:** clarify doc (transitive is fine; the user-facing `get_transitive_dependents` covers them).

### 1.13 `recording_status_page_design.md` lists `BG Full, BG DS` (cosmetic)
- **Doc:** `docs/recording_status_page_design.md:152`.
- **Code:** registry has unitary `background` step.
- **Resolution direction:** update doc.

### 1.14 `cluster_batching_guide.md` keypoints prereqs differ from code (metadata)
- **Doc:** `docs/cluster_batching_guide.md:45–50` says keypoints requires `detect, crop` (no background).
- **Code:** `pipeline.py:179–180` and `interactive_launcher.py:62–69` say `crop, background` (no detect).
- **Resolution direction:** update doc.

### 1.15 Crop/skip flag asymmetry in cluster guide (cosmetic)
- **Doc:** `cluster_batching_guide.md:46–50` table: detect uses `--overwrite`, crop uses `--force-new`.
- **Resolution direction:** unclear (could be intentional; document the asymmetry).

### 1.16 `recording_analysis_pipeline_contract.md` references possibly-missing modules (open)
- **Doc:** `docs/recording_analysis_pipeline_contract.md:33–36, 56–66` cites `fisheye.utils.import_recording_analysis`, `fisheye.utils.run_recording_analysis_pipeline`, `fisheye.utils.import_recordings_analysis`, `fisheye.utils.detect_quality_batch`, `fisheye.refinement.detect_quality`, `fisheye.refinement.refine_detect`.
- **Resolution direction:** verify each module exists at the cited path.

### 1.17 `analysis/<X>_profile_runs` (zarr) vs `<X>_data_profile` (registry table) naming (cosmetic)
- **Doc:** `artifact_storage_map.md:29–37` lists `analysis/detection_profile_runs`, `keypoint_profile_runs`, `eye_mask_profile_runs`.
- **Code:** registry tables are `detection_data_profile`, `keypoint_data_profile`, `eye_mask_data_profile`.
- **Resolution direction:** unclear; these are different layers (zarr vs SQL) but the divergent base names invite confusion.

### 1.18 `review_status_schema_unification_contract.md` symmetry of detect/keypoint quality review columns (open)
- **Doc:** `docs/review_status_schema_unification_contract.md:84–97`.
- **Code:** `db.py:3086, 11337+` — needs column-by-column verification.
- **Resolution direction:** open.

### 1.19 `recording_step_status` table mutates `latest` while doc says append-only (cosmetic)
- **Doc:** `docs/repo_wide_staleness_policy.md:30–44` says raw provenance runs are append-only.
- **Code:** `src/fisheye/registry/status_ledger.py:160–201` upserts `recording_step_status`; only `recording_step_status_history` is append-only.
- **Resolution direction:** none — design is intentional; clarify doc if confusing.

---

## 2. Derived analysis run layer

### 2.1 `bout_kinematics` does not write top-level `created_at_utc` (load-bearing)
- **Doc:** `docs/derived_analysis_run_contract.md` §"Required Run Attributes".
- **Code:** `src/fisheye/analysis/bout_kinematics.py:2714–2725` — `created_at_utc` only inside `provenance.timestamp_utc`.
- **Resolution direction:** update code.

### 2.2 `eye_angle_runs` lacks top-level `created_at_utc`, `source_refs`, `parameters` (load-bearing)
- **Doc:** `derived_analysis_run_contract.md` §"Required Run Attributes".
- **Code:** `src/fisheye/analysis/eye_angle_analysis.py:2748–2800` — flat `source_*_run` attrs instead of unified `source_refs`; no top-level `parameters` dict; no `created_at_utc`.
- **Resolution direction:** both — either update code or have contract accept flat attrs.

### 2.3 `track_kinematics` lacks `schema_id`, `schema_version`, `row_axis`, `source_refs` at run root (load-bearing)
- **Doc:** `derived_analysis_run_contract.md` §"Required Run Attributes".
- **Code:** `src/fisheye/analysis/track_kinematics.py:2374–2395, 2650–2683` — schema metadata only on nested groups.
- **Resolution direction:** both.

### 2.4 `swim_bout_runs` schema_id namespace differs (`palette.*` vs `analysis.*`) (metadata)
- **Doc:** other families use `analysis.<family>` form.
- **Code:** `src/fisheye/analysis/detect_bouts_multi_level.py:99` — `SWIM_BOUT_RUN_SCHEMA_ID = "palette.swim_bout_runs"`.
- Also missing `method`, `method_version`, `row_axis`, `source_refs` dict at run root (uses flat `detection_method` instead).
- **Resolution direction:** both.

### 2.5 `swim_bout_statistics` writer barely meets contract (load-bearing)
- **Doc:** `derived_analysis_run_contract.md` §"Required Run Attributes".
- **Code:** `src/fisheye/analysis/swim_bout_statistics.py:915–929` — only `created_at_utc`, `provenance`, `dataset_fields`, `report_version`. Missing `schema_id`, `schema_version`, `method`, `method_version`, `row_axis`, `source_refs`, `parameters`.
- Note: `track_kinematics_bout_status.md` recommends folding into a summary layer.
- **Resolution direction:** update code (or remove writer per status doc).

### 2.6 `tail_kinematics_runs` `row_axis="roi_rows"` not in contract's recommended list (metadata)
- **Doc:** `derived_analysis_run_contract.md` recommends `refined_subject_mask_rows` for that lineage; `tail_kinematics_run_design.md:~157` documents `roi_rows`.
- **Code:** `src/fisheye/analysis/tail_kinematics_runs.py:494`.
- **Resolution direction:** unclear.

### 2.7 `tail_kinematics_runs` `method_version` is integer; other writers use string (cosmetic)
- **Doc:** examples use `"<module>.v<N>"` form.
- **Code:** `tail_kinematics_runs.py:29` — `TAIL_KINEMATICS_METHOD_VERSION = 1`.
- **Resolution direction:** update code.

### 2.8 Tail kinematics design vs writer arrays (metadata)
- **Doc:** `docs/tail_kinematics_run_design.md:~148–198` lists `time_s`, `frame_index`, `valid`, `failure_reason_bytes`, `failure_reason`.
- **Code:** `tail_kinematics_runs.py:484` writes `frame_index`, `valid`, `failure_reason_bytes`. No `time_s`, no separate `failure_reason` string array. Adds `row_index/` lineage group + `frame_index_source` attr not in design.
- **Resolution direction:** update doc.

### 2.9 Tail kinematics design lists `tail_lateral_deflection_mm` (and tip variant) as optional; writer never produces them (metadata)
- **Doc:** `tail_kinematics_run_design.md:182–184`.
- **Code:** `tail_kinematics_runs.py:589–600` only `*_px`.
- **Resolution direction:** unclear (drop from design or implement).

### 2.10 `subject_shape_runs` schema v3 + method v8 naming asymmetry (cosmetic)
- **Code:** `src/fisheye/analysis/subject_shape_runs.py:51–55` — `SUBJECT_SHAPE_SCHEMA_VERSION = 3`, `SUBJECT_SHAPE_METHOD = "subject_shape_from_refined_masks_v8"`, `SUBJECT_SHAPE_METHOD_VERSION = 8`.
- Other writers use string `method_version` like `"eye_angle_analysis.v5"`. Subject_shape uses int.
- **Resolution direction:** unclear.

### 2.11 `derived_metrics_schema` attr not written by analysis-run writers (metadata)
- **Doc:** `docs/derived_metrics_schema_contract.md` §"Canonical Placement". `derived_analysis_run_contract.md` §"Relationship To `derived_metrics_schema`" allows analysis runs to use it.
- **Code:** grep finds zero hits in `src/fisheye/analysis/`. Only refined-keypoint runs emit it.
- **Resolution direction:** unclear (deferred-by-design or gap?).

### 2.12 `bout_kinematics` `source_refs` content is a superset of design (metadata)
- **Doc:** `bout_kinematics_run_design.md:382–407` lists keys.
- **Code:** `bout_kinematics.py:2635–2660` adds `source_track_kinematics_scope`, `source_track_kinematics_path`, `source_position_arrays` not in design.
- **Resolution direction:** update doc.

### 2.13 `*_valid` shadow array convention violated by bout_kinematics (metadata)
- **Doc:** `derived_analysis_run_contract.md` §"Validity And Failure State" recommends `<group>/{value_array, valid, failure_reason_bytes}`.
- **Code:** `bout_kinematics.py` uses one columnar table (`per_bout_metrics`) with inline `*_valid` columns instead of sibling arrays.
- **Resolution direction:** update doc to acknowledge columnar pattern.

### 2.14 `analysis_dense_array_migration_todo.md` items still valid (no drift, listed for clarity)
- **Doc:** `docs/analysis_dense_array_migration_todo.md` claims dense migration not done.
- **Code:** confirmed not done. No drift.

### 2.15 `pose_kinematics_run_design.md` aspirational only (no drift, listed for clarity)
- No `pose_kinematics` module exists in `src/fisheye/analysis/`. Design is forward-looking.

### 2.16 `cross_recording_analytics_export_design.md` unimplemented (no drift)
- Design only; no exporter code in scope.

### 2.17 `swim_bout_runs` `METHOD_VERSION` only in provenance, not run attrs (metadata)
- **Code:** `detect_bouts_multi_level.py` defines `METHOD_VERSION = "detect_bouts_multi_level.v7"` but writes only via `build_stage_provenance(version=…)`.
- **Resolution direction:** update code.

---

## 3. Detection / refinement / crop layer

### 3.1 Refined-run dense root still treated as required by code (load-bearing)
- **Doc:** `docs/refined_detect_collapse_v2.md:14–21` says run root is "metadata-only for current runs"; canonical data lives in `instances/` and `source_detections/`.
- **Code:** `src/fisheye/shared/refined_detect_curation.py:50–63` `CURATED_REFINED_REQUIRED_ARRAYS` enumerates dense root tuple (`refined_row_ids, frame_indices, entity_ids, …`) as required. `:996, 1050` falls back to `dense_frame_entity_v3`.
- **Resolution direction:** both — confirm whether new runs ever write dense root; relabel as legacy if not.

### 3.2 `source_kind_codes` label set differs between two refined-detect docs (metadata)
- **Doc A:** `refined_detect_collapse_v2.md` lists `none, raw_detect, interpolated, manual`.
- **Doc B:** `refined_detect_sparse_instances_schema.md` lists `raw_detect, manual, derived` (no `interpolated`, no `none`).
- **Code:** `refine_detect.py:572` writes only `raw_detect`; reverse map exists in `REFINED_SOURCE_KIND_CODE_MAP`. `detection_refinement_workflow.md:71–72` keeps `interpolated` "for legacy compatibility/provenance".
- **Resolution direction:** update doc (pick canonical set).

### 3.3 `frame_counts` advertised "recommended" in doc; required in code (metadata)
- **Doc:** `refined_detect_sparse_instances_schema.md` `(F,)` "recommended compatibility summary".
- **Code:** `refined_detect_curation.py:64–74` `CURATED_REFINED_INSTANCES_REQUIRED_ARRAYS` includes `frame_counts`.
- **Resolution direction:** update doc.

### 3.4 `source_detect_row_index` advertised optional; required in code (metadata)
- **Doc:** `refined_detect_sparse_instances_schema.md` lists optional.
- **Code:** `refined_detect_curation.py:64–74` includes it as required.
- **Resolution direction:** update doc.

### 3.5 `reason_bytes` row-side encoding not enumerated in writer's required tuple (metadata)
- **Doc:** sparse schema + `crimson_detect_bbox_read_contract.md:79` recommend `reason_bytes` for both `instances/` and `source_detections/` with attrs `reason_encoding`, `reason_bytes_width`, `reason_bytes_null_terminated`, `reason_fallback_order`.
- **Code:** `refine_detect.py:570–583` produces `instance_reason_labels`, `source_detection_reason_labels` (object-array strings) passed to `write_curated_refined_detect_surfaces`. Whether bytes encoding actually happens is internal to that helper and was not visible in the audit.
- **Resolution direction:** open — verify writer.

### 3.6 Refined-detect `interpolation` permanently disabled but doc still describes (cosmetic)
- **Doc:** `detection_refinement_workflow.md:41` says no longer normal; `refined_detect_collapse_v2.md` keeps `interpolated` source-kind code.
- **Code:** `refine_detect.py:33–51, 683–710` rejects `--max-gap` / `--method`; `interpolate_detections()` and `interpolate_gap()` retained at `:322–503` but unreachable.
- **Resolution direction:** update code (remove dead paths) or note retained for tests.

### 3.7 Crimson manual contract is `status: historical` describing legacy paths (load-bearing for Crimson interop)
- **Doc:** `docs/crimson_refined_detect_manual_contract.md` (`status: historical`, last_verified 2026-04-15).
- **Code:** No Palette writer creates `manual` subgroups for new runs. Legacy writer in `src/fisheye/tune/detect_review.py`. Crimson C++ side still expects this format.
- **Resolution direction:** update doc — publish a sparse-first Crimson manual write contract.

### 3.8 `detection_merged_export_contract.md` CLI flags suppressed in `prepare_*` and renamed in `export_*` (load-bearing)
- **Doc:** `docs/detection_merged_export_contract.md` defines `--merge`, `--out-zarr`, `--split`, `--seed`, etc.
- **Code:** `src/fisheye/utils/prepare_detect_training_from_registry.py:561–568` lists those as `argparse.SUPPRESS`. Active exporter is `src/fisheye/utils/export_detect_training_zarr.py` with renamed flags (`--merge-out-zarr`, etc.).
- **Resolution direction:** both — reconcile flag names.

### 3.9 `--merge-frame-chunk` chunking-findings recommendation not implemented (metadata)
- **Doc:** `docs/detection_chunking_findings.md:40` recommends adding `--merge-frame-chunk`; chunk depth 8–32 frames.
- **Code:** `export_detect_training_zarr.py` uses hardcoded chunks (`(8192,)`); no flag.
- **Resolution direction:** update doc (mark open) or implement.

### 3.10 Detect decode-backend benchmark TODO partially implemented (metadata)
- **Doc:** `docs/detect_decode_backend_benchmark_todo.md` proposes Decord GPU/CPU, OpenCV, Crimson backends.
- **Code:** `diagnostics/benchmark_detect_decode_backends.py` and `benchmark_video_decode.py` exist; production `detect_yolo.py:154–189` only switches Decord-GPU vs Decord-CPU. No backend selector abstraction.
- **Resolution direction:** update doc (mark partial).

### 3.11 `detect_keypoints_parity_contract` Phase 3 status (open)
- **Doc:** `docs/detect_keypoints_parity_contract.md:171–175` Phase 3 unchecked.
- **Code:** Phase 1 + 2 entrypoints exist; Phase 3 enforcement not audited.
- **Resolution direction:** verify.

### 3.12 Crop storage migration TODO phases done in code but unchecked in doc (metadata)
- **Doc:** `docs/crop_storage_mode_migration_todo.md` Phases 0/1 unchecked; Phase 5 ("writer opt-in mode") unchecked.
- **Code:** `src/fisheye/shared/crop_image_source.py:177` supports `materialized` and `geometry_only`; `:111` resolves `latest_any` / `latest_materialized` / `latest`. `tracking/crop.py:134` `_VALID_CROP_STORAGE_MODES = {"materialized", "geometry_only"}`. Phase 5 first three items appear done.
- **Resolution direction:** update doc.

### 3.13 Crop image source mode names differ between doc and code (metadata)
- **Doc:** `crop_storage_mode_migration_todo.md` uses `geometry_live`, `geometry_live_gpu`, `geometry_cache_build`, `geometry_cache_reuse`.
- **Code:** `crop_image_source.py:548–651` uses `materialized` vs `geometry_only`; `roi_read_mode = "materialized_crop_run"` or `"geometry_only_live"`.
- **Resolution direction:** update doc.

### 3.14 `predict_detections.py` writer surface vs `frame_counts`/`n_detections` aliasing (open)
- **Code:** `src/fisheye/inference/predict_detections.py:70` references both keys.
- **Doc:** `crimson_refined_detect_manual_contract.md:55–57` says they are aliases; `crimson_detect_bbox_read_contract.md:87–88` lists `frame_indices` + `bbox_norm_coords` for raw-detect.
- **Resolution direction:** open — needs deeper writer audit.

### 3.15 Detection profile writer field-by-field vs contract (open)
- **Doc:** `docs/detection_data_profile_schema_contract.md:84` storage at `analysis/detection_profile_runs/<run>/`; required attrs include `schema_name`, `schema_version`, `profile_summary`.
- **Code:** `src/fisheye/utils/backfill_detection_profiles.py:216` references the path; field-by-field verification not done.
- **Resolution direction:** verify.

### 3.16 `validate_detect_training_zarr.py` invariant coverage (open)
- **Doc:** `detection_merged_export_contract.md` §"Invariants".
- **Code:** validator exists; invariant-by-invariant coverage not audited.
- **Resolution direction:** verify.

### 3.17 Phase-1 "preferred detect/crop" docs are archived; user task scope lists them at top level (cosmetic)
- **Doc:** `docs/archive/preferred_detect_crop_phase1_*.md` (4 files).
- **Code:** No live `preferred_detect_runs` writes; only `DEFAULT_PREFERRED_CROP_POLICY_NAME`/`DEFAULT_PREFERRED_ROI_SIZE` in `src/fisheye/shared/crop_geometry.py:12–13` (unrelated naming residue).
- **Resolution direction:** confirm phase-1 archived stays archived.

### 3.18 `crimson_palette_zarr_alignment_todo.md` is archived (cosmetic)
- File now lives at `docs/archive/`.
- **Resolution direction:** confirm done.

### 3.19 Crimson `--target-group` / preference-chain parity OK (no drift, listed for clarity)
- **Doc:** `crimson_detect_review_acceptance_contract.md:96–99`.
- **Code:** `src/fisheye/shared/refined_detect_review.py:12` matches.

---

## 4. Eye mask / keypoint / pose layer

### 4.1 `POSE_SCHEMA_GUIDE.md` describes nodes as `List[str]` but writer uses dicts (load-bearing)
- **Doc:** `POSE_SCHEMA_GUIDE.md:24–37, 56–64` says `pose_schema['nodes']` is `["bladder", "eye_left", "eye_right"]`; node ID = list index.
- **Code:** `src/fisheye/pose/schema.py:294` `schema_to_attr_payload` writes `nodes` as `[{"id": …, "name": …}]` dicts. Also writes a separate `keypoint_labels` field.
- **Resolution direction:** update doc.

### 4.2 `POSE_SCHEMA_GUIDE.md` references nonexistent `traditional_feret_v1` (metadata)
- **Doc:** `POSE_SCHEMA_GUIDE.md:188–206`.
- **Code:** `configs/fisheye/pose_schemas/` only has `traditional_v1.json`, `traditional_v2.json`.
- **Resolution direction:** update doc.

### 4.3 `POSE_SCHEMA_GUIDE.md` uses legacy `bladder` label (cosmetic)
- **Doc:** `POSE_SCHEMA_GUIDE.md:27, 64, 109, 160, 184, 307`.
- **Code:** `src/fisheye/pose/schema.py:13–23` canonical `swim_bladder`; `bladder` is alias.
- **Resolution direction:** update doc.

### 4.4 `dish_mask_batch_review.md` references missing `backfill_experimental_chamber` module (load-bearing)
- **Doc:** `docs/dish_mask_batch_review.md:28–30`.
- **Code:** No `src/fisheye/utils/backfill_experimental_chamber.py` exists.
- **Resolution direction:** unclear (ship code or remove doc step).

### 4.5 Eye-mask row-mapping contract attr name mismatch (metadata)
- **Doc:** `docs/eye_mask_row_mapping_contract.md:53–56` uses `source_eye_masks_run`.
- **Code:** writers use `source_eye_run` (singular). `eye_mask_training_artifact_contract.md:115` agrees with code.
- **Resolution direction:** update doc.

### 4.6 Eye-mask training artifact attr coverage (open)
- **Doc:** `docs/eye_mask_training_artifact_contract.md:113–121` requires `source_eye_run`, `source_eye_stage_label`, `source_eye_stage`, `source_eye_stage_role`, `source_eye_authority_stage`.
- **Code:** `src/fisheye/segmentation/infer_unet_eye_masks.py`, `eye_segmentation_yolo.py`, `eye_segmentation.py` — only `source_crop_run` and `source_eye_run` confirmed.
- **Resolution direction:** open — verify per producer.

### 4.7 Eye-mask profile required attrs vs writer (open)
- **Doc:** `docs/eye_mask_data_profile_schema_contract.md:99–110`.
- **Code:** `src/fisheye/utils/eye_mask_profile.py` and `backfill_eye_mask_profiles.py` exist; field-by-field coverage not verified.
- **Resolution direction:** verify.

### 4.8 `pose_schema_heuristics_split_proposal.md` status header stale (cosmetic)
- **Doc:** `docs/pose_schema_heuristics_split_proposal.md:7` says "design proposal, not active contract"; body claims packaged profiles already exist.
- **Resolution direction:** update doc (status: partially shipped).

### 4.9 Heading-computation contract vs runtime resolver — aligned (no drift)
- **Doc:** `docs/keypoint_heading_computation_contract.md`.
- **Code:** `src/fisheye/pose/heading.py:37–66` `resolve_heading_computation` honors override → `pose_schema.metadata.heading_computation` → deprecated alias. Formula `atan2(-dy, dx)` matches.

### 4.10 `derived_metrics_schema` refined-keypoint v1 example — aligned (no drift)
- **Doc:** `derived_metrics_schema_contract.md:177–196`.
- **Code:** `src/fisheye/shared/derived_metrics_schema.py:24+` `build_refined_keypoint_derived_metrics_schema` matches; consumed by `refine_keypoints.py:1711` and backfill at `utils/backfill_keypoint_derived_metrics_schema.py:12`.

### 4.11 `derived_metric_*` named layer overlaps with `derived_metrics_schema` semantic layer (metadata)
- **Doc A:** `docs/keypoint_derived_metric_schema_contract.md` defines named anatomy metrics.
- **Doc B:** `docs/derived_metrics_schema_contract.md` defines semantic gate metrics.
- **Code:** Both layers exist (`pose/metric_schema.py` writes `derived_metric_*`; `shared/derived_metrics_schema.py` writes `derived_metrics_schema`). Sample profile (`keypoint_data_profile_schema_contract.md:177–184`) carries both.
- **Resolution direction:** update doc (clarify cross-references and which attr means which).

### 4.12 `eye_mask_parity_parallel_agents_contract.md` lacks closeout marker (cosmetic)
- **Doc:** Wave 0/1/2/3 contract.
- **Code:** All target modules exist (`sync_eye_mask_profile_registry`, `aggregate_eye_mask_training_data_card`, `plot_*`, `export_*`, `validate_*`, `backfill_eye_mask_profiles`, `prune_legacy_eye_mask_profile_runs`, `resolve_eye_mask_stale`).
- **Resolution direction:** update doc (status closeout).

### 4.13 Refined-eye lineage arrays (open)
- **Doc:** `eye_mask_row_mapping_contract.md` lists `frame_counts`, `source_refined_row_ids`, `source_detect_row_index`.
- **Code:** `src/fisheye/shared/row_lineage.py:13–28` declares `ROW_LINEAGE_ARRAYS`. `copy_row_lineage_arrays` invoked in `eye_segmentation_yolo.py:857` and `infer_unet_eye_masks.py:791`. Not all five names verified individually.
- **Resolution direction:** verify per array.

### 4.14 Heading mismatch warning enforcement (open)
- **Doc:** `keypoint_heading_computation_contract.md` `dependent_keypoints` mismatch warning.
- **Code:** No validator emits the warning.
- **Resolution direction:** open.

---

## 5. Cross-cutting (schema/provenance/body-frame/heading)

### 5.1 Top-level `schema_id` / `schema_version` missing on `track_kinematics` (load-bearing)
See **2.3**. Restated here as a cross-cutting symptom of writer-by-writer divergence.

### 5.2 `row_axis` missing on track_kinematics, stimulus_response, stimulus, speed, profile runs (metadata)
- **Code:** see writer × attr matrix in §6.
- **Resolution direction:** update code.

### 5.3 `analysis/speed_runs/` not declared in any contract (load-bearing)
- **Doc:** `derived_analysis_run_contract.md`, `current_pipeline_contract.md` do not mention.
- **Code:** `src/fisheye/analysis/compute_speed.py:665, 1020` writes `analysis/speed_runs/<run>` and bypasses `build_stage_provenance`.
- **Resolution direction:** unclear — deprecate in favor of track_kinematics or document.

### 5.4 `tail_kinematics_runs` not in `derived_analysis_run_contract.md` relationship list (cosmetic)
- **Doc:** `derived_analysis_run_contract.md:156–204` does not mention; `current_pipeline_contract.md:60` does.
- **Resolution direction:** update doc.

### 5.5 `schema_version` is write-only for almost all run families (metadata)
- **Doc:** `derived_metrics_schema_contract.md` §"Reader Rules" implies presence is authoritative.
- **Code:** Reader gates only at `utils/check_recording_steps.py:842` and `analysis/bout_kinematics.py:643–646` (eye-angle only). Visualizers read but only for display. All other schema_versions are decorative.
- **Resolution direction:** unclear — relax contract or add reader gates.

### 5.6 `stimulus_runs`, `speed_runs`, `*_profile_runs`, `chaser_state_interpolator` skip `build_stage_provenance` (load-bearing)
- **Doc:** `provenance_todo.md` does not list these as migrated or as TODO.
- **Code:** confirmed via grep that none of these writers call the helper.
- **Resolution direction:** update doc + code.

### 5.7 `provenance_todo.md` open items still open (metadata)
- **Doc:** `provenance_todo.md:34, 44` — open: model_path/hash, config_hash, environment normalization, persist resolved source group, link manual review artifacts, consolidated_metadata as source of truth.
- **Code:** `build_stage_provenance` lacks `model` field; inputs dict varies by writer.
- **Resolution direction:** update both.

### 5.8 `body_frame_estimator_version` / `coordinate_space` / `angle_convention` missing on all writers (metadata)
- **Doc:** `docs/body_frame_contract.md:153–171`.
- **Code:** subject_shape writes `body_frame_schema_id`, `body_frame_schema_version`, `body_frame_estimator`, `body_frame_source_refs` but missing the three above. eye_angle writes only schema_id/version. tail_kinematics uses non-canonical names.
- **Resolution direction:** update code.

### 5.9 `tail_kinematics` uses non-canonical body-frame attr names (load-bearing)
- **Doc:** `body_frame_contract.md:153–171` mandates `body_frame_angle_convention`, `body_frame_source_refs` (dict).
- **Code:** `tail_kinematics_runs.py:499–500` writes `body_frame_convention` and `body_frame_source` (string).
- **Resolution direction:** update code.

### 5.10 No coded resolver between pose-schema heading and subject_shape body-frame heading (metadata)
- **Doc:** `body_frame_contract.md:280–292` mentions both producers can exist.
- **Code:** No arbiter; consumers prefer subject_shape ad-hoc.
- **Resolution direction:** unclear.

### 5.11 `CRITICAL_REIMPORT_NEEDED.md` post-migration leftover (load-bearing if archives still exist in old format)
- **Doc:** Root-level `CRITICAL_REIMPORT_NEEDED.md`.
- **Code:** Writers conform to new format (`chaser_state_interpolator.py:133–157`, `import_stimulus_to_zarr.py:_copy_enums`). Readers (`diagnostics/inspect_stimulus_events.py:49`) assume new format — no automated detection of old encoding.
- **Resolution direction:** unclear — deprecate doc as one-time event OR add tolerant reader.

### 5.12 ENUM_*.md cluster matches code state but reads as migration log (cosmetic)
- **Doc:** Root-level `ENUM_COLUMNAR_FORMAT_CHANGES.md`, `ENUM_FINAL_SUMMARY.md`, `ENUM_IMPLEMENTATION_SUMMARY.md`, `ENUM_PATHS_QUICK_REFERENCE.md`.
- **Code:** writers conform.
- **Resolution direction:** move to `docs/archive/` or delete.

### 5.13 `latest` pointer write policy not centralized (metadata)
- **Doc:** `src/fisheye/docs/zarr_structure.md:43–55` — "most `*_runs/` groups carry `attrs[latest]`"; vague.
- **Code:** ad-hoc per writer; no helper, no atomicity.
- **Resolution direction:** update doc to say best-effort, or add helper.

### 5.14 Run identity is timestamp-only, not content-deterministic (no doc claims otherwise — listed for §8 themes)
- No drift; flagged because the lack of content-hash run UIDs makes dedupe and "regenerate this exact run" queries impossible.

### 5.15 `zarr_structure.md` says `tracking_runs/` at root; code writes `analysis/track_kinematics_runs/{online,offline}` (load-bearing)
- **Doc:** `src/fisheye/docs/zarr_structure.md:36`.
- **Code:** `src/fisheye/analysis/track_kinematics.py:2362–2647`.
- **Resolution direction:** update doc.

### 5.16 `refined_eye_masks_runs/` listed as root child in zarr_structure but compatibility-only in pipeline contract (metadata)
- **Doc A:** `src/fisheye/docs/zarr_structure.md:34` lists it.
- **Doc B:** `current_pipeline_contract.md:57` says "historical or derived compatibility layout".
- **Code:** still written.
- **Resolution direction:** update doc (decide canonical).

### 5.17 `stimulus_runs` writer skips contract entirely (load-bearing)
- **Doc:** `derived_analysis_run_contract.md` declares it a member family.
- **Code:** `src/fisheye/analysis/import_stimulus_to_zarr.py:866` writes ad-hoc attrs (no `schema_id`, `schema_version`, `method`, `method_version`, `row_axis`, `source_refs`, `parameters`, no `provenance` block).
- **Resolution direction:** update code.

### 5.18 `derived_metrics_schema` declared scope is refined-keypoint only; eye_angle emits a similar shape (open)
- **Doc:** `derived_metrics_schema_contract.md:36` declares immediate target = refined-keypoint.
- **Code:** `eye_angle_analysis.py:454–545` builds shape-compatible payloads.
- **Resolution direction:** widen contract or relabel eye_angle's structure.

---

## 6. Writer × required-attr matrix

Required by `docs/derived_analysis_run_contract.md` §"Required Run Attributes": `schema_id`, `schema_version`, `method`, `method_version`, `created_at_utc`, `row_axis`, `source_refs`, `parameters`, `provenance`.

| Writer (path) | schema_id | schema_version | method | method_version | created_at_utc | row_axis | source_refs | parameters | provenance |
|---|---|---|---|---|---|---|---|---|---|
| `subject_shape_runs.py:1055` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | (in prov) | ✓ |
| `tail_kinematics_runs.py:486` | ✓ | ✓ | ✓ | ✓ (int, not string) | ✓ | ✓ (`roi_rows`, non-standard) | ✓ | (in prov) | ✓ |
| `eye_angle_analysis.py:2748` | ✓ | ✓ | ✓ | ✓ | ✗ (only in prov) | ✓ | ✗ (flat attrs, not dict) | ✗ (only in prov) | ✓ |
| `bout_kinematics.py:2714` | ✓ | ✓ | ✓ | ✓ | ✗ (only in prov) | ✓ | ✓ | ✓ | ✓ |
| `detect_bouts_multi_level.py:2002` (`swim_bout_runs`) | ✓ (`palette.*`) | ✓ | ✗ (`detection_method` instead) | ✗ | ✓ | ✗ | ✗ (only in prov inputs) | (in prov) | ✓ |
| `track_kinematics.py:2374, 2638` | ✗ (only nested) | ✗ (only nested) | ✓ | ✗ | ✓ | ✗ | ✗ | (in prov) | ✓ |
| `compute_speed.py:993` (`speed_runs`) | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | (inline) | ✗ (no `build_stage_provenance` call) |
| `stimulus_response.py:1710` | ✗ | ✗ | ✗ (only in prov) | ✗ | ✗ (only in prov) | ✗ | (only in prov inputs) | (in prov) | ✓ |
| `import_stimulus_to_zarr.py:866` (`stimulus_runs`) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| `detection_profile.py:836` | ✗ (nested only) | partial | n/a | n/a | n/a | ✗ | n/a | n/a | n/a |
| `keypoint_profile.py:875` | ✗ | partial | n/a | n/a | n/a | ✗ | n/a | n/a | n/a |
| `eye_mask_profile.py:1026` | ✗ | partial | n/a | n/a | n/a | ✗ | n/a | n/a | n/a |
| `arena_assignment.py` / `single_subject_per_arena.py` | (not visible in grep) | (not visible) | ✓ | n/a | (in prov) | ✗ | partial | partial | ✓ |
| `chaser_state_interpolator.py:717` | ✗ | ✗ | n/a | n/a | n/a | ✗ | n/a | n/a | ✗ |
| `swim_bout_statistics.py:915` | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ |

Cleanest writer: `subject_shape_runs`. Worst writer: `import_stimulus_to_zarr` (zero of nine).

### Schema-version write/read asymmetry

Writers (places that emit `schema_version`):
- `bout_kinematics.py:2715` (`SCHEMA_VERSION = 7`)
- `subject_shape_runs.py:1058` (`SUBJECT_SHAPE_SCHEMA_VERSION = 3`)
- `detect_bouts_multi_level.py:2003` (`SWIM_BOUT_RUN_SCHEMA_VERSION = 6`)
- `tail_kinematics_runs.py:489` (`TAIL_KINEMATICS_SCHEMA_VERSION = 1`)
- `subject_shape_runs.py:1080`, `eye_angle_analysis.py:214/594` (`BODY_FRAME_SCHEMA_VERSION`)
- `subject_shape_runs.py:1095` (`TAIL_GEOMETRY_SCHEMA_VERSION`)
- `eye_angle_analysis.py:383, 505` (`EYE_ANGLE_VARIANT_*`, `EYE_ANGLE_OUTPUT_*`)
- `detect_bouts_multi_level.py` (`DETECTION_SIGNAL_*`, `PEAK_EVENT_*`)
- `track_kinematics.py` (`MOVEMENT_*`, `SPEED_DERIVATIVE*`)
- `refinement/finalize_subject_masks.py:1237` (`SUBJECT_BODY_MASK_QC_SCHEMA_VERSION`, refined mask metrics `1`)
- `pose/metric_schema.py:245` (`derived_metric_schema_version`)
- `shared/zarr/schema.py:183` (root `schema_version='3.0.0'`)

Readers (places that actually check):
- `utils/check_recording_steps.py:842` — eye-angle run schema_version vs `_EYE_ANGLE_RUN_SCHEMA_VERSION`.
- `analysis/bout_kinematics.py:643–646` — refuses eye-angle source if `schema_version < 2`.
- `visualization/interactive_track_kinematics.py:786` — informational rendering only.
- `visualization/visualize_eye_angles.py:960` — informational rendering only.

**Net:** ~12 schema_version slots are written, 1 has a real reader gate (eye-angle).

### `build_stage_provenance` / `write_stage_provenance` callers vs non-callers

Callers (in scope): `arena_assignment`, `single_subject_per_arena`, `tail_kinematics_runs`, `bout_kinematics`, `subject_shape_runs`, `stimulus_response`, `track_kinematics`, `refine_keypoints`, `finalize_subject_masks`, `refine_eye_masks`, `crop`, `detect_bouts_multi_level`, `refine_detect`, `detect_keypoints_traditional`, `eye_segmentation`, `swim_bladder_segmentation`, `eye_segmentation_yolo`, `infer_unet_subject_masks`, `infer_unet_eye_masks`, `subject_segmentation`, `detect_yolo`, `detect_keypoints_yolo`, `detect_traditional`, `refined_subject_mask_review`, `materialize_refined_eye_masks_compat`, `run_sam_subject_masks`, `merge_subject_mask_runs`, `refined_detect_curation`.

Non-callers (skip the helper despite producing run-level groups): `compute_speed`, `import_stimulus_to_zarr`, `chaser_state_interpolator`, `detection_profile`, `keypoint_profile`, `eye_mask_profile`, training writers (`train_pose`, `train_detection`, `train_unet_subject_masks`), legacy `src/` tools, deferred `refine_online_detect` (per `provenance_todo.md:64`).

---

## 7. Open questions (consolidated)

These need your input before they can be classified.

**Pipeline & registry**
- Should `track` (pipeline) align with `tracks` (registry)? (Likely yes; no contract states this.)
- Should `Pipeline.STAGE_DEPENDENCIES['refined_subject_masks']` be `['subject_masks']`?
- Are derived analysis runs intentionally outside `recording_step_status` coverage, or a registry gap?
- `subject_mask_tuning` registered but not pivoted — partial migration or oversight?
- `recording_status_page_design.md` "BG Full / BG DS" — planned future split or stale?
- Did `fisheye.refinement.detect_quality` and `fisheye.refinement.refine_detect` survive recent refactors as cited?
- Should `cluster_batching_guide.md` be updated to point at unified subject-mask batching?

**Derived analysis**
- `tail_kinematics_runs` `row_axis="roi_rows"`: rename to `refined_subject_mask_rows` or extend recommended list?
- `method_version`: int or string (current writers split)?
- `derived_metrics_schema` rollout to analysis runs: deferred-by-design or gap?
- `palette.swim_bout_runs` schema_id: rename to `analysis.*` or keep prefix?
- Is `track_kinematics` grandfathered out of the contract or in violation?
- Should `swim_bout_statistics` be removed before adding contract metadata?

**Detection / refinement / crop**
- Does `write_curated_refined_detect_surfaces` actually emit `reason_bytes` with the encoding attrs Crimson requires?
- Does the dense root projection still get written for new runs or only legacy?
- Is `intended_use ∈ {training, full_recording}` enforced in `accept_detect_review.py` strict mode?
- Detection profile writer: required-attr coverage per contract?
- `validate_detect_training_zarr.py` invariant coverage?

**Eye / pose**
- Does every eye-mask producer emit the full attr set required by `eye_mask_training_artifact_contract.md`?
- Does `utils/eye_mask_profile.py` emit all required `profile_summary` keys?
- Refined eye-mask producer attr names: `source_eye_masks_run` (doc) vs `source_eye_run` (code) — which is canonical?
- `pose_schema_heuristics_split_proposal.md`: shipped or proposal? (Verify `configs/fisheye/pose_heuristics/`.)
- Should the `dependent_keypoints` mismatch warning be enforced in code?
- Is `POSE_SCHEMA_GUIDE.md` the canonical guide for downstream tools (Crimson)?
- Do all five `ROW_LINEAGE_ARRAYS` flow through every refined-eye writer?

**Cross-cutting**
- Is `analysis/speed_runs/` intentional parallel surface or legacy?
- Promote `body_frame_estimator_version` / `coordinate_space` / `angle_convention` to required (writer fix) or make optional (doc fix)?
- Are root-level ENUM_*.md / CRITICAL_REIMPORT_NEEDED.md living references or one-time logs?
- Is `latest` pointer transactional or explicitly best-effort?
- `provenance_todo.md` excludes `stimulus`/`speed`/`profile` writers — intentional scope or oversight?
- Should `derived_metrics_schema_contract` widen to claim the eye_angle-shape payloads?

---

## 8. Patterns & themes

These cut across all five sections:

**A. Stage-name vocabularies are converging through the canonical catalog.** Pipeline class and interactive launcher still keep runtime command names for compatibility, but they now expose canonical IDs. Registry cascade/maintenance derive from the same catalog. Remaining work is to keep new writer/status code using these translations instead of reintroducing hand-written DAGs.

**B. Required-attr drift is concentrated in the cheaply-built writers.** `subject_shape_runs` and `tail_kinematics_runs` are nearly perfect. `import_stimulus_to_zarr`, `compute_speed`, profile writers, and `chaser_state_interpolator` are nearly empty. The contract describes the discipline of the best writers. The worst writers are the ones that quietly accreted around the edges of the contract.

**C. `schema_version` is mostly informational.** Several schemas are versioned by writers, and a few readers/export utilities now inspect versions. Most schemas still do not have reader gates, so the remaining question is which schemas need hard compatibility checks versus metadata-only provenance.

**D. The registry has presence-level derived-analysis visibility, but not full freshness.** `track_kinematics`, `swim_bouts`, `bout_kinematics`, `eye_angles`, `subject_shape`, and `stimulus_response` now appear in `recording_step_status` and the wide status view. The remaining gap is semantic staleness: comparing each derived run's source refs against current upstream run IDs/revisions.

**E. Body-frame contract adoption is uneven.** Subject-shape remains the strongest canonical writer, and eye-angle now writes canonical body-frame attrs through the shared helper. Tail_kinematics still uses non-canonical attr names. The contract was written assuming uniform writer discipline; the actual pattern is improving but not uniform.

**F. Several docs describe state that's already shipped, with unchecked checkboxes.** `crop_storage_mode_migration_todo.md` Phase 5 and `repo_wide_staleness_workflow_edge_checklist.md` `crop → subject_masks` and `detect → subject_masks` are all done in code but unchecked in docs. The opposite (doc says done, code doesn't) is rarer — usually the doc lags.

**G. Root-level scratch describes one-time migrations.** ENUM_*.md cluster + CRITICAL_REIMPORT_NEEDED.md describe migrations whose targets are now the live state of the code. Without a closeout marker, future-you can't tell whether they're load-bearing reference or expired log.

**H. `latest` pointer atomicity is uneven.** Every `*_runs/` parent has a latest pointer, but only some newer writers stage run state carefully before promoting it. There is still no shared helper or repo-wide rule that prevents a partial run from becoming latest.

---

## What this inventory is NOT

- It is **not** a fix list. Resolution direction is a hint, not a plan.
- It is **not** a contract. It snapshots state on 2026-05-01 and goes stale immediately.
- It is **not** exhaustive. Six items are explicitly listed as "needs deeper writer audit" (open questions).
- It does **not** prescribe whether to update doc, code, or both. That's the strategy doc.

The strategy doc is the next step. It should answer:
1. Which drift items are worth fixing vs accepting?
2. Which docs are worth keeping vs archiving?
3. What forcing functions prevent re-drift?
4. What single source of truth replaces the five stage-name vocabularies?

---

*Generated 2026-05-01 by parallel audit across five focus areas. ~120 drift items. ~33 open questions.*
