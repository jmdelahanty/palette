<!-- ARCHIVED 2026-07-04: dated point-in-time diagnostic snapshot, retained for history only. -->

# Analysis & Storage Docs Audit (2026-05-20)

## Summary

Shard scope: analysis pipeline, zarr/storage layout, stimulus/bout/kinematics
analytics, analytics query layer. 47 docs reviewed.

Headline:
- The compact-v2 rollout docs (swim_bout, bout_kinematics, stimulus_response,
  eye_angle) are well synced with code. Defaults verified in
  `src/fisheye/analysis/{detect_bouts_multi_level,bout_kinematics,stimulus_response,eye_angle_analysis}.py`.
- `analysis_post_detection_workflow_status.md` is the largest single drift:
  it lists `stimulus_response` as "design, not current workflow", yet
  `stimulus_response.py`, `stimulus_response_omr.py`,
  `stimulus_response_concentric_omr.py`, and `stimulus_response_io.py` are all
  implemented and the contract dates from 2026-05-09 forward have superseded it.
- `experiment_types_reference.md` still asserts the H5 protocol key-mismatch
  bug; the code already checks both keys
  (`src/fisheye/analysis/import_stimulus_to_zarr.py:756-759`).
- `analysis_dense_array_migration_todo.md` describes a deferred proposal whose
  trigger ("a second consumer needs dense data") has occurred but no migration
  has happened — still accurate as a deferred TODO.
- Two checklists are effectively complete: `analysis_zarr_creation_todo.md`
  has all Phase B steps done; Phase C/D items remain valid follow-ups.
- `stimulus_response_analysis_flow.md`, `stimulus_response_compact_v2_design.md`,
  `compact_v2_readiness_audit_2026-05-11.md` are all current.
- Several low-volatility primers (math primer, ellipse notes, raw_vs_smoothed,
  artifact_storage_map, dask_zarr_write_safety, sandbox_zarr_fallback,
  zarr_split_policy, zarr_storage_lifecycle_policy) are stable.

## Findings

### analysis_dense_array_migration_todo.md
**Classification:** CURRENT (deferred proposal)
**Evidence:** Doc still labels `track_kinematics.py` as sparse-per-track; that
matches `src/fisheye/analysis/track_kinematics.py` (tracks/id_<track> layout
confirmed in `analysis_writer_compact_layout_inventory.md:96`). No dense-array
migration in code.
**Action:** Keep. Update "Last reviewed" line; note that
`stimulus_response.expand_sparse_through_loader` now uses
`track_kinematics_io` (per inventory line 200), so the rationale stands.

### analysis_post_detection_workflow_status.md
**Classification:** STALE-EDIT
**Evidence:** Doc line 176 says "I do not see a corresponding implementation
module under src/fisheye/analysis" for stimulus_response. Code has
`stimulus_response.py`, `stimulus_response_omr.py`,
`stimulus_response_concentric_omr.py`, `stimulus_response_grating.py`,
`stimulus_response_io.py`. Also lines 213-215 cite stale protocol-import bug
in `experiment_types_reference.md` and `protocol_parameter_registry_todo.md`;
import code checks both keys at `import_stimulus_to_zarr.py:756-759`.
**Action:** Rewrite Executive Summary and "Multi-Stimulus Readiness" sections
to reflect implemented stimulus_response stack plus compact-v2 default, or
mark archived in favor of `current_pipeline_contract.md` +
`stimulus_response_implementation_plan.md`.

### analysis_to_training_promotion_contract.md
**Classification:** CURRENT
**Evidence:** Modules referenced exist:
`src/fisheye/tune/detect_training_promotion_backend.py`,
`src/fisheye/utils/promote_analysis_detect_to_training.py`,
`src/fisheye/tune/video_detect_review_web.py`. Doc explicitly dated 2026-05-20.
**Action:** None.

### analysis_writer_compact_layout_inventory.md
**Classification:** CURRENT
**Evidence:** Defaults verified —
`eye_angle_analysis.py:99` (EYE_ANGLE_LAYOUT_DEFAULT = compact_dense_v2),
`detect_bouts_multi_level.py:116` (SWIM_BOUT_LAYOUT_DEFAULT = compact_v2),
`stimulus_response.py:1556` (STIMULUS_RESPONSE_LAYOUT_DEFAULT = compact_v2),
`bout_kinematics.py:79` (BOUT_KINEMATICS_LAYOUT_DEFAULT = compact_tabular_v2).
**Action:** None.

### analysis_zarr_creation_contract.md
**Classification:** CURRENT
**Evidence:** `src/fisheye/analysis/create_analysis_zarr.py` exists and the
contract `last_verified: 2026-02-27` describes a still-implemented surface.
**Action:** Bump `last_verified` after next operator review.

### analysis_zarr_creation_todo.md
**Classification:** CHECKLIST-COMPLETE (mostly)
**Evidence:** Phase A and B fully marked done; Phase C task "Update
import_recordings_analysis to call create_analysis_zarr first" remains open
in doc and is corroborated by separate
`import_recordings_analysis.py` orchestrator. Phase D detect_yolo boundary
items still open.
**Action:** Close Phases A-B explicitly; carry C/D into a leaner follow-up
or merge into `recording_analysis_pipeline_contract.md`.

### analysis_zarr_object_count_schema_direction.md
**Classification:** CURRENT
**Evidence:** Audit snapshot dated 2026-05-08 with concrete counts;
`audit_zarr_group_counts.py` exists at `src/fisheye/utils/`. Family direction
matches inventory and code defaults.
**Action:** None.

### analytics_math_primer.md
**Classification:** CURRENT
**Evidence:** Output schema version 7 referenced in doc (line 267) matches
`eye_angle_analysis.py:90` (EYE_ANGLE_OUTPUT_SCHEMA_VERSION = 7). Movement
grouped layout matches code. `last_verified: 2026-04-28`.
**Action:** None.

### analytics_query_layer_design.md
**Classification:** CURRENT
**Evidence:** References `protocol_signature_hash` and DuckDB/Parquet model;
exporter `src/fisheye/utils/export_cross_recording_analytics.py` exists.
**Action:** None.

### artifact_storage_map.md
**Classification:** CURRENT
**Evidence:** `fisheye.shared.plot_artifacts` exists; aggregator utilities
named (`aggregate_detection_training_data_card`, etc.) all present in
`src/fisheye/utils/`.
**Action:** None.

### bout_classification_runs_contract.md
**Classification:** CURRENT
**Evidence:** `src/fisheye/analysis/bout_classification_runs.py` exposes
`resolve_bout_classification_run`, `validate_bout_classification_run`,
`summarize_bout_classification_run` matching contract. Producer
`megabouts_classifier.py` present.
**Action:** None.

### bout_kinematics_compact_v2_layout.md
**Classification:** CURRENT
**Evidence:** Default writer set per `bout_kinematics.py:79`; doc dated
2026-05-10 with default-update note for 2026-05-11.
**Action:** None.

### bout_kinematics_run_design.md
**Classification:** CURRENT
**Evidence:** Per-bout heading + movement metrics in `bout_kinematics.py`;
date anchor 2026-04-27.
**Action:** None.

### camera_metadata.md
**Classification:** CURRENT
**Evidence:** Conventions stable; consumed by training-manifest tooling.
**Action:** None.

### citrus_arena_topology_design.md
**Classification:** CURRENT (design, partly aspirational)
**Evidence:** Explicitly a design proposal; code paths it cites all exist.
**Action:** None.

### compact_v2_readiness_audit_2026-05-11.md
**Classification:** CURRENT
**Evidence:** All "Pass" rows independently verifiable in this audit.
**Action:** None.

### concentric_omr_stimulus_response_design.md
**Classification:** CURRENT
**Evidence:** `stimulus_response_concentric_omr.py` present at expected path;
doc marks v0 centering + first numeric radial OMR slice, matches code state.
**Action:** None.

### cross_recording_analytics_export_design.md
**Classification:** CURRENT
**Evidence:** `export_cross_recording_analytics.py`, `query_analytics_exports.py`,
`resolve_analytics_export.py`, `check_analytics_exports.py` exist under
`src/fisheye/utils/`. `last_updated: 2026-05-05`.
**Action:** None.

### current_pipeline_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-05-01`. Table rows match implemented
families (refined_subject_masks_runs, analysis/subject_shape_runs,
analysis/eye_angle_runs with schema v5/output v7 confirmed).
**Action:** None.

### dask_zarr_write_safety.md
**Classification:** CURRENT
**Evidence:** Architectural rule, stable.
**Action:** None.

### derived_analysis_run_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-05-01`. Required attrs match writers.
**Action:** None.

### derived_metrics_schema_contract.md
**Classification:** CURRENT
**Evidence:** Implementation target `refined_keypoints_runs.attrs["derived_metrics_schema"]`
is keypoints-domain; this doc remains the shared schema reference.
**Action:** None.

### ellipse_fitting_notes.md
**Classification:** CURRENT
**Evidence:** Conceptual; no code dependencies that drift.
**Action:** None.

### experiment_types_reference.md
**Classification:** STALE-EDIT
**Evidence:** Doc line 29: "Protocol extraction is blocked on a key-name
mismatch... See protocol_parameter_registry_todo.md." Code already supports
both keys at `src/fisheye/analysis/import_stimulus_to_zarr.py:756-759`.
**Action:** Replace the "blocked" sentence with a backlog-only note: the
importer accepts both keys; some recordings still lack imported protocol
because they were imported before the dual-key path.

### eye_angle_compact_v2_design.md
**Classification:** CURRENT
**Evidence:** `last_updated: 2026-05-11`. Schema v5/output v7 matches code
(`eye_angle_analysis.py:90, 510`). Default layout matches `:99`.
**Action:** None.

### grating_analysis_acquisition_questions.md
**Classification:** CURRENT
**Evidence:** Open-question checklist; acquisition-side outstanding items.
**Action:** None.

### moving_grating_downstream_prerequisites.md
**Classification:** CURRENT
**Evidence:** Names matching scripts/stages; runner
`scripts/run_moving_grating_downstream_pipeline.sh` referenced.
**Action:** None.

### omr_stimulus_response_design.md
**Classification:** CURRENT
**Evidence:** Implementation lives in `stimulus_response_omr.py`; matches
"first single-fish implementation" claim.
**Action:** None.

### organize_recordings_logging_schema.md
**Classification:** CURRENT
**Evidence:** Producer `src/fisheye/utils/organize_recordings.py` and log
event schema stable.
**Action:** None.

### palette_pipeline_overview.mmd
**Classification:** CURRENT
**Evidence:** Mermaid graph nodes match implemented run families.
**Action:** None.

### plot_visualization_artifact_contract.md
**Classification:** CURRENT
**Evidence:** Run families with `visualizations/<artifact>_png` confirmed in
inventory; `shared/plot_artifacts.py` present.
**Action:** None.

### raw_vs_smoothed_metrics_behavioral_geometry.md
**Classification:** CURRENT
**Evidence:** Architectural principle; not code-dependent.
**Action:** None.

### recording_analysis_pipeline_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-05-09`. Modules cited
(`import_recording_analysis`, `run_recording_analysis_pipeline`,
`fisheye.refinement.detect_quality`) all exist.
**Action:** None.

### recording_manifest_contract.md
**Classification:** CURRENT
**Evidence:** Schema contract; `last_verified: 2026-02-27`. Fields still
match registry normalization path.
**Action:** Bump `last_verified` next pass.

### sandbox_zarr_fallback.md
**Classification:** CURRENT
**Evidence:** Sandbox workaround; stable.
**Action:** None.

### stimulus_response_analysis_flow.md
**Classification:** CURRENT
**Evidence:** Compact-readiness audit (2026-05-11) line 53 explicitly notes
this doc was updated to remove "compact-v2 as planned" language.
**Action:** None.

### stimulus_response_compact_v2_design.md
**Classification:** CURRENT
**Evidence:** `last_updated: 2026-05-11`. Default verified at
`stimulus_response.py:1556`.
**Action:** None.

### stimulus_response_data_model.md
**Classification:** CURRENT
**Evidence:** Step-first layout consistent with stimulus_response writer;
backfill scripts referenced exist.
**Action:** None.

### stimulus_response_implementation_plan.md
**Classification:** CURRENT
**Evidence:** Pass table matches implemented modules
(`stimulus_response.py`, `stimulus_response_omr.py`,
`stimulus_response_concentric_omr.py`); `shared/zarr/analysis_stage_arrays.py`
present.
**Action:** None.

### stimulus_response_run_design.md
**Classification:** CURRENT
**Evidence:** Matches implementation across base/grating/OMR/concentric. The
archive ancestor base_analysis_moving_grating_design.md is properly noted as
superseded.
**Action:** None.

### swim_bout_exponential_segmentation.md
**Classification:** CURRENT
**Evidence:** Implementation in `detect_bouts_multi_level.py`; speed levels
documented match writer.
**Action:** None.

### swim_bout_peak_event_detector_design.md
**Classification:** CURRENT
**Evidence:** Implemented in `detect_bouts_multi_level.py`; status "first
implementation slice available" still matches deferred valley-split note.
**Action:** None.

### swim_bout_runs_v2_compact_layout.md
**Classification:** CURRENT
**Evidence:** Default switch confirmed (`detect_bouts_multi_level.py:116`).
Resolver `swim_bout_io.py` present.
**Action:** None.

### virtual_collection_manifest_schema.md
**Classification:** CURRENT (draft)
**Evidence:** `last_updated: 2026-05-07`. Design-only; matches analytics
export design.
**Action:** None.

### zarr_nfs_audit_todo.md
**Classification:** CURRENT
**Evidence:** Measurements dated 2026-04-03; behavioral context still valid.
**Action:** None.

### zarr_parquet_sidecar_exports_design.md
**Classification:** CURRENT (deferred design)
**Evidence:** Explicitly marked deferred; consistent with implemented
cross-recording Parquet export under `src/fisheye/utils/`.
**Action:** None.

### zarr_sharding_design_note.md
**Classification:** CURRENT
**Evidence:** Matches `create_palette_zarr` sharded raw_video behavior.
**Action:** None.

### zarr_spec_runtime_drift_todo.md
**Classification:** CURRENT
**Evidence:** Drift items 1-2 still verifiable in
`create_analysis_zarr.py` and `import_video_metadata.py`.
**Action:** None.

### zarr_split_policy.md
**Classification:** CURRENT
**Evidence:** Stable policy; matches `*_analysis.zarr` vs `*_training.zarr`
naming used throughout codebase.
**Action:** None.

### zarr_storage_lifecycle_policy.md
**Classification:** CURRENT
**Evidence:** Policy doc; thresholds referenced by object-count direction doc.
**Action:** None.

### zarr_transfer_benchmark_plan.md
**Classification:** CURRENT (plan)
**Evidence:** Benchmark plan not invalidated by code changes.
**Action:** None.

## Overlaps / Gaps

- `analysis_post_detection_workflow_status.md` overlaps with and now
  contradicts `current_pipeline_contract.md` and
  `stimulus_response_implementation_plan.md`. Recommend archiving or rewriting
  the status doc.
- The protocol-import "blocked" claim recurs in three docs
  (`experiment_types_reference.md`, `analysis_post_detection_workflow_status.md`,
  and per-line citation `provenance_backfill_todo.md`); a single corrective
  pass on the importer wording should patch all three (importer code at
  `import_stimulus_to_zarr.py:756-759` already handles both keys).
- `analysis_zarr_creation_contract.md` (Feb 2026) and
  `recording_analysis_pipeline_contract.md` (May 2026) both describe
  archive-creation orchestration. They are mostly complementary, but the
  creation contract has an open "Open decisions" section that
  `recording_analysis_pipeline_contract.md` has effectively decided. Worth a
  short close-out pass.
- No analysis-shard doc currently calls out the
  `track_kinematics_runs` compact-tabular migration as actively deferred —
  the inventory captures this at line 96/355, but other docs (math primer,
  bout_kinematics_run_design) still reference `tracks/id_<track>` without
  noting the future ragged layout. Low priority.
- `stimulus_response_analysis_flow.md` does not include the
  `concentric_grating/radial_omr/` flow that is documented in
  `concentric_omr_stimulus_response_design.md`. Minor gap.
