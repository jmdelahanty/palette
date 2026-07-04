<!-- ARCHIVED 2026-07-04: dated point-in-time diagnostic snapshot, retained for history only. -->

# Keypoints/Masks Docs Audit (2026-05-20)

## Summary

Audited 73 docs in the keypoints/pose/eye/subject mask/segmentation/swim
bladder/tracking shard against code under `src/fisheye/{pose,segmentation,refinement,inference,tracking,training}/`
and `src/fisheye/analysis/`. The shard is largely current. Most contracts mark
`last_verified` between 2026-02-27 and 2026-05-01, and code spot-checks match
the claims (e.g. `heading_finite`/`heading_usable` writes in
`refine_keypoints.py:1373`, `UNetSmall(base_channels=32)` in
`segmentation/train_unet_eye_masks.py:860`, `EYE_ANGLE_METHOD_VERSION =
"eye_angle_analysis.v5"` and `MAJOR_AXIS_MARGINAL_DOT_THRESHOLD = 0.1` in
`analysis/eye_angle_analysis.py:94,100`, `SUBJECT_SHAPE_METHOD =
"subject_shape_from_refined_masks_v8"` in `analysis/subject_shape_runs.py:65`,
`single_subject_per_arena` tracking writer in
`tracking/single_subject_per_arena.py:60`).

Almost no docs are flatly stale. The dominant pattern is "STALE-EDIT light": a
handful of TODO/checklist docs have items now done that should be ticked off or
moved to ARCHIVE. One TODO (`subject_mask_component_provenance_followthrough_checklist`)
is fully done and qualifies as CHECKLIST-COMPLETE under
`docs/legacy_archive_migration_policy.md`. Two parallel-agent contracts
(`pose_detect_parity_parallel_agents_contract`, `eye_mask_parity_parallel_agents_contract`)
describe completed work waves and should archive.

Notable overlap exists in the subject-mask family (5 docs covering largely
overlapping unification/refinement TODOs), the tracking ID family (3 docs), and
the swim-bladder family (5 docs).

## Findings

### body_frame_contract.md
**Classification:** CURRENT
**Evidence:** Body frame writers at `src/fisheye/analysis/subject_shape_runs.py` (`BODY_FRAME_SCHEMA_VERSION` referenced at line 659, 1055); `body_frame/` group described matches arrays under `analysis/subject_shape_runs/<run>/components/subject_body/`.
**Action:** None.

### body_spline_tail_anchor_design.md
**Classification:** CURRENT
**Evidence:** `analysis/subject_shape_runs.py` materializes `tail_tip_xy`, `tail_base_xy`, `bspline_*` arrays as described; `caudal_swim_bladder_contour_point_xy` matches landmark conventions doc.
**Action:** None.

### eye_angle_compact_v2_design.md
**Classification:** CURRENT
**Evidence:** `analysis/eye_angle_analysis.py:97` defines `EYE_ANGLE_LAYOUT_COMPACT_DENSE_V2 = "compact_dense_v2"`; resolver in `analysis/eye_angle_io.py`; `EYE_ANGLE_OUTPUT_SCHEMA_VERSION` at line 510; canary references dated 2026-05-11 align with doc's "default switched on 2026-05-11".
**Action:** None.

### eye_angle_legacy_vergence_gaze_todo.md
**Classification:** CURRENT (active TODO, draft 2026-05-01)
**Evidence:** Implementation tasks all `[ ]`; `vergence_gaze_deg` semantics described match the v5 output schema in `analysis/eye_angle_analysis.py`.
**Action:** None.

### eye_angle_variants.md
**Classification:** CURRENT
**Evidence:** Schema context (v5/v7) matches `eye_angle_analysis.py:94` and variant schema version constants.
**Action:** None.

### eye_axis_half_plane_margin.md
**Classification:** CURRENT
**Evidence:** `MAJOR_AXIS_MARGINAL_DOT_THRESHOLD = 0.1` at `analysis/eye_angle_analysis.py:100`; `schema_version = 5` and method `eye_angle_analysis.v5` match.
**Action:** None.

### eye_mask_data_profile_schema_contract.md
**Classification:** CURRENT
**Evidence:** `sync_eye_mask_profile_registry` and `backfill_eye_mask_profiles` exist under `src/fisheye/utils/`; `aggregate_eye_mask_training_data_card` referenced from runbook lines up.
**Action:** None.

### eye_mask_parity_parallel_agents_contract.md
**Classification:** ARCHIVE
**Evidence:** Contract describes a parallel-agent execution plan (waves 0-3) for a one-time delivery (EM-A..F). All target modules exist (`utils/sync_eye_mask_profile_registry.py`, `utils/aggregate_eye_mask_training_data_card.py`, etc.). The work is delivered.
**Action:** Move to `docs/archive/` per legacy policy; the contract is no longer load-bearing.

### eye_mask_profile_registry_ops_runbook.md
**Classification:** CURRENT
**Evidence:** Commands resolve to existing modules (`backfill_eye_mask_profiles`, `sync_eye_mask_profile_registry`, `aggregate_eye_mask_training_data_card`).
**Action:** None.

### eye_mask_row_mapping_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-02-27`; stage-array shape claims match `src/fisheye/shared/zarr/stage_arrays.py` (referenced by other docs).
**Action:** None.

### eye_mask_training_artifact_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-22`; `utils/export_eye_mask_training_zarr.py` and `utils/validate_eye_mask_training_zarr.py` exist; eye-stage `auto` selection logic described matches `shared/eye_geometry_source` referenced in mask_review_save_approval_policy.
**Action:** None.

### eye_mask_training_data_card_contract.md
**Classification:** CURRENT
**Evidence:** Schema name/version contract aligns with `aggregate_eye_mask_training_data_card.py`.
**Action:** None.

### eye_mask_training_workflow.md
**Classification:** CURRENT
**Evidence:** Commands invoke existing modules `utils.run_eye_mask_training_pipeline`, `segmentation.train_unet_eye_masks`; `--eye-stage auto` flow consistent with eye_mask_training_artifact_contract.
**Action:** None.

### eye_mask_unet_design_recommendations.md
**Classification:** CURRENT
**Evidence:** `segmentation/unet.py:83` defines `UNetSmall` with `base_channels: int = 32`; `train_unet_eye_masks.py:860` instantiates with `base_channels=32`; `label_mode` handling at lines 90, 859 matches.
**Action:** None.

### eye_subject_mask_unification_design.md
**Classification:** CURRENT
**Evidence:** Doc describes `finalize_subject_masks.py` as the first assignment helper; that file exists at `src/fisheye/refinement/finalize_subject_masks.py`. Canary statements aligned with current refinement/finalize_subject_masks.py and assemble_refined_subject_masks.py.
**Action:** None.

### keypoint_auto_approval_todo.md
**Classification:** STALE-EDIT
**Evidence:** Phase 1 + registry filters checklist boxes are `[x]`; Phase 2 (drift) still `[ ]`. Implementation in `utils/auto_keypoint_review.py` matches Phase 1 description.
**Action:** Either re-classify as design note ("Phase 2 design") or leave as active TODO. Header still says "TODO" though most is shipped; trim Phase 1 details to a brief retrospective and refocus on the Phase 2 work.

### keypoint_data_profile_schema_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-10`; `sync_keypoint_profile_registry`, `backfill_keypoint_profiles`, `backfill_keypoint_derived_metrics_schema` all exist.
**Action:** None.

### keypoint_derived_metric_schema_contract.md
**Classification:** CURRENT
**Evidence:** `utils/backfill_keypoint_derived_metrics.py` exists; schema config path described as `configs/fisheye/keypoint_metric_schemas/*.json` consistent with the multi-skeleton doc.
**Action:** None.

### keypoint_heading_computation_contract.md
**Classification:** CURRENT
**Evidence:** `pose/heading.py` is the runtime evaluator (per keypoint_pose_rollout_status); `utils/backfill_keypoint_heading_computation.py` exists; precedence rules consistent with code.
**Action:** None.

### keypoint_heading_validity_todo.md
**Classification:** STALE-EDIT (Phase 1+2 complete; Phase 3+4 open)
**Evidence:** Doc states "Phase 1 complete" and confirms code references; verified `heading_finite_dst`/`heading_usable_dst` writes at `src/fisheye/refinement/refine_keypoints.py:1373-1382`. Phase 3 (one-time backfill on production archives) and Phase 4 (doc cleanup) still `[ ]`.
**Action:** When Phase 3 is executed against production, tick boxes and archive (CHECKLIST-COMPLETE candidate at that point). For now, keep but trim Phase 1/2 detail into a brief "done" summary.

### keypoint_late_correction_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-24`; `utils/resolve_eye_mask_stale.py` and `tune/keypoint_review.py --manual --frames` referenced commands match.
**Action:** None.

### keypoint_merged_row_gate_contract.md
**Classification:** CURRENT
**Evidence:** Policy semantics match keypoint_row_gating_workflow.md and `utils/export_keypoint_training_zarr.py` (referenced indirectly).
**Action:** None.

### keypoint_multi_skeleton_todo.md
**Classification:** STALE-EDIT (live TODO; many items completed)
**Evidence:** Phase 0/1/2/3 mostly `[x]`; Phase 4/5 mixed. Code claims verified: `traditional_v2.json` config and `extend_keypoint_skeleton.py` exist.
**Action:** Keep active. When remaining "[ ]" items close, archive.

### keypoint_multi_skeleton_training_selection_todo.md
**Classification:** STALE-EDIT (most patches now implemented)
**Evidence:** "Status: implemented" appears under each Required Patch. Only tie-break fix is documented as still pending; covered by `keypoint_training_refined_run_tie_fix_todo.md`.
**Action:** Either archive or shrink to a "Status" stub pointing at `keypoint_training_refined_run_tie_fix_todo.md` and `keypoint_multi_skeleton_todo.md`. There is significant overlap with both.

### keypoint_pose_rollout_status.md
**Classification:** CURRENT
**Evidence:** Status addendum 2026-05-17; verified `pose/heading.py`, `pose/heuristics.py`, `pose/schema.py`, packaged configs at `configs/fisheye/pose_heuristics/traditional_pose/{traditional_v1,traditional_v2}.json` (referenced by multiple docs and consistent with `audit_keypoint_skeleton_attrs.py`, `backfill_keypoint_skeleton_attrs.py` in utils listing).
**Action:** None.

### keypoint_quality_registry_workflow.md
**Classification:** CURRENT
**Evidence:** `utils/backfill_keypoint_heading_fields.py`, `utils/review_keypoints_batch.py`, `utils/export_keypoint_quality_overview.py`, `utils/finalize_keypoint_refinement_artifacts.py` referenced; all exist under `src/fisheye/utils/`.
**Action:** None.

### keypoint_refined_coordinate_space_incident_2026-03-04.md
**Classification:** CURRENT (incident record)
**Evidence:** Dated; refers to one-time recovery; follow-ups list invariant checks still pending.
**Action:** None. Incident notes should not be archived.

### keypoint_retune_notes.md
**Classification:** CURRENT
**Evidence:** `src/fisheye/tune/keypoint_tuner.py` exists; `refine_keypoints.py` writes `retune_id` (consistent with claims; not separately spot-checked).
**Action:** None.

### keypoint_review_policy.md
**Classification:** CURRENT
**Evidence:** Reason tags and review status payload shape are consistent with `keypoint_late_correction_contract.md` and `tune/keypoint_review.py`.
**Action:** None.

### keypoint_review_status_notes.md
**Classification:** CURRENT
**Evidence:** Operational note explaining Zarr v3 attr readback; referenced files `tune/keypoint_failure_review.py`, `utils/set_keypoint_review_status.py`, `utils/show_keypoint_review_status.py`, `utils/check_recording_steps.py` exist.
**Action:** None.

### keypoint_row_gating_workflow.md
**Classification:** CURRENT
**Evidence:** Policies described match `keypoint_merged_row_gate_contract.md`; commands invoke existing modules.
**Action:** None.

### keypoint_temporal_heading_heuristic_todo.md
**Classification:** CURRENT (active design TODO)
**Evidence:** `last_updated: 2026-04-06`; quality registry workflow references temporal-heading status now, so partial Phase 1 may already be present (e.g. `heading_temporal_outlier` queue mentioned in `keypoint_quality_registry_workflow.md`). The design doc accurately frames remaining policy/threshold work.
**Action:** None (verify in code if Phase 1 fully landed; if so, downgrade scope).

### keypoint_training_data_card_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-02-27`; pose-schema-derived skeleton metrics aligned with `aggregate_keypoint_training_data_card.py`.
**Action:** None.

### keypoint_training_refined_run_tie_fix_todo.md
**Classification:** CURRENT (active TODO)
**Evidence:** `utils/repair_keypoint_training_refined_run_ties.py` exists; documented as workaround; Patch 1/2 not yet applied per doc.
**Action:** None.

### keypoint_training_workflow.md
**Classification:** CURRENT
**Evidence:** Commands resolve to existing modules `utils.prepare_keypoint_training_from_registry`, `utils.run_keypoint_training_pipeline`, `utils.repair_keypoint_training_refined_run_ties`, `utils.validate_keypoint_training_zarr`.
**Action:** None.

### keypoints_pipeline_inline_registry_report.md
**Classification:** STALE-EDIT
**Evidence:** Table reports several stages as "No" inline writes (Keypoints traditional/YOLO, Eye masks YOLO/UNet). `keypoint_pose_rollout_status.md` and `eye_mask_parity_parallel_agents_contract.md` suggest much of this gap has been closed via batch entrypoints and pipeline integration. Cannot verify every row without deeper code inspection, but the audit shape is older than the parity contracts.
**Action:** Re-audit and refresh table, or mark this report as historic (it is dated implicitly; no `last_verified`).

### mask_review_save_approval_policy.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-02`; refined_subject_mask_review.py and eye_mask_review.py routing described matches segmentation_stage_split_review.md and current code listing.
**Action:** None.

### mask_rle_storage_design_and_benchmark_plan.md
**Classification:** CURRENT (design + benchmark plan)
**Evidence:** Code references to `mask_source.py`, `segmentation/*`, `refinement/*`, `training/zarr_*` modules all present; no claim of completion that needs verification.
**Action:** None.

### paintera_palette_subject_mask_workflow.md
**Classification:** CURRENT
**Evidence:** `scripts/sync_refined_subject_mask_metadata` and `utils/sync_refined_subject_mask_metadata.py` referenced; `utils/run_sam_subject_masks.py` exists. Canary archive details preserved.
**Action:** None.

### pose_detect_parity_parallel_agents_contract.md
**Classification:** ARCHIVE
**Evidence:** Same pattern as eye_mask_parity_parallel_agents_contract — defines P2-A/B/VAL-E2E parallel work that has been integrated (e.g. `run_keypoints_batch.py` and `run_recording_analysis_pipeline.py` both exist and are described in surrounding docs as the steady-state surfaces).
**Action:** Move to `docs/archive/`.

### pose_heuristic_profile_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-17`; packaged JSON files described exist at `configs/fisheye/pose_heuristics/traditional_pose/{v1,v2}.json`; runtime loader `pose/heuristics.py` exists.
**Action:** None.

### pose_kinematics_run_design.md
**Classification:** CURRENT (forward-looking design)
**Evidence:** No `pose_kinematics_runs` writer exists under `src/fisheye/analysis/` yet (verified by grep). Doc explicitly frames it as future architecture.
**Action:** None.

### pose_schema_heuristics_split_proposal.md
**Classification:** CURRENT (design proposal, partially realized)
**Evidence:** Packaged profiles exist; the proposal's "current state" section accurately notes that `detect_keypoints_traditional`/`keypoint_tuner` load packaged profiles.
**Action:** None.

### sam3_colleague_handoff.md
**Classification:** CURRENT
**Evidence:** `utils/run_sam_subject_masks.py`, `utils/run_sam_subject_masks_batch.py`, `visualization/visualize_sam_subject_prompts.py` all exist.
**Action:** None.

### sam3_subject_mask_canary_plan.md
**Classification:** STALE-EDIT
**Evidence:** Phase 1/2/3/4 checkboxes all still `[ ]` despite the workflow being shipped (per `paintera_palette_subject_mask_workflow.md` canary results dated 2026-04-05+, and `utils/run_sam_subject_masks.py` containing the documented behaviors). Phases 1-3 are effectively complete.
**Action:** Tick the completed boxes (Phase 1-3) and trim the doc to focus on Phase 4 (QC) and Phase 5 (training use) which remain open.

### segmentation_pipeline_step_todo.md
**Classification:** CURRENT (active TODO)
**Evidence:** `last_updated: 2026-04-04`; describes future broad segmentation step. Implementation note for `infer_unet_subject_masks.py` and `train_unet_subject_masks.py` verified in code listing.
**Action:** None.

### segmentation_stage_split_review.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-22`; verified all listed writers/entrypoints exist (`eye_segmentation.py`, `eye_segmentation_yolo.py`, `infer_unet_eye_masks.py`, `subject_segmentation.py`, `swim_bladder_segmentation.py`, `run_sam_subject_masks.py`, `infer_unet_subject_masks.py`, `backfill_subject_mask_runs.py`, `refine_eye_masks.py`, `refined_subject_mask_review.py`).
**Action:** None.

### single_subject_per_arena_tracking_contract.md
**Classification:** CURRENT
**Evidence:** Producer in `src/fisheye/tracking/single_subject_per_arena.py:192` ("Write a tracking_runs entry"); attrs `num_tracks`, `tracking_qc_state` written at lines 99, 300; matches doc's required arrays/attrs.
**Action:** None.

### subject_body_mask_qc_design.md
**Classification:** STALE-EDIT (initial implementation done; one item open)
**Evidence:** Most checklist items `[x]`; "Surface `requires_review` and reason tags in mask review and overlay tooling" still `[ ]`. `refinement/subject_body_mask_qc.py` exists.
**Action:** Close item or keep as a thin pending-followup note.

### subject_mask_component_provenance_followthrough_checklist.md
**Classification:** CHECKLIST-COMPLETE
**Evidence:** Every checklist item under sections 1-7 is `[x]`. Done definition met.
**Action:** Archive per `docs/legacy_archive_migration_policy.md`.

### subject_mask_keypoint_coverage_runbook.md
**Classification:** CURRENT
**Evidence:** `diagnostics/check_subject_mask_keypoint_coverage` referenced; consistent with `keypoint_review_policy.md`.
**Action:** None.

### subject_mask_refinement_todo.md
**Classification:** STALE-EDIT (largely done; track remainder)
**Evidence:** "Current State" lists most pieces implemented (`refined_subject_mask_review.py`, `refine_subject_masks.py`, `assemble_refined_subject_masks.py`, `finalize_subject_masks.py`, registry tables, component provenance). Document is large (50KB) and likely accumulates done items.
**Action:** Trim "Current State" into a short summary; restrict open work to a focused TODO list. Significant overlap with `subject_mask_stage_unification_todo.md` and `subject_mask_component_provenance_followthrough_checklist.md`.

### subject_mask_registry_contract.md
**Classification:** CURRENT
**Evidence:** Draft v1, `last_verified: 2026-03-10`. Tables `subject_mask_performance` / `subject_mask_component_quality` exist as described (referenced by other docs).
**Action:** None.

### subject_mask_runs_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-28`; schema `subject_v1_union`/`subject_v1_lr` and `available_channels` semantics align with code (`export_subject_mask_training_zarr.py` per mask_rle doc).
**Action:** None.

### subject_mask_stage_unification_todo.md
**Classification:** STALE-EDIT (mostly done)
**Evidence:** All Immediate TODO items 1-6 are `[x]`. Open Questions remain.
**Action:** Move to CHECKLIST-COMPLETE candidate or merge remaining Open Questions into `subject_mask_refinement_todo.md` and archive.

### subject_mask_training_artifact_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-26`; `subject_v1_union` path described as implemented; `training/zarr_subject_mask_dataset.py` and `training/train_unet_subject_masks` (per segmentation_pipeline_step_todo) exist.
**Action:** None.

### subject_mask_tuning_workflow.md
**Classification:** CURRENT
**Evidence:** All referenced CLIs exist (`tune.subject_mask_tuner`, `utils.apply_tuning_by_camera`, `utils.audit_swim_bladder_tuning_metadata`, `segmentation.subject_segmentation`, `tune.refined_subject_mask_review`, `visualization.subject_mask_inspector`, `refinement.refine_subject_masks`, `tune.swim_bladder_mask_tuner`, `segmentation.swim_bladder_segmentation`, `refinement.assemble_refined_subject_masks`).
**Action:** None.

### subject_shape_landmark_conventions.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-05-01`; schema v3 / `subject_shape_from_refined_masks_v8` matches `analysis/subject_shape_runs.py:65`.
**Action:** None.

### subject_shape_runs_contract.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-28`; `analysis/subject_shape_runs.py` writer present; references body_frame_contract.md and refined_subject_masks contract consistently.
**Action:** None.

### subject_shape_snout_centerline_workflow.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-05-01`; method version 8 and `SUBJECT_SHAPE_METHOD = "subject_shape_from_refined_masks_v8"` verified at `analysis/subject_shape_runs.py:65`.
**Action:** None.

### swim_bladder_patch_review_design.md
**Classification:** CURRENT
**Evidence:** `tune/swim_bladder_mask_tuner.py`, `segmentation/swim_bladder_segmentation.py` listed in code.
**Action:** None.

### swim_bladder_polar_boundary_design.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-01`; tuner has `--method-family polar_boundary` per workflow doc; materializer dispatches by `subject_method_family` per code listing.
**Action:** None.

### swim_bladder_review_policy.md
**Classification:** CURRENT
**Evidence:** `last_verified: 2026-04-02`; consistent with mask_review_save_approval_policy.md.
**Action:** None.

### swim_bladder_tuning_metadata_audit.md
**Classification:** CURRENT
**Evidence:** `utils/audit_swim_bladder_tuning_metadata.py` and `utils/apply_tuning_by_camera.py` exist.
**Action:** None.

### swim_bout_exponential_segmentation.md
**Classification:** CURRENT
**Evidence:** Consistent with `analysis/detect_bouts_multi_level.py` (existence verified); cross-references swim_bout_peak_event_detector_design.
**Action:** None. (Borderline shard scope — analysis-adjacent; shard rule includes `swim_bout_*`.)

### swim_bout_peak_event_detector_design.md
**Classification:** CURRENT
**Evidence:** Canary findings dated 2026-04-27; current-feeding-canary runs named (compact_v2_fresh_20260509) align with swim_bout_runs_v2_compact_layout.md.
**Action:** None.

### swim_bout_runs_v2_compact_layout.md
**Classification:** CURRENT
**Evidence:** Implementation notes through 2026-05-11 default-switch; `analysis/swim_bout_io.py` referenced. Cannot verify file directly but consistent with track_kinematics_bout_status.md.
**Action:** None.

### track_assignment_id_status.md
**Classification:** STALE-EDIT
**Evidence:** Doc opens with explicit historical note: "captures the pre-tracking_runs integration state. Current implemented contract is in `tracking_runs_contract_status.md`". This is a deliberate historical record but is misleading without the disclaimer being more prominent.
**Action:** Either archive (it is self-marked as historical) or rename to `track_assignment_id_history.md` and add a banner pointing at `tracking_runs_contract_status.md`.

### track_identity_target_architecture.md
**Classification:** CURRENT (forward-looking)
**Evidence:** Target architecture proposal; the `single_subject_per_arena` strategy is now implemented; `multi_subject_within_arena` remains future. Doc explicitly proposes future state.
**Action:** None.

### track_kinematics_bout_status.md
**Classification:** CURRENT
**Evidence:** Last reviewed 2026-04-27; verified `analysis/track_kinematics.py`, `analysis/compute_speed.py`, `analysis/detect_bouts_multi_level.py` exist.
**Action:** None.

### track_validity_timeline_design.md
**Classification:** CURRENT
**Evidence:** Last reviewed 2026-04-26; Implementation Status section accurately describes shipped vs pending arrays.
**Action:** None.

### tracking_runs_contract_status.md
**Classification:** CURRENT
**Evidence:** Producer claims verified at `src/fisheye/tracking/single_subject_per_arena.py:192,300`; `derive_tracking_qc_state` at line 60; `num_tracks` at line 300.
**Action:** None.

### tracking_unassigned_row_policy.md
**Classification:** CURRENT
**Evidence:** Consistent with tracking_runs_contract_status.md; `track_kinematics --include-unassigned` flag matches doc.
**Action:** None.

### traditional_subject_segmentation_scaling_todo.md
**Classification:** CURRENT (deferred TODO)
**Evidence:** Explicitly marked "Status: Deferred"; all checklist items `[ ]`. Current single-process implementation in `segmentation/subject_segmentation.py` confirmed.
**Action:** None.

### traditional_v2_keypoint_migration_design.md
**Classification:** CURRENT
**Evidence:** `configs/fisheye/pose_schemas/traditional_v2.json` referenced; `utils/extend_keypoint_skeleton.py` and `utils/batch_extend_keypoint_skeleton.py` exist (per utils listing); status note dated 2026-05-17.
**Action:** None.

## Overlaps / Gaps

### Overlaps

1. **Subject-mask refinement/unification cluster (largest)** — `subject_mask_refinement_todo.md`, `subject_mask_stage_unification_todo.md`, `subject_mask_component_provenance_followthrough_checklist.md`, `eye_subject_mask_unification_design.md`, `segmentation_stage_split_review.md` cover overlapping scope. `subject_mask_component_provenance_followthrough_checklist.md` is done. `subject_mask_stage_unification_todo.md` items all `[x]`. Consider consolidating into one active TODO (`subject_mask_refinement_todo.md`) and archiving the rest.

2. **Tracking ID family** — `track_assignment_id_status.md` (self-marked historical), `tracking_runs_contract_status.md` (current implemented), `track_identity_target_architecture.md` (target). Three docs for three layers is defensible, but `track_assignment_id_status.md` is functionally archive material.

3. **Keypoint multi-skeleton family** — `keypoint_multi_skeleton_todo.md`, `keypoint_multi_skeleton_training_selection_todo.md`, `traditional_v2_keypoint_migration_design.md`, `keypoint_training_refined_run_tie_fix_todo.md`, `keypoint_pose_rollout_status.md`. Significant redundancy in "what is done / what remains" sections. Recommend pose_rollout_status as the single status doc and the others as focused TODOs.

4. **Swim-bladder** — `swim_bladder_patch_review_design.md`, `swim_bladder_polar_boundary_design.md`, `swim_bladder_review_policy.md`, `swim_bladder_tuning_metadata_audit.md`, plus implicit overlap with `subject_mask_tuning_workflow.md`. These describe distinct things (UI design, polar method, review policy, metadata audit, operator workflow) but cross-reference heavily.

5. **Parallel-agent contracts** — `eye_mask_parity_parallel_agents_contract.md` and `pose_detect_parity_parallel_agents_contract.md` are completed coordination docs; both should archive.

6. **Eye-angle docs** — `eye_angle_compact_v2_design.md`, `eye_angle_variants.md`, `eye_axis_half_plane_margin.md`, `eye_angle_legacy_vergence_gaze_todo.md` together with `src/fisheye/docs/eye_angle_conventions.md`. Mostly cleanly partitioned (storage layout, naming guide, geometric primer, deprecation plan), no archive candidates.

### Gaps

1. **No active doc for `pose_kinematics_runs` once it ships.** Currently only a design note exists; no contract or runbook. When implementation lands, a `pose_kinematics_runs_contract.md` will be needed.

2. **No runtime contract for `single_subject_per_arena_tracking` blocking-threshold behavior.** `tracking_runs_contract_status.md` mentions thresholds are "future policy metadata", but there is no doc that tells operators when these will become enforced.

3. **No active runbook for `auto_keypoint_review` operator usage.** `keypoint_auto_approval_todo.md` describes the design and Phase 1 is shipped (`utils/auto_keypoint_review.py`), but there is no operator-facing how-to (`docs/keypoint_auto_approval_runbook.md` would mirror `eye_mask_profile_registry_ops_runbook.md`).

4. **`keypoints_pipeline_inline_registry_report.md` lacks `last_verified` and is likely stale.** It is the only doc in the shard without a contract-meta or status header.

5. **`subject_mask_runs_contract.md` and `refined_subject_masks_runs_contract.md`** — the second is referenced by many docs in this shard but not in the shard's file list directly (it was not in the `ls` output as a top-level file matching the patterns). If it does not exist, that's a gap; if it does, it falls outside the agent's strict pattern filter.
