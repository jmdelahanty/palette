# Review/Training/Clipped Docs Audit (2026-05-20)

## Summary

Audited 28 review-UX, training-lifecycle, crop, clipped, mask-policy, and label-provenance docs against current code. The three docs corrected this morning (`detection_review_web_todo.md`, `analysis_to_training_promotion_contract.md`, `clipped_training_zarr_implementation_checklist.md`) are consistent with `src/fisheye/tune/video_detect_review_web.py:1240-1329` and `src/fisheye/tune/detect_training_promotion_backend.py`; no remaining contradictions for those three. Bulk of docs CURRENT. A few are CHECKLIST-COMPLETE candidates for archive. `crop_review_workflow.md`, `crop_distributed_tradeoffs.md`, `crop_persistence_tradeoff.md` are STALE-EDIT (predate multi-instance/geometry-only). No severe drift in this shard.

## Findings

### detection_review_web_todo.md
**Classification:** CURRENT (corrected today)
**Evidence:** `--edit --promote-training-zarr` matches `video_detect_review_web.py:1240,1275,1322`; proxy builder `fisheye.utils.build_review_proxy_videos` exists; MVP files all present.
**Action:** Keep as-is. Remaining open item (proxy frame-count validation, line 270) is legitimate TODO.

### analysis_to_training_promotion_contract.md
**Classification:** CURRENT (corrected today)
**Evidence:** Backend `src/fisheye/tune/detect_training_promotion_backend.py` and CLI `src/fisheye/utils/promote_analysis_detect_to_training.py` both resolve; both `promote_detection_frames` and `promote_clipped_detection_frames` exist; batch save grouping matches `video_detect_review_web.py:707` batch_save logging.
**Action:** None.

### clipped_training_zarr_implementation_checklist.md
**Classification:** CURRENT (corrected today)
**Evidence:** Detection-bbox promotion and finalized clip collections checked; keypoint/mask promotion deferred (line 401). Matches backend scope.
**Action:** None.

### crop_review_workflow.md
**Classification:** STALE-EDIT
**Evidence:** Line 6 points at nonexistent `docs/crop_review_registry_todo.md`. Otherwise referenced CLIs (`visualize_crops`, `review_crops`, `generate_review_list`, `review_keypoints_batch`) resolve. No mention of geometry-only viewer limits (vs `crop_reader_geometry_only_inventory_2026-05-16.md:69` classifying `visualize_crops.py` as deferred direct-roi-images reader).
**Action:** Fix or drop broken link; consider one line on geometry-only viewer behavior.

### crop_distributed_tradeoffs.md
**Classification:** STALE-EDIT
**Evidence:** Frames itself around "single fish / one detection per frame" base case. Multi-instance refined detect (`refined_detect_sparse_instances_schema.md`) and geometry-only crops (`crop_storage_mode_migration_todo.md` Phase 2) have superseded these tradeoffs. Recommended Options 1-4 were not followed.
**Action:** Archive or add redirect to `geometry_only_crop_workflow_cache_design.md`.

### crop_live_view_vs_materialized_stream_design.md
**Classification:** CURRENT
**Evidence:** Top-of-file update note (line 6, 2026-05-16) redirects to `geometry_only_crop_workflow_cache_design.md`. Benchmarks at 2026-04-04 / 2026-04-05 match `crop_storage_mode_migration_todo.md`.
**Action:** None.

### crop_persistence_tradeoff.md
**Classification:** STALE-EDIT (minor)
**Evidence:** "Future option: `crop_mode = persist | on_demand`" is implemented as `crop_storage_mode = materialized | geometry_only`. Cross-link to live-view doc present.
**Action:** Archive or fold into `crop_live_view_vs_materialized_stream_design.md`.

### crop_storage_mode_migration_todo.md
**Classification:** CURRENT
**Evidence:** Phase 0-2 checkboxes verified against `src/fisheye/shared/crop_image_source.py` and batch wrappers; flat ROI cache utilities exist (`build_flat_roi_cache.py`). Phase 3-7 genuinely open.
**Action:** None.

### crop_reader_geometry_only_inventory_2026-05-16.md
**Classification:** CURRENT
**Evidence:** Each listed path resolves. Dated inventory with bounded scope.
**Action:** None.

### detection_refinement_workflow.md
**Classification:** CURRENT
**Evidence:** All CLIs verified: `refine_detect`, `detect_quality`, `detect_review`, `accept_detect_review`, `review_detect_batch`, `registry.maintenance`, `migrate_legacy_detect_labels`, `detection_visualizer`, `diagnostics/check_crop_sources`, `inspect_refined_detect_run`, `arena_assignment`. Status vocabulary (`present/missing/filtered_out/ambiguous`) matches `mask_review_save_approval_policy.md`.
**Action:** None.

### detection_training_plan.md
**Classification:** CURRENT
**Evidence:** All 6 plan-of-record steps implemented (registry scanner, prepare, train+log, exports).
**Action:** None. (Note: detection-shard agent classified this STALE-EDIT for module-name drift; reconcile in master.)

### detection_training_zarr_edit_todo.md
**Classification:** CHECKLIST-COMPLETE for Phase 0-3
**Evidence:** All Phase 0-3 checkboxes done; backend/web/static/tests all exist. Phase 4 (parity, arena, legacy fallback) is real open work.
**Action:** Condense — move Phase 0-3 to completed section, keep Phase 4 as active.

### keypoint_review_policy.md
**Classification:** CURRENT
**Evidence:** Hotkeys (`c/x/d`), reason vocabulary, `--row-gate-policy raw_success_plus_box_only` all consistent with `keypoint_row_gating_workflow.md` and code.
**Action:** None.

### keypoint_review_status_notes.md
**Classification:** CURRENT
**Evidence:** Write-path files (`keypoint_failure_review.py`, `set_keypoint_review_status.py`) and read-path files (`show_keypoint_review_status.py`, `check_recording_steps.py`) all exist.
**Action:** None.

### keypoint_auto_approval_todo.md
**Classification:** CURRENT
**Evidence:** `fisheye.utils.auto_keypoint_review` exists; `refine_keypoints_batch --auto-review-full-recording` wired; registry filters for `method/policy_id/policy_version` implemented. Phase 1 done, Phase 2 open, Phase 3 deferred per doc.
**Action:** None.

### keypoint_row_gating_workflow.md
**Classification:** CURRENT
**Evidence:** Policy semantics match `training_quality_gate_contract.md:160-175`; `keypoint_box_only` array path matches export code.
**Action:** None.

### keypoint_training_workflow.md
**Classification:** CURRENT
**Evidence:** All modules resolve (`registry_query`, `prepare_keypoint_training_from_registry`, `run_keypoint_training_pipeline`, `export_keypoint_training_zarr`, `validate_keypoint_training_zarr`, `repair_keypoint_training_refined_run_ties`).
**Action:** None.

### keypoint_training_refined_run_tie_fix_todo.md
**Classification:** CURRENT
**Evidence:** Workaround (`repair_keypoint_training_refined_run_ties.py`) exists; Patch 1 (preflight tie-break) and Patch 2 (`extend_keypoint_skeleton.py` created_utc) still open.
**Action:** None.

### mask_review_save_approval_policy.md
**Classification:** CURRENT
**Evidence:** `refined_subject_mask_review.py`, `eye_mask_review.py` exist; `--manual` / `--legacy-manual` modes and aggregation policy match `subject_mask_tuning_workflow.md`.
**Action:** None.

### swim_bladder_review_policy.md
**Classification:** CURRENT
**Evidence:** Method families (`swim_bladder_patch_threshold_v1`, `swim_bladder_polar_boundary_v1`) match `swim_bladder_mask_tuner.py`. `refined_subject_mask_review.py` exists.
**Action:** None.

### subject_mask_tuning_workflow.md
**Classification:** CURRENT
**Evidence:** Twelve referenced CLIs all resolve.
**Action:** None.

### subject_mask_keypoint_coverage_runbook.md
**Classification:** CURRENT
**Evidence:** `fisheye.diagnostics.check_subject_mask_keypoint_coverage` exists.
**Action:** None.

### subject_mask_component_provenance_followthrough_checklist.md
**Classification:** CHECKLIST-COMPLETE
**Evidence:** Every section `[x]` complete; "Done Definition" satisfied.
**Action:** Archive (move to `docs/archive/`).

### subject_mask_stage_unification_todo.md
**Classification:** CHECKLIST-COMPLETE for Immediate TODO; Open Questions still valid
**Evidence:** All `Immediate TODO 1-6` `[x]`; canary run `refined_subject_masks_canary_body_eyes_swim_001` recorded. Open Questions 1-5 are genuine future decisions.
**Action:** Reframe as architecture/status note, or archive checklist and keep Open Questions section.

### sam3_colleague_handoff.md
**Classification:** CURRENT
**Evidence:** `run_sam_subject_masks.py`, `run_sam_subject_masks_batch.py`, `visualize_sam_subject_prompts.py` all exist.
**Action:** None.

### sam3_subject_mask_canary_plan.md
**Classification:** STALE-EDIT (checkbox drift)
**Evidence:** Phase 1-4 implementation checkboxes are unchecked, but `run_sam_subject_masks.py` exists and 2026-04-05 timing data appears in the doc body. `subject_v1_union` output schema matches shipped code.
**Action:** Mark Phase 1-4 checkboxes `[x]` to match executed reality.

### session_context.md
**Classification:** CURRENT
**Evidence:** `import_stimulus_to_zarr`, `registry.scan`, `registry.status`, `diagnostics.inspect_session_context` all resolve.
**Action:** None.

### training_data_workflow.md
**Classification:** CURRENT
**Evidence:** Twenty-plus CLIs verified — all resolve.
**Action:** None.

### training_crop_representation_migration.md
**Classification:** CURRENT
**Evidence:** `batch_migrate_training_crop_pixel_contract.py` and `regenerate_training_crops_pynvvc.py` exist; clipped decode-mode policy matches `clipped_training_zarr_implementation_checklist.md` `source_frame_index.parquet` design.
**Action:** None.

### training_label_origin_phase1_audit.md
**Classification:** CURRENT (active audit)
**Action:** None.

### training_label_origin_provenance_todo.md
**Classification:** CURRENT
**Action:** None.

### training_dataset_versioning_todo.md
**Classification:** CURRENT
**Evidence:** Completed items verified in `train_detection.py` and `check_training_registry.py`.
**Action:** None.

### training_performance_todo.md
**Classification:** CURRENT
**Evidence:** Dataloader knobs match `training_params` in detect training.
**Action:** None.

### training_quality_gate_contract.md
**Classification:** CURRENT
**Evidence:** Activation conditions and exclusion-reason vocabulary match `prepare_detect_training_from_registry.py`, `prepare_keypoint_training_from_registry.py`, `export_keypoint_training_zarr.py`. `refined_detect_review_current`/`detect_quality_current` alias consistent.
**Action:** None.

### orange_rolling_clip_recording_contract.md
**Classification:** CURRENT
**Evidence:** `build_recording_frame_index`, `create_clipped_analysis_zarr`, `plan_orange_style_clips`, `materialize_orange_style_clips`, `verify_orange_style_clips` all resolve.
**Action:** None.

### crimson_palette_integration_acceptance_checklist.md
**Classification:** CURRENT
**Action:** None.

### crimson_detect_review_acceptance_contract.md
**Classification:** CURRENT
**Evidence:** `accept_detect_review.py` and `set_detect_review_status.py` exist; `resolved_group="refined"` consistent.
**Action:** None.

### training_data_api_surface_audit.md
**Classification:** CURRENT
**Evidence:** Spot-checked many table rows — all resolve.
**Action:** None.

## Overlaps / Gaps

- **Overlap:** `detection_review_web_todo.md`, `detection_training_zarr_edit_todo.md`, and `analysis_to_training_promotion_contract.md` all describe the `video_detect_review_web --promote-training-zarr` path. They agree; future consolidation could collapse the editing TODO into the contract.
- **Overlap:** `mask_review_save_approval_policy.md`, `swim_bladder_review_policy.md`, `subject_mask_tuning_workflow.md` repeat save-vs-approve principle; cross-links explicit, fine.
- **Gap (consistent):** Crimson save-hook promotion listed as not-yet-wired in `analysis_to_training_promotion_contract.md:62`, `detection_review_web_todo.md`, and `crimson_detect_review_acceptance_contract.md` — consistent across the shard.
- **Broken link:** `crop_review_workflow.md:6` references nonexistent `docs/crop_review_registry_todo.md`.
- **Stale framing (not contradictory):** `crop_distributed_tradeoffs.md` and `crop_persistence_tradeoff.md` predate geometry-only/multi-instance work.
- **Archive candidates:** `subject_mask_component_provenance_followthrough_checklist.md` (all done), Immediate-TODO section of `subject_mask_stage_unification_todo.md` (all done), Phase 0-3 narrative of `detection_training_zarr_edit_todo.md` (all done; Phase 4 remains).
