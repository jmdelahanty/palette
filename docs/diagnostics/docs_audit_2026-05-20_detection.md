# Detection Docs Audit (2026-05-20)

## Summary

Reviewed 18 docs in the detection / refinement / detect-review shard (after excluding 3 out-of-shard files: `crimson_stimulus_step_*`, `crimson_swim_bout_compact_v2_*`, `crimson_track_motion_*` — these are stimulus/track docs that match the include-pattern but are not detection). Buckets:

- CURRENT: 11
- STALE-EDIT: 4
- ARCHIVE: 0
- CHECKLIST-COMPLETE: 3

The shard is in noticeably good shape — most sparse-first refined-detect docs were updated through 2026-04 and 2026-05. Main stale items are checklist/TODOs whose items are now all checked, and two contracts still in proposal voice for behavior that is now implemented. `crimson_refined_detect_manual_contract.md` is already explicitly `status: historical`; kept as legacy reference.

## Findings

### crimson_detect_bbox_read_contract.md
**Classification:** CURRENT
**Evidence:** `refined_detect_runs/<run>/instances` as canonical surface matches `src/fisheye/shared/refined_detect_curation.py` (`refined_storage_semantics = "sparse_instances_v1"` at line 1204; frame_offsets at line 70/924). `fisheye.utils.validate_refined_detect_run`, `rechunk_refined_detect_bbox_arrays`, `backfill_refined_detect_bbox_img_xyxy` all exist.
**Action:** Keep. `last_verified: 2026-05-16` is recent.

### crimson_detect_review_acceptance_contract.md
**Classification:** CURRENT
**Evidence:** `src/fisheye/utils/set_detect_review_status.py` and `accept_detect_review.py` both exist. Resolved-group preference chain matches sparse-first contract.
**Action:** Keep. Consider bumping `last_verified` from 2026-04-15.

### crimson_refined_detect_manual_contract.md
**Classification:** CURRENT (intentionally historical)
**Evidence:** Explicitly `status: historical`. Referenced legacy modules `src/fisheye/tune/detect_review.py` and `src/fisheye/shared/refined_detect_review.py` still exist.
**Action:** Keep as historical reference.

### detect_batch_analysis_zarr_parallel_agents_contract.md
**Classification:** CURRENT
**Evidence:** All named modules exist: `src/fisheye/utils/run_detections_batch.py`, `run_detect_with_registry_model.py`, `list_unapproved_analysis_zarrs.py`, `scripts/submit_detect_batches_bsub.sh`.
**Action:** Keep. Candidate for archive once DBI tasks fully closed.

### detect_decode_backend_benchmark_todo.md
**Classification:** CURRENT
**Evidence:** `detect_yolo` exposes `--decode-backend auto|pynvvc_nv12_rgb|pynvvc_luma_rgb|decord_gpu|decord_cpu|opencv` at `src/fisheye/detection/detect_yolo.py:55-813`; auto-prefers pynvvc when CUDA available (lines 779-799). `stream_event_owned_batch_v2` marker present.
**Action:** Keep, but verbose; consider trimming 2026-05-09/-11 narrative into a focused findings-plus-current-defaults doc since integration tasks are all Done.

### detection_chunking_findings.md
**Classification:** STALE-EDIT
**Evidence:** Bottom half ("Decord vs Native Decode Benchmark Plan") duplicates `detect_decode_backend_benchmark_todo.md` in proposal voice; that work is done. Loader line citations (`zarr_yolo_dataset_loader.py:919, 752, 778`) should be re-verified.
**Action:** Trim the Decord vs Native section to a pointer; re-verify the loader line numbers.

### detection_data_profile_schema_contract.md
**Classification:** CURRENT
**Evidence:** `src/fisheye/utils/detection_profile.py`, `backfill_detection_profiles.py`, `sync_detection_profile_registry.py` all present. Approval-side profile writing matches.
**Action:** Keep. Bump `last_verified` from 2026-04-15.

### detection_merged_export_contract.md
**Classification:** CURRENT
**Evidence:** Updated to active v3 on 2026-05-20. The merged exporter now writes the canonical `refined_detect_runs/<run>/instances` label surface, records `training_export.canonical_label_path`, and rejects interpolated source-kind rows by default. `validate_merged_training_zarr` now validates the canonical instances surface rather than a crop-run label mirror.
**Action:** Keep. Re-check after the next full exported-dataset smoke.

### detection_refinement_workflow.md
**Classification:** CURRENT
**Evidence:** All CLIs exist; `--per-frame-top-k` flag verified at `src/fisheye/refinement/refine_detect.py:502, 532, 897`. `review_detect_batch`, `migrate_legacy_detect_labels`, `inspect_refined_detect_run`, `accept_detect_review` all present.
**Action:** Keep. Consider adding `<!-- contract-meta -->` with last_verified.

### detection_review_web_todo.md
**Classification:** STALE-EDIT (largely implemented; recently updated to 2026-05-20)
**Evidence:** Slice 1 MVP boxes all `[x]`. `src/fisheye/tune/detect_review_backend.py`, `detect_review_web.py`, static assets, and `video_detect_review_web.py` all exist. `--promote-training-zarr` wired at `video_detect_review_web.py:1240-1322`. Promotion backend implemented at `src/fisheye/tune/detect_training_promotion_backend.py`. Remaining unchecked: real-zarr smoke, browser smoke, arena-aware editing, proxy frame-count validation.
**Action:** Rename / restructure. The "MVP TODO" framing no longer matches scope (now covers two web reviewers + proxy videos + promotion). Suggest renaming to `detection_review_web_status.md` or splitting into a contract for detect_review_web vs video_detect_review_web, with a small leftover checklist.

### detection_training_plan.md
**Classification:** STALE-EDIT
**Evidence:** Plan-of-record items 1–6 mostly implemented. References single `prepare_detect_training` module; active CLI is `prepare_detect_training_from_registry.py`. Default registry path `runs/registry/palette_registry.sqlite` differs from the de-facto `/nvme1/palette_registry.sqlite` used elsewhere. H5 capture section reads as future plan but is wired today.
**Action:** Rewrite as "Detection Training Pipeline (Current)" pointing at the live contracts (`detection_refinement_workflow.md`, `detection_data_profile_schema_contract.md`, `detection_merged_export_contract.md`), or mark as historical-design.

### detection_training_zarr_edit_todo.md
**Classification:** CHECKLIST-COMPLETE
**Evidence:** All Phase 0/1/2 items checked. One Phase 3 end-to-end smoke unchecked; Phase 4 deferred. Backend/web/tests all present.
**Action:** Move to `docs/archive/`. Overlaps heavily with `detection_review_web_todo.md`; absorb the few unchecked items there.

### detect_quality_parallel_agents_contract.md
**Classification:** CHECKLIST-COMPLETE
**Evidence:** Companion TODO `docs/detect_quality_registry_todo.md` is already in `docs/archive/`. View rename `refined_detect_review_current` (with `detect_quality_current` compat alias) implemented at `src/fisheye/registry/db.py:3722-3765`. All owned modules exist.
**Action:** Move to `docs/archive/`.

### refined_detect_collapse_v2.md
**Classification:** CURRENT
**Evidence:** Matches implemented sparse-first contract; behaviors align with `refine_detect.py` and `refined_detect_curation.py`.
**Action:** Keep. Primary design reference.

### refined_detect_downstream_adoption_checklist.md
**Classification:** CHECKLIST-COMPLETE (Palette side; small cross-repo residue)
**Evidence:** All Palette-side items `[x]`. Remaining `[ ]` items are cross-repo (contracts repo, Crimson runtime, optional historical cleanup). 2026-04-15 audit shows zero latest-run legacy conflicts.
**Action:** Archive Palette-internal portions; either move entirely to `docs/archive/` and create a short cross-repo tracker, or trim to just the cross-repo items.

### refined_detect_multisubject_goal.md
**Classification:** CURRENT
**Evidence:** `status: draft` long-term design note. Accurately describes target vs the active sparse v1 surface.
**Action:** Keep.

### refined_detect_row_identity_contract.md
**Classification:** CURRENT
**Evidence:** `fisheye.shared.refined_detect_identity` exists at `src/fisheye/shared/refined_detect_identity.py`; `inspect_refined_detect_run` and `backfill_crop_row_lineage` both present. Required arrays match curation writer.
**Action:** Keep. `last_updated: 2026-04-24` recent.

### refined_detect_sparse_instances_schema.md
**Classification:** CURRENT (label drift)
**Evidence:** Schema fields all written by `refined_detect_curation.py` (lines 70, 924, 1204-1206, 1495). `refined_storage_semantics = "sparse_instances_v1"` literal matches code.
**Action:** Bump `status: draft -> active` (or `implemented`); update `last_updated`. The "draft" label is the only stale element.

## Overlaps / Gaps

### Overlaps worth consolidating

1. **`detection_review_web_todo.md` vs `detection_training_zarr_edit_todo.md`** — heavy overlap on the same web reviewer MVP. Fold the latter into the former and archive the standalone TODO.

2. **`detect_decode_backend_benchmark_todo.md` vs bottom half of `detection_chunking_findings.md`** — the chunking doc duplicates the "decode vs native" plan that the benchmark doc has since superseded with measurements. Trim chunking doc to just chunking.

3. **`detection_training_plan.md` vs `detection_refinement_workflow.md` / `detection_data_profile_schema_contract.md` / `detection_merged_export_contract.md`** — the "plan" doc is a higher-level recap of what is now implemented across the other three. Rewrite as a short index or archive.

4. **`refined_detect_collapse_v2.md` vs `refined_detect_sparse_instances_schema.md` vs `refined_detect_multisubject_goal.md`** — three sibling design docs with reasonable pattern (active contract → schema → long-term goal), but a one-line "see also" header in each would help.

### Gaps

1. **Clipped refined-detect resolver contract.** Multiple docs reference `fisheye.utils.resolve_clipped_refined_detect_collection`, `experiment_index/finalized_runs/`, and `recording_frame_index.parquet` but no dedicated `clipped_refined_detect_*` doc in this shard. `crimson_detect_bbox_read_contract.md` defers to `docs/clipped_recording_consumer_mapping_contract.md` — confirm that anchor or add a dedicated resolver contract.

2. **`detect_training_promotion_backend` contract.** Module `src/fisheye/tune/detect_training_promotion_backend.py` (untracked in git status) is the post-save promotion hook for `video_detect_review_web`. `docs/analysis_to_training_promotion_contract.md` (currently being edited) should cover it; cross-link from `detection_review_web_todo.md`.

3. **`refine_detect` per-frame top-k and dish-mask spatial gating.** Documented only in `detection_refinement_workflow.md` prose. No contract-level doc covers the `outside_dish_mask` decision label, the `quality_filtered_per_frame_top_k_sparse_instances_no_interpolation` method string (`refine_detect.py:1412`), or the sampled-import passthrough mode. Worth a section in `refined_detect_sparse_instances_schema.md` or a small dedicated contract.

4. **`detect_quality` artifact layer** (`detect_runs/<run>/quality_reports/<qrun>`) referenced in workflow + collapse-v2 docs but has no schema/contract file in this shard. Archived `detect_quality_registry_todo.md` covered the registry projection but not the artifact itself.

5. **`detect_review_web` vs `video_detect_review_web` divergence.** Two web reviewers with different backing data (materialized images vs source video, traditional vs clipped). A short comparison contract would help operators pick the right tool.
