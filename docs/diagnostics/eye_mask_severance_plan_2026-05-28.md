# Eye-Mask Stage Severance Plan

**Date:** 2026-05-28
**Method:** Read-only parallel dependency trace — 10 agents: crux geometry severance + registry map + 8 file-classification batches covering all ~102 eye-mask-touching files, then synthesis. No files modified.
**Goal:** Plan removal of the legacy standalone `eye_masks` / `refined_eye_masks` pipeline stages. Eyes are now a channel within the unified subject mask (`SUBJECT_MASK_LABEL_SCHEMAS`: `eyes_union` / `eye_left` / `eye_right`).
**Companion docs:** [`repo_eval_2026-05-28.md`](repo_eval_2026-05-28.md), [`contract_drift_audit_2026-05-28.md`](contract_drift_audit_2026-05-28.md).

---

## 1. Blocker verdict — YES, re-sourceable. No hard blocker.

Eye-angle analysis can be fully re-sourced from subject-mask eye channels. The consumer reads only two arrays — `ellipse_params[:, eye_idx, :]` and `ellipse_success[:, eye_idx]` — using only elements `[2] major`, `[3] minor`, `[4] angle_deg` (`eye_angle_analysis.py:945-949`). `masks_roi` is carried in `EyeGeometrySource` but **never touched during angle computation**.

Both replacement sources expose byte-identical 5-float ellipse arrays, fitted with the **same** `cv2.fitEllipse` / `_measure_mask` code the legacy stage uses:
- `refined_subject_masks_runs/<run>/components/eye_left/geometry/` — `refined_subject_eye_geometry.py:100-102`
- `analysis/subject_shape_runs/<run>/components/eye_left/` — `subject_shape_runs.py:1198-1207`
- (same `_measure_mask` as `refine_eye_masks.py:827`)

The resolver **already prefers** them: `eye_angle_analysis.py` calls `resolve_eye_geometry_source(prefer_subject_shape=True, prefer_subject=True)`.

**The real blocker is sequencing, not capability.** `eye_angle_analysis.py:43-48` still *imports* `EYE_GEOMETRY_STAGE_REFINED_EYE` and passes a `refined_eye_run=` argument into the resolver. While that legacy fallback branch (`_build_refined_eye_source` in `eye_geometry_source.py`) exists and is wired, you cannot delete `refine_eye_masks.py` / `eye_geometry_source.py` without breaking the import. **Order is forced: re-point the consumer first, then delete the producer.**

### Honest quality risk: LOW, with two caveats not to gloss over
- The geometry *method* is identical, so for any recording that has refined-subject eye channels, angle output is equivalent. High confidence.
- **Caveat 1 — silent recompute, not bit-identical history.** Subject masks may have undergone different refinement/cleanup than isolated eye masks. Angle depends only on orientation, so this is second-order, but it is *not* "the same numbers" — it is "the same method on possibly-different mask boundaries." If you have cached/published eye-angle results, expect small numeric drift on re-run.
- **Caveat 2 — coverage gap is the real exposure.** Any recording with legacy `refined_eye_masks_runs` but *no* refined-subject eye geometry will silently lose its eye-angle source once the fallback is removed. This is the thing to verify before deleting — not the math.

---

## 2. Ordered severance plan (each phase independently shippable)

### Phase 0 — Coverage audit (gates everything; not a code change)
Confirm every recording you care about has `refined_subject_masks_runs/<run>/components/eye_left/geometry/ellipse_params` populated. For any that only have legacy `refined_eye_masks_runs`, run `backfill_subject_mask_runs.py` (projects eye_masks → subject_mask_runs) first. **This is the only thing that can cause silent data loss.** Also: enumerate how many `.sqlite` registry copies are deployed (needed for Phase 4 migration safety).

### Phase 1 — Sever the consumer (re-point) — behavior-neutral
The forcing function: cut the legacy import so everything downstream can be deleted.
- `eye_angle_analysis.py:43-48` — drop `EYE_GEOMETRY_STAGE_REFINED_EYE` import; stop passing `refined_eye_run=`.
- `eye_geometry_source.py` — remove `_build_refined_eye_source`, the `EYE_GEOMETRY_STAGE_REFINED_EYE` constant, and the `refined_eye_run` fallback branch; simplify resolver to `subject_shape → refined_subject`.
- `visualize_eye_angle_overlays.py:33-42` — drop the refined-eye constant (already abstracts via `eye_geometry_source`).
- `keypoint_failure_review.py` **and `patch_keypoints_from_crops.py:38,938,970`** — both call `mark_downstream_eye_mask_runs_stale`; re-point to the subject-mask stale marker.
- `resolve_eye_mask_stale.py:13`, `apply_tuning_by_camera.py`, `audit_refined_mask_metrics.py`, `materialize_refined_eye_masks_compat.py`, `dispatcher.py:158-168` (drop the deprecation shim) — re-point / strip eye branches.

After Phase 1, eye-angle analysis runs entirely off subject masks; legacy producers are dead code but still present.

### Phase 2 — Delete-now subtraction win — pure deletion
Delete the 36 eye-mask-only files with zero live consumers (§3). Strip the export shims that name them: `segmentation/__init__.py:8-9,29,33`, `inference/__init__.py:3,5`, `benchmark_roi_inference_cache.py:299` (eye scenario). Safe the moment Phase 1 lands.

### Phase 3 — Delete producers after severance
Delete the `delete-after-severance` files: `refinement/refine_eye_masks.py`, `shared/eye_geometry_source.py` (or its remnant), `shared/keypoint_stale.py`, `tune/eye_mask_review.py`, `tune/eye_mask_tuner.py`, `diagnostics/preview_eye_mask_background_subtraction.py`, `diagnostics/review_refined_eye_mask_failures.py`, `utils/run_eye_masks_batch.py`, `utils/run_eye_masks_with_registry_model.py`, `utils/export_eye_mask_training_zarr.py`, and (only after Phase 0 migration confirmed complete) `utils/backfill_subject_mask_runs.py` + `utils/backfill_subject_mask_tuning.py`. Strip eye stages from `core/pipeline.py` (STAGE_ORDER 172-173, deps 190-191, dispatcher 478-481, `_run_eye_masks`/`_run_refined_eye_masks` 662-830, import line 36) and the eye specs in `shared/zarr/schema.py:552-636,750-765`.

### Phase 4 — Registry edits (migrations LAST)
- `stage_catalog.py`: remove `eye_masks`/`refined_eye_masks` StageSpecs (115-127); **remove `'eye_masks'` from `refined_keypoints.invalidates` (line 111)**; remove both from `RECORDING_STATUS_STAGE_IDS` (280-281).
- `stage_complete.py:38-39`: remove both `_STEP_RUN_PARENTS` entries.
- `db.py:231-232,252`: remove eye_mask task_type aliases; strip the 11 `upsert_/replace_/refresh_/query_eye_mask_*` methods.
- `extractors/masks.py:588-773`: delete `_extract_eye_mask_performance_rows` / `_extract_eye_mask_quality_rows` (keep subject extractors 275-585).
- `maintenance.py`: strip `_eye_mask_*` backfill/refresh functions + CLI flags (2540-4535, 8805-9535).
- **Migrations last:** add new drop-migrations for `eye_mask_performance` / `eye_mask_data_profile` / `eye_mask_quality` tables+views, run everywhere, *then* remove the dead `MIGRATION_METHODS` entries (`migrations.py:32,34,42,43`) and the 015/017/025/026 bodies. Dropping bodies before the drop-migration has executed on every registry breaks history replay on older DBs.

---

## 3. Classification

Counts are approximate — category boundaries are fuzzy and one file was reclassified (below). The **delete-now list is exact and actionable.**

| Category | ~Count | Meaning |
|---|---|---|
| delete-now | 36 | eye-mask-only, no live consumer; delete once Phase 1 lands |
| delete-after-severance | 13 | core producers/UI a live consumer depends on until severed |
| re-point | 15 | keepable but consumes eye masks; redirect to subject-mask channels |
| shared-keep-strip | ~20 | mixed-concern; keep the file, strip eye-mask code |
| keep | ~18 | subject-mask or unrelated; no action |

### DELETE-NOW (36 files — the immediate subtraction win)
```
src/fisheye/segmentation/eye_segmentation.py
src/fisheye/segmentation/eye_segmentation_yolo.py
src/fisheye/segmentation/infer_unet_eye_masks.py
src/fisheye/segmentation/train_unet_eye_masks.py
src/fisheye/training/train_eye_masks.py
src/fisheye/training/zarr_eye_mask_dataset.py
src/fisheye/inference/predict_eye_masks.py
src/fisheye/analysis/inspect_refined_eye_masks.py
src/fisheye/diagnostics/check_eye_mask_ellipse_axes.py
src/fisheye/diagnostics/check_eye_mask_keypoint_coverage.py
src/fisheye/diagnostics/check_eye_mask_lineage.py
src/fisheye/diagnostics/check_eye_masks.py
src/fisheye/diagnostics/check_mask_components.py
src/fisheye/diagnostics/parse_eye_mask_lineage_log.py
src/fisheye/diagnostics/show_eye_mask_provenance.py
src/fisheye/diagnostics/show_eye_mask_runs.py
src/fisheye/diagnostics/show_refined_eye_mask_reason_tags.py
src/fisheye/tune/eye_mask_failure_review.py
src/fisheye/visualization/t.py
src/fisheye/visualization/visualize_eye_mask_ellipse_fit_comparison.py
src/fisheye/visualization/visualize_eye_mask_patches.py
src/fisheye/visualization/visualize_eye_masks.py
src/fisheye/utils/aggregate_eye_mask_training_data_card.py
src/fisheye/utils/backfill_eye_mask_lineage_attrs.py
src/fisheye/utils/backfill_eye_mask_profiles.py
src/fisheye/utils/export_eye_mask_quality_overview.py
src/fisheye/utils/eye_mask_profile.py
src/fisheye/utils/finalize_eye_mask_profile_artifacts.py
src/fisheye/utils/inspect_eye_mask_source_areas.py
src/fisheye/utils/plot_eye_mask_training_data_card.py
src/fisheye/utils/prepare_eye_mask_training_from_registry.py
src/fisheye/utils/prune_legacy_eye_mask_profile_runs.py
src/fisheye/utils/review_eye_masks_batch.py
src/fisheye/utils/run_eye_mask_training_pipeline.py
src/fisheye/utils/sync_eye_mask_profile_registry.py
src/fisheye/utils/validate_eye_mask_training_zarr.py
```
> `visualization/t.py` is fully commented-out dead code (not actually eye-mask-coupled) — delete it regardless, just don't count it as an eye-mask win.

### SHARED-KEEP-STRIP — careful surgery (what else each serves)
- `segmentation/__init__.py`, `inference/__init__.py` — serve subject/pose/detect exports; strip eye lines.
- `training/losses.py` — `MaskedBCEDiceCriterion` serves subject masks; strip `BCEDiceCriterion` + `overlap_weight`.
- `training/config.py` — keep `SubjectMaskTrainingParams` + `SUBJECT_MASK_LABEL_SCHEMAS` (182-185); strip `EyeMaskTrainingParams`.
- `core/pipeline.py` — orchestrates all stages; strip eye branches only.
- `shared/zarr/schema.py` — serves every stage; delete only the two eye specs.
- `registry/db.py`, `registry/maintenance.py` — serve subject-mask + all-stage registry; strip eye methods.
- `diagnostics/check_full_provenance.py`, `check_provenance_capture.py`, `check_provenance_consistency.py` — serve subject/arena/detect/crop/keypoint provenance; strip eye blocks.
- `diagnostics/benchmark_roi_inference_cache.py` — serves keypoint benchmarking.
- `utils/check_training_registry.py`, `index_source_recording_profiles.py`, `index_training_data_cards.py`, `registry_query.py`, `registry_tui.py`, `prune_zarr_runs.py`, `migrate_training_label_runs_identity.py` — serve detect/keypoint/subject profiles; strip eye paths. **`check_training_registry.py` alone has ~40 eye-mask call sites — budget real time.**

### Classification correction (unflinching)
**`utils/patch_keypoints_from_crops.py` was mis-classified `keep` → it is `re-point`.** It imports and calls `mark_downstream_eye_mask_runs_stale` at lines 38, 938, 970 — identical coupling to `keypoint_failure_review.py`. When `keypoint_stale.py` is deleted in Phase 3, its import breaks. Fix it in Phase 1.

---

## 4. Risks & unknowns

1. **Coverage gap (highest risk; verify before Phase 3).** The trace asserts geometry *parity* but cannot confirm *which recordings* have populated `refined_subject_masks_runs/.../eye_left/geometry/`. Verify per-recording before deleting producers. `backfill_subject_mask_runs.py` must outlive the delete-now phase.
2. **`patch_keypoints_from_crops.py`** — confirmed live coupling, folded into Phase 1.
3. **Migration replay hazard.** Dropping migration bodies before the drop-migration runs on every registry breaks history replay. Order in Phase 4 is mandatory. **Unknown: how many `.sqlite` registries are deployed** — enumerate first (Phase 0).
4. **`materialize_refined_eye_masks_compat.py` / `refined_eye_masks_compat.py`** synthesize a `refined_eye_masks_runs` *compatibility view from subject masks*. If any external/notebook tooling outside this repo reads that compat group, deletion has off-repo blast radius the trace can't see. Verify no external consumers before Phase 3.
5. **String-command coupling** (not imports): `benchmark_roi_inference_cache.py:299`, `registry_query.py:1233`, `check_training_registry.py:326` reference eye-mask modules as text. They won't fail at import but will print dead commands after deletion — strip in the same pass.
6. **Unresolved:** whether `subject_shape_runs` is materialized for all recordings or only some. The resolver prefers `subject_shape` first; if sparse, the effective source is `refined_subject` for most data. Not a blocker (both parity), but know which source your numbers come from before asserting "nothing changed."

> **Forcing function:** Phase 0 + Phase 1 are the whole risk. Once the consumer is severed and coverage is confirmed, Phases 2–4 are mechanical subtraction. Do not start deleting files (the satisfying part) before the audit (the boring part).
