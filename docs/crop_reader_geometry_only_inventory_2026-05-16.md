# Crop Reader Geometry-Only Inventory - 2026-05-16
<!-- contract-meta
status: inventory
last_verified: 2026-05-16
purpose: Classify Palette crop readers before making analysis crop runs default to geometry-only.
-->

This inventory supports the geometry-only crop migration described in
`docs/geometry_only_crop_workflow_cache_design.md`. It is a code-facing
snapshot, not a storage contract.

## Decision

Analysis crop runs may default to `crop_storage_mode=geometry_only` only for
pipeline stages whose readers go through `CropImageSource` or otherwise declare
mixed-mode support. Training archives and materialized-only tools must continue
to resolve `crop_runs.attrs["latest_materialized"]` or an explicit materialized
run.

## Mixed-Mode Safe Readers

These readers use `CropImageSource.open(...)` and can resolve explicit
`--crop-run`, `latest_any`, legacy `latest`, then `latest_materialized`.

| Area | Paths | Notes |
|------|-------|-------|
| YOLO pose inference | `src/fisheye/detection/detect_keypoints_yolo.py` | Reads materialized crops or live geometry-derived crops. |
| YOLO eye masks | `src/fisheye/segmentation/eye_segmentation_yolo.py` | Uses shared ROI source. |
| U-Net eye masks | `src/fisheye/segmentation/infer_unet_eye_masks.py` | Uses shared ROI source. |
| U-Net subject masks | `src/fisheye/segmentation/infer_unet_subject_masks.py` | Uses shared ROI source. |
| Subject segmentation | `src/fisheye/segmentation/subject_segmentation.py` | Uses shared ROI source. |
| Swim-bladder segmentation | `src/fisheye/segmentation/swim_bladder_segmentation.py` | Uses shared ROI source. |
| SAM subject masks | `src/fisheye/utils/run_sam_subject_masks.py` | Uses shared ROI source. |
| Keypoint retry | `src/fisheye/utils/keypoint_retry.py` | Uses shared ROI source. |
| ROI-cache benchmark | `src/fisheye/diagnostics/benchmark_roi_inference_cache.py` | Mixed-mode diagnostic path. |
| Eye-mask visual overlays | `src/fisheye/visualization/visualize_eye_masks.py`, `src/fisheye/visualization/visualize_eye_mask_patches.py`, `src/fisheye/visualization/visualize_swim_bladder_mask_patches.py`, `src/fisheye/visualization/visualize_eye_mask_ellipse_fit_comparison.py`, `src/fisheye/visualization/visualize_subject_shape_overlays.py` | Visualization paths already use the resolver. |

Batch planners for these paths should treat `latest_any` as crop-ready and
should pass explicit crop run names between workflow stages.

## Materialized-Only By Design

These paths intentionally require persisted ROI pixels and should not silently
accept geometry-only crop runs.

| Area | Paths | Reason |
|------|-------|--------|
| Training datasets and exports | `src/fisheye/training/*`, `src/fisheye/utils/export_*_training_zarr.py`, `src/fisheye/utils/prepare_keypoint_training_from_registry.py`, `src/fisheye/utils/validate_eye_mask_training_zarr.py`, `src/fisheye/utils/validate_keypoint_training_zarr.py`, `src/fisheye/utils/validate_subject_mask_training_zarr.py`, `src/fisheye/utils/view_merged_pose_training_zarr.py` | Training archives are portable artifacts and should carry persisted images. |
| Traditional keypoints | `src/fisheye/detection/detect_keypoints_traditional.py` | Uses `resolve_materialized_crop_run(...)` and directly reads `roi_images`. |
| Traditional eye masks | `src/fisheye/segmentation/eye_segmentation.py` | Directly reads `crop_runs/<run>/roi_images`. |
| Crop patch tools | `src/fisheye/utils/patch_crops_from_refined.py`, `src/fisheye/utils/patch_keypoints_from_crops.py` | These edit or inspect persisted crop pixels. |
| Crop review tools | `src/fisheye/utils/review_crops.py`, crop-focused tuning tools | Their purpose is reviewing or editing persisted ROI pixels. |
| Keypoint refinement/backfill helpers | `src/fisheye/refinement/refine_keypoints.py`, `src/fisheye/refinement/assemble_refined_subject_masks.py`, `src/fisheye/utils/backfill_keypoint_derived_metrics.py`, `src/fisheye/utils/backfill_keypoint_edge_distances.py`, `src/fisheye/training/analyze_keypoints_integrity.py`, `src/fisheye/utils/repair_keypoint_offset_corruption.py` | These use persisted crop shape or pixels for legacy metrics and diagnostics. |
| Interpolated crop regeneration | `src/fisheye/refinement/regenerate_interpolated_crops.py` | Writes or patches persisted `roi_images`. |
| Training data-card and validation utilities | `src/fisheye/utils/aggregate_keypoint_training_data_card.py`, `src/fisheye/utils/validate_subject_mask_training_zarr.py`, `src/fisheye/utils/audit_subject_mask_training_sources.py` | These inspect self-contained training artifacts. |

Batch planners for these paths should resolve `latest_materialized` first and
fail clearly if no materialized crop run exists.

## Stale Or Deferred Direct Readers

These paths still dereference `crop_group["roi_images"]` directly. Some should
remain materialized-only; others may later move to `CropImageSource` if they
become analysis workflows.

| Area | Paths | Current policy |
|------|-------|----------------|
| Keypoint and eye-mask tuners | `src/fisheye/tune/keypoint_tuner.py`, `src/fisheye/tune/keypoint_review.py`, `src/fisheye/tune/keypoint_failure_review.py`, `src/fisheye/tune/eye_mask_tuner.py`, `src/fisheye/tune/eye_mask_review.py`, `src/fisheye/tune/eye_mask_failure_review.py` | Defer; these are review/editing surfaces and can require materialized crops until redesigned. |
| Subject/swim-bladder review and tuning | `src/fisheye/tune/subject_mask_tuner.py`, `src/fisheye/tune/refined_subject_mask_review.py`, `src/fisheye/tune/swim_bladder_mask_tuner.py`, `src/fisheye/visualization/subject_mask_inspector.py` | Defer; review UI may need explicit materialization or a future live-render mode. |
| Simple visualizers | `src/fisheye/visualization/visualize_crops.py`, `src/fisheye/visualization/visualize_keypoints.py`, `src/fisheye/visualization/visualize_eye_angle_overlays.py`, `src/fisheye/visualization/visualize_sam_subject_prompts.py`, `src/fisheye/visualization/t.py` | Defer or migrate to `CropImageSource` when these are needed for geometry-only archives. |
| Diagnostics | `src/fisheye/diagnostics/preview_eye_mask_background_subtraction.py`, `src/fisheye/diagnostics/check_crop_runs.py`, `src/fisheye/diagnostics/check_provenance_consistency.py`, `src/fisheye/diagnostics/check_refined_roi_links.py`, `src/fisheye/diagnostics/show_eye_mask_runs.py`, `src/fisheye/utils/benchmark_zarr_destination_reads.py`, `src/fisheye/utils/audit_keypoint_coordinate_spaces.py`, `src/fisheye/utils/analyze_bad_keypoint_row_overlap.py` | Keep diagnostics explicit about whether they inspect materialized arrays or mixed-mode reads. |
| Storage/export utilities | `src/fisheye/utils/export_sharded_zarr_clone.py` | Preserve existing array-selection semantics; geometry-only support here means skipping absent `roi_images`, not reconstructing them. |
| Registry, schema, and metadata summaries | `src/fisheye/registry/db.py`, `src/fisheye/shared/zarr/stage_arrays.py`, `src/fisheye/utils/backfill_crop_storage_metadata.py`, `src/fisheye/utils/crop_batch.py`, `src/fisheye/utils/run_keypoints_batch.py`, `src/fisheye/utils/run_eye_masks_batch.py` | These should treat `roi_images` as conditional and use storage-mode-aware pointers. |
| Analysis summaries | `src/fisheye/analysis/chaser_phase_analysis.py`, `src/fisheye/training/train_pose.py` | These only summarize ROI counts/shapes and should tolerate absent `roi_images` when used on geometry-only analysis archives. |

## Batch Planner Implications

- `run_keypoints_batch` is method-dependent: YOLO pose can accept
  `latest_any`; traditional pose requires `latest_materialized`.
- `run_eye_masks_batch` is method-dependent: YOLO and U-Net can accept
  `latest_any`; traditional eye masks require `latest_materialized`.
- `run_subject_mask_batch_pipeline` invokes mixed-mode U-Net subject-mask
  inference, so crop readiness should use `latest_any`.
- `crop_batch` may default analysis archives to `geometry_only`, but training
  archives must continue to reject geometry-only crop writes.

## Open Follow-Ups

- Inventory Crimson crop consumers before changing any cross-repo contract.
- Add a materialization command for geometry-only analysis crop runs when a
  review surface needs persisted `roi_images`.
- Build and benchmark the shared workflow ROI cache after the planner and
  pointer semantics are stable.
