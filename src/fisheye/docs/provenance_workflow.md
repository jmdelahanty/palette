# Provenance-Safe Refinement Workflow

For the active operator-facing source-of-truth contract across stage families,
see
[current_pipeline_contract.md](../../../docs/current_pipeline_contract.md).
This note remains the deeper provenance workflow reference.

This note captures the sequencing we used to align refined detections, crops, keypoints, arena assignment, and tracking so downstream analysis (track kinematics, metrics, plots) stays consistent.

## Stage Overview

```
detect_runs → refined_detect_runs → crop_runs → keypoints_runs → refined_keypoints_runs
                                           ├─ subject_mask_runs → refined_subject_masks_runs
                                           └─ eye_masks_runs → refined_eye_masks_runs
refined_detect_runs → arena_assignment_runs → tracking_runs → track_kinematics_runs
keypoints_runs ─(sealed assignment proof)─┐
refined_subject_masks_runs ──────────────└→ analysis/subject_shape_runs → analysis/eye_angle_runs
```

Current mask-local eye geometry authority is `refined_subject_masks_runs/<run>`
when that run contains `eye_left` and `eye_right` components.
`analysis/subject_shape_runs/<run>` is the canonical analysis-facing coherent
body/eyes/swim shape surface for future-normal eye-angle publication.
`analysis/eye_angle_runs` requires its eye geometry and the exact base-keypoint
publication sealed by its assignment proof. It does not fall back to latest or
refined keypoints. Historical refined keypoints and refined-subject geometry
remain available only through an explicit diagnostic request; the resulting
eye-angle output is permanently nonselector.
Shared fish-relative body-frame outputs should materialize under
`analysis/subject_shape_runs/<run>/body_frame/` when a subject-shape run
produces them; analysis-local body-frame caches remain support data.

Relevant provenance attributes:

| Stage | Primary Array | Key Attrs |
| --- | --- | --- |
| `detect_runs/<run>` | `bbox_norm_coords` | `detect_timestamp_utc`, `total_detections` |
| `refined_detect_runs/<run>` | `instances/bbox_norm_coords`, `source_detections/bbox_norm_coords` | `source_detect_run`, `detect_review_status`, `refined_storage_semantics`, `source_detection_decision_code_map` |
| `crop_runs/<run>` | `roi_images`, `source_refined_row_ids` *(when sourced from canonical refined detect)* | `detection_source_path`, `detect_review_status_ref`, `detect_review_status` (snapshot), `detection_selection_policy`, `crop_signature`, `roi_image_representation`, `roi_pixel_contract_name`, `roi_pixel_contract`, `crop_review_status` |
| `keypoints_runs/<run>` | `heading`, `frame_indices` | `source_crop_run`, `source_crop_storage_mode`, `source_crop_signature`, `source_crop_revision`, `source_detect_review_status_ref`, `source_roi_image_representation`, `source_roi_pixel_contract_name`, `source_roi_pixel_contract`, `requested_device`, `normalized_torch_device`, `resolved_model_device`, `scheduler_job_id`, `scheduler_hosts` |
| `refined_keypoints_runs/<run>` | `heading`, `usable_keypoints`, `reason_bytes` | `source_keypoints_run`, `source_crop_run`, `source_crop_storage_mode`, `source_crop_signature`, `source_crop_revision`, `source_detect_review_status_ref`, `keypoint_signature`, `keypoint_review_status`, `reason_authority`, `reason_fallback_order`, `pose_schema`, `heading_computation_override`, `derived_metrics_schema` |
| `eye_masks_runs/<run>` | `masks_roi` | `source_crop_run`, `source_crop_storage_mode`, `source_crop_signature`, `source_crop_revision`, `source_detect_review_status_ref`, `source_roi_image_representation`, `source_roi_pixel_contract_name`, `source_roi_pixel_contract`, `source_keypoint_group`, `source_keypoints_run` *(legacy alias: `source_keypoint_run`)* |
| `subject_mask_runs/<run>` | `mask_probs_roi`; optional `masks_roi` convenience threshold | `source_crop_run`, `source_crop_storage_mode`, `source_crop_signature`, `source_crop_revision`, `source_detect_review_status_ref`, `source_roi_image_representation`, `source_roi_pixel_contract_name`, `source_roi_pixel_contract`, `source_roi_read_mode`, `roi_cache_policy`, `source_roi_cache_used`, `source_roi_cache_backend`, `source_roi_cache_canonical_path`, `label_schema_id`, `run_semantics`, `masks_roi_materialized` |
| `refined_subject_masks_runs/<run>` | logical mask store via `MaskStore` (`masks_roi`, `mask_bitpacked`, and/or `mask_rle`), component geometry, `relations/eye_pair/metrics/separation_px` | `source_subject_mask_run`, `source_crop_run`, `source_crop_storage_mode`, `source_crop_signature`, `source_crop_revision`, `source_detect_review_status_ref`, `source_roi_image_representation`, `source_roi_pixel_contract_name`, `source_roi_pixel_contract`, `source_roi_read_mode`, `source_roi_cache_canonical_path`, `mask_storage_encoding`, `mask_store_encodings`, `refined_subject_mask_review_status`, `component_review_statuses`, `source_refined_eye_masks_run` *(when seeded from compatibility eye data)* |
| `refined_eye_masks_runs/<run>` | `masks_roi`, `ellipse_params` | Compatibility/historical refined-eye layout. Key attrs include `source_eye_masks_run`, `source_keypoint_group`, `source_keypoints_run` *(legacy alias: `source_keypoint_run`)*, and `source_refined_subject_masks_run` when derived from canonical refined-subject masks. |
| `analysis/subject_shape_runs/<run>` | component summaries, eye/swim ellipse summaries, body axes, `body_frame/`, relation metrics | `schema_id`, `schema_version`, `row_axis`, `method`, `method_version`, `source_refined_subject_masks_run`, `source_mask_labels`, `source_mask_label_schema_id`, `source_mask_store_encoding`, `source_mask_storage_surface`, `source_mask_store_path`, `source_refs`, optional `body_frame_schema_id`, `body_frame_estimator`, `body_frame_source_refs` |
| `analysis/eye_angle_runs/<run>` | logical `angles/roi`, `angles/frame`, `qa/roi`, `qa/frame`; `support/instance_key`, `support/source_acquisition_frame_index`, and equality-required `support/frame_indices` alias | `schema_id`, run `schema_version = 6`, `method`, `method_version`, `row_axis`, output schema v9, `keypoint_source_mode`, `source_eye_geometry_stage`, `source_eye_geometry_run`, `source_geometry_kind`, `source_subject_shape_run`, `source_base_keypoints_run`, `source_instance_key_path`, `source_acquisition_frame_index_path`, `eye_angle_source_contracts`, `eye_angle_algorithm_contract`, `stage_selector_eligible` |
| `arena_assignment_runs/<run>` | `arena_ids` | `source_detect_run`, `source_refined_run` |
| `tracking_runs/<run>` | `track_ids`, `track_arena_ids` | `source_detect_run`, `source_refined_run`, `source_arena_assignment_run`, `tracking_qc_state` |

Canonical eye-angle row identity is the ordered `instance_key`; acquisition
frames are a temporal coordinate, not an alternate identity. The writer
recomputes its body frame and output heading from the proof-bound keypoints and
success mask. It does not consume a separately persisted upstream heading.

Keypoint-lineage attribute contract for legacy eye-mask stages:

- Canonical attr is `source_keypoints_run`.
- `source_keypoint_run` is a legacy compatibility alias and should not be used as the primary key in new tooling.
- Legacy writers should write canonical lineage (and may mirror the legacy alias during migration).
- Readers/diagnostics resolve canonical first, then legacy alias fallback.
- `check_eye_masks` reports legacy-only lineage as `legacy` (warning) and missing/empty lineage as `incomplete`.

Keypoint GPU and scheduler provenance:

- GPU cluster keypoint jobs should be submitted with an explicit device, e.g.
  `--gpus 1` on `scripts/submit_keypoints_batches_bsub.sh`, which requests
  `-gpu num=1` from LSF and defaults the per-recording command to `--device 0`.
- Raw YOLO keypoint runs record requested and effective device metadata:
  `requested_device`, `normalized_torch_device`, `initial_model_device`, and
  `resolved_model_device`.
- Cluster placement is recorded both in run attrs/provenance and registry
  `recording_step_status.details_json`: `execution_hostname`,
  `scheduler_job_id`, `scheduler_job_name`, `scheduler_job_index`,
  `scheduler_queue`, `scheduler_hosts`, `scheduler_mcpu_hosts`,
  `scheduler_gpu_request`, and `scheduler_cuda_visible_devices`.
- Explicit flat-cache runs record `roi_cache_staging_policy` and the nested
  `roi_cache_staging` payload. Large PRFS flat caches should use
  `node_scratch_staged_flat_cache`; direct reads should be represented as
  `direct_manifest_read` and treated as a deliberate comparison or small-cache
  path.
- Use these fields when comparing packed array jobs against one-recording-per-GPU
  jobs. If many recordings share one L4 node, slower throughput may reflect
  resource contention rather than model or cache regressions.

Eye-mask lineage arrays (`frame_indices`, `detection_indices`, `frame_counts`,
`source_refined_row_ids`, `source_detect_row_index`) follow the contract in
`docs/archive/eye_mask_row_mapping_contract.md`:

- segmentation writes should anchor lineage to source crop runs;
- keypoint lineage arrays are used for cross-check/fallback compatibility;
- refinement copies lineage arrays from the source eye-mask run.

`check_provenance_consistency` now validates four crop-side contracts:

- `crop_runs/<run>` must still match the current upstream detect/refined rowset.
- latest keypoint, eye-mask, subject-mask, and refined-subject-mask runs must
  carry a crop snapshot (`source_crop_*` plus
  `source_detect_review_status_ref`) that still matches the current crop run
  they reference.

Legacy refined-detect sparse subgroups such as `interpolated` and `manual_*`
may still exist in older archives, but they are no longer the primary current
provenance surface for detect.

For in-place crop repair runs (`patch_crops_from_refined`), provenance is
captured at two levels:

- run-level cumulative arrays: `patched_detection_indices`,
  `patched_frame_indices`
- per-event history entries in `crop_patch_history`, including exact
  crop-run row targets in `patched_detection_indices`, mapped source rows in
  `patched_source_detection_indices`, `patched_frame_indices`, and when the
  source is curated refined detect, `patched_refined_row_ids`

`bbox_norm_coords` in detect/refined-detect groups use normalized
`[cx, cy, w, h]`.

For current refined detect/keypoint reason labels, use `reason_bytes` and then
labels derived from `detection_source`. Historical readers may use legacy
`reason` between those steps, but writers must not create or update it.

For keypoint heading semantics, resolve metadata in this order:

1. run attr `heading_computation_override`
2. `pose_schema.metadata.heading_computation`
3. deprecated run attr `heading_computation`
4. no heading metadata available

See `docs/keypoint_heading_computation_contract.md`.

For fish-relative body-frame semantics, use
`docs/body_frame_contract.md`. Keypoint heading is the fallback estimator;
mask/spline/hybrid body-frame products should declare estimator provenance and
source refs.

For run-level derived metric semantics and boolean/status gates, prefer
`derived_metrics_schema` when present.

See `docs/derived_metrics_schema_contract.md`.

## Legacy: Regenerating Interpolated Crops

This section is for legacy archives that still carry interpolated detect boxes.
It is not part of the normal current detect-refinement workflow.

1. **Regenerate crops** (GPU decode will be used when available):
   ```bash
   scripts/py -m fisheye.refinement.regenerate_interpolated_crops \
     <archive>.zarr \
     --video-path /path/to/video.mp4 \
     --source-crop-run <existing_crop_run> \
     --overwrite
   ```
   * `--source-crop-run` identifies which crop run to augment. If omitted, the script now attempts to infer the crop run based on `detection_source_path`.

2. Inspect linkage:
   ```bash
   scripts/py -m fisheye.diagnostics.check_refined_roi_links <archive>.zarr
   ```
   Confirms `interpolated_roi_path`, counts, decoder, and crop metadata stay in sync.

## Re-running Dependent Stages

After new ROIs exist, rerun stages that depend on them:

1. YOLO keypoints:
   ```bash
   scripts/py -m fisheye.detection.detect_keypoints_yolo <archive>.zarr --model <weights>.pt
   scripts/py -m fisheye.refinement.refine_keypoints <archive>.zarr
   ```
   The keypoint script now auto-applies refined ROI overrides when available.
   The refinement step also runs a coordinate-space audit by default and writes:
   `/tmp/<archive_name>_audit.json`.

   Optional overlap analysis and output-dir override:
   ```bash
   scripts/py -m fisheye.refinement.refine_keypoints <archive>.zarr \
     --post-overlap \
     --post-audit-output-dir /tmp
   ```
   This additionally writes `/tmp/<archive_name>_overlap.json`.

   Equivalent batch flags on `run_keypoints_batch`:
   - `--no-refine-post-audit` (disable default audit)
   - `--refine-post-overlap` (enable overlap report)
   - `--refine-post-audit-output-dir <dir>` (override `/tmp`)

2. Eye masks (defaults to refined keypoints when present):
   ```bash
   scripts/py -m fisheye.segmentation.eye_segmentation <archive>.zarr
   scripts/py -m fisheye.refinement.refine_eye_masks <archive>.zarr
   ```
   `refine_eye_masks` source keypoint resolution is strict by default:
   `--keypoint-run` override -> source lineage attrs (`source_keypoint_group` + canonical/legacy keypoint run attr) -> error.
   Use `--allow-latest-keypoint-fallback` only for temporary legacy recovery.
   Optional review/audit:
   ```bash
   scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --retune
   scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --manual
   scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --audit
   ```
   Full tuning/review checklist: `src/fisheye/docs/eye_mask_tuning_workflow.md`.

3. Arena assignment (if arenas align with refined detections):
   ```bash
   scripts/py -m fisheye.tracking.arena_assignment <archive>.zarr
   ```
   This current workflow also writes a matching `tracking_runs/<run>` entry for
   `single_subject_per_arena`, including `source_arena_assignment_run` and
   `tracking_qc_state`.

4. Track kinematics / downstream pipelines can now run without length mismatches.

## Provenance Validation

Two diagnostics help verify alignment:

* `scripts/py -m fisheye.diagnostics.check_provenance_consistency <archive>.zarr`
  * Compares row counts across detect, refined detect, crop, keypoint, arena-assignment, and tracking runs.
* `scripts/py -m fisheye.diagnostics.check_refined_roi_links <archive>.zarr`
  * Audits refined detection runs and ensures regenerated ROIs are discoverable from the crop side.

Expected healthy output shows consistent row counts (e.g., refined detections = crop ROIs = keypoint headings = arena-assignment rows = tracking rows) and no reported issues.

## Summary Checklist

1. `regenerate_interpolated_crops` with correct crop run.
2. Rerun YOLO keypoints and keypoint refinement.
3. Rerun arena assignment / tracking.
4. Rerun track kinematics or other downstream analysis.
5. Confirm with diagnostics.

## Detection Training Preflight

Before training a YOLO detection model, run the preflight builder to validate
crop provenance, bbox integrity, and downsample inputs while generating a
training config + manifest:

```bash
scripts/py -m fisheye.diagnostics.prepare_detect_training \
  <archive>.zarr \
  --source-type refined \
  --input-format gray \
  --out-config configs/fisheye/detect_config_<dataset>.yaml \
  --project runs/detect
```

Use `--dry-run` to print the generated config + manifest without writing files,
or `--provenance-policy strict` to fail if arena/camera metadata is missing.

Following these steps keeps provenance clean and prevents downstream stage
failures when refined detect state changes propagate downstream.
