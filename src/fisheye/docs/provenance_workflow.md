# Provenance-Safe Refinement Workflow

This note captures the sequencing we used to align refined detections, crops, keypoints, arena assignment, and tracking so downstream analysis (track kinematics, metrics, plots) stays consistent.

## Stage Overview

```
detect_runs → refined_detect_runs → crop_runs → keypoints_runs → refined_keypoints_runs → eye_masks_runs → refined_eye_masks_runs → arena_assignment_runs → tracking_runs → track_kinematics_runs
```

Relevant provenance attributes:

| Stage | Primary Array | Key Attrs |
| --- | --- | --- |
| `detect_runs/<run>` | `bbox_norm_coords` | `detect_timestamp_utc`, `total_detections` |
| `refined_detect_runs/<run>/interpolated` | `bbox_norm_coords`, `detection_source`, `reason_bytes` | `source_detect_run`, `interpolated_roi_path`, `detect_review_status` |
| `refined_detect_runs/<run>/<manual_group>` | `bbox_norm_coords`, `reason_bytes`, `reason`, `retune_id` | `manual_review_latest`, `detection_source_type`, `retune_base_group` |
| `crop_runs/<run>` | `roi_images` | `detection_source_path`, `detect_review_status_ref`, `detect_review_status` (snapshot), `detection_preferred_policy`, `crop_signature`, `crop_review_status` |
| `keypoints_runs/<run>` | `heading`, `frame_indices` | `source_crop_run` |
| `refined_keypoints_runs/<run>` | `heading`, `usable_keypoints`, `reason_bytes`, `reason` | `source_keypoints_run`, `source_crop_run`, `keypoint_signature`, `keypoint_review_status`, `reason_fallback_order` |
| `eye_masks_runs/<run>` | `masks_roi` | `source_crop_run`, `source_keypoint_group`, `source_keypoints_run` *(legacy alias: `source_keypoint_run`)* |
| `refined_eye_masks_runs/<run>` | `masks_roi`, `ellipse_params` | `source_eye_masks_run`, `source_keypoint_group`, `source_keypoints_run` *(legacy alias: `source_keypoint_run`)* |
| `arena_assignment_runs/<run>` | `arena_ids` | `source_detect_run`, `source_refined_run` |
| `tracking_runs/<run>` | `track_ids`, `track_arena_ids` | `source_detect_run`, `source_refined_run`, `source_arena_assignment_run`, `tracking_qc_state` |

Keypoint-lineage attribute contract for eye-mask stages:

- Canonical attr is `source_keypoints_run`.
- `source_keypoint_run` is a legacy compatibility alias and should not be used as the primary key in new tooling.
- Writers for new eye-mask/refined-eye-mask runs should write canonical lineage (and may mirror the legacy alias during migration).
- Readers/diagnostics resolve canonical first, then legacy alias fallback.
- `check_eye_masks` reports legacy-only lineage as `legacy` (warning) and missing/empty lineage as `incomplete`.

Eye-mask lineage arrays (`frame_indices`, `detection_indices`, `frame_counts`)
follow the contract in `docs/eye_mask_row_mapping_contract.md`:

- segmentation writes should anchor lineage to source crop runs;
- keypoint lineage arrays are used for cross-check/fallback compatibility;
- refinement copies lineage arrays from the source eye-mask run.

`bbox_norm_coords` in detect/refined-detect groups use normalized `[cx, cy, w, h]`.

For refined detect/keypoint reason labels, use fallback order:
`reason_bytes` -> `reason` -> labels derived from `detection_source`.

## Regenerating Interpolated Crops

When refined detections introduce interpolated boxes, we need matching ROI imagery.

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
  --source-type filtered \
  --input-format gray \
  --out-config configs/fisheye/detect_config_<dataset>.yaml \
  --project runs/detect
```

Use `--dry-run` to print the generated config + manifest without writing files,
or `--provenance-policy strict` to fail if arena/camera metadata is missing.

Following these steps keeps provenance clean and prevents downstream stage failures when interpolated detections are added. 
