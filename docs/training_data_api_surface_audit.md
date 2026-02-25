# Training Data API Surface Audit

Audit of the registry operations, CLI commands, and data schemas across the
three training data pipelines: **detection**, **keypoints**, and **eye masks**.

Generated 2025-02-25.

---

## 1. Shared Workflow Skeleton

All three pipelines follow the same six-step workflow:

| Step | Detection | Keypoints | Eye Masks |
|------|-----------|-----------|-----------|
| Profile backfill | `backfill_detection_profiles.py` | `backfill_keypoint_profiles.py` | `backfill_eye_mask_profiles.py` |
| Profile sync | `sync_detection_profile_registry.py` | `sync_keypoint_profile_registry.py` | `sync_eye_mask_profile_registry.py` |
| Prepare manifest | `prepare_detect_training_from_registry.py` | `prepare_keypoint_training_from_registry.py` | `prepare_eye_mask_training_from_registry.py` |
| Export training zarr | `export_detect_training_zarr.py` | `export_keypoint_training_zarr.py` | `export_eye_mask_training_zarr.py` |
| Validate training zarr | `validate_detect_training_zarr.py` | `validate_keypoint_training_zarr.py` | `validate_eye_mask_training_zarr.py` |
| Aggregate data card | `aggregate_detection_training_data_card.py` | `aggregate_keypoint_training_data_card.py` | `aggregate_eye_mask_training_data_card.py` |
| Plot data card | `plot_detection_training_data_card.py` | `plot_keypoint_training_data_card.py` | `plot_eye_mask_training_data_card.py` |

All utilities live under `src/fisheye/utils/` and use `argparse` with the
entry-point pattern `if __name__ == "__main__": raise SystemExit(main())`.

---

## 2. Shared Composition Filters

Every `prepare_*_training_from_registry.py` accepts the same hardware and
recording filters:

- `--dish-design`, `--dish-design-like`
- `--camera-id`, `--camera-model`, `--camera-serial`
- `--rig-id`, `--arena-id`
- `--fps-min`, `--fps-max`
- `--exposure-min`, `--exposure-max`
- `--frame-rate-min`, `--frame-rate-max`
- `--gain-min`, `--gain-max`
- `--video-codec`, `--video-pix-fmt`
- `--format-encoder`, `--format-title`, `--format-comment`
- `--encoder-name`, `--encoder-codec`, `--encoder-preset`, `--encoder-tuning`, `--encoder-rc`
- `--compression`
- `--path-contains`, `--limit`

All three also accept `--registry` (auto-detected from env if omitted) and
`--set-name` / `--set-version` for training set naming.

---

## 3. Registry Table Pattern

Each data type has three core tables plus associated views:

| Table pattern | Detection | Keypoints | Eye Masks |
|---------------|-----------|-----------|-----------|
| Data profile | `detection_data_profile` | `keypoint_data_profile` | `eye_mask_data_profile` |
| Quality | `detect_quality` | `keypoint_quality` | `eye_mask_quality` |
| Performance | `detect_performance` | `keypoint_performance` | `eye_mask_performance` |
| Profile latest view | `detection_data_profile_latest` | `keypoint_data_profile_latest` | `eye_mask_data_profile_latest` |
| Quality current view | `detect_quality_current` | `keypoint_quality_current` | `eye_mask_quality_current` |
| Performance latest view | `detect_performance_latest` | `keypoint_performance_latest` | `eye_mask_performance_latest` |

Primary key for all profile tables: `(dataset_id, profile_run)`.

Registry operations per table follow the same verb set:

- `upsert_*()` — insert or update a single row
- `replace_*()` — bulk replace all rows for a dataset_id
- `refresh_*_for_dataset()` — re-extract from zarr and replace
- `query_*_latest()` — read latest rows via the `_latest` view

---

## 4. Profile Schema Names

| Data type | Schema name | Version |
|-----------|-------------|---------|
| Detection | `detection_dataset_profile` | v1 |
| Keypoints | `keypoint_dataset_profile` | v1 |
| Eye Masks | `eye_mask_dataset_profile` | v1 |

All profiles are serialized as JSON in a `profile_json` column and also
projected into typed columns for indexed queries.

---

## 5. Shared Composition Fields in Profile Tables

All three profile tables include the same composition columns:

- `rig_id`, `camera_id`, `arena_id`
- `dish_design`, `canvas_name`, `protocol_name`
- `genotype`, `dpf_at_acquisition`

---

## 6. Divergences: CLI Arguments

### 6a. Quality gate threshold

Each pipeline uses a differently-named argument for the minimum quality
threshold when filtering datasets for training:

| Pipeline | Argument | Semantics |
|----------|----------|-----------|
| Detection | `--max-interpolated-detections-rate` | Upper bound on interpolation fraction |
| Keypoints | `--min-usable-keypoints-rate` | Lower bound on usable keypoint fraction |
| Eye Masks | `--min-usable-rate` | Lower bound on usable row fraction |

Detection's gate is an upper bound (max bad), while keypoints and eye masks
use lower bounds (min good). The argument names are also inconsistent.

### 6b. Source selection

Each pipeline has a different mechanism for selecting which upstream run to
use as training source:

| Pipeline | Argument(s) | Options |
|----------|-------------|---------|
| Detection | `--source-type` | `detect`, `filtered`, `interpolated`, `manual` |
| Keypoints | `--keypoint-run` | `latest_traditional`, `latest_yolo`, explicit name |
| Eye Masks | `--eye-stage` + `--eye-run` | `auto`, `eye_masks_runs`, `refined_eye_masks_runs` + explicit run |

### 6c. Split configuration location

| Pipeline | Where splits are configured |
|----------|-----------------------------|
| Detection | `export_detect_training_zarr.py` only (`--split`) |
| Keypoints | `export_keypoint_training_zarr.py` only |
| Eye Masks | Both `prepare_eye_mask_training_from_registry.py` and `export_eye_mask_training_zarr.py` (`--split-train`, `--split-val`, `--split-test`, `--split-seed`) |

Detection uses a single `--split` string (e.g., `0.8/0.2`). Eye masks uses
separate `--split-train`, `--split-val`, `--split-test` float arguments.

### 6d. Input format argument

| Pipeline | Argument | Options |
|----------|----------|---------|
| Detection | `--input-format` + `--model-input` | `gray`, `rgb` |
| Keypoints | (inferred from config) | — |
| Eye Masks | `--input-format` | `gray`, `rgb` |

Detection has both `--input-format` (source) and `--model-input` (target);
eye masks has only `--input-format`; keypoints infers format from config.

### 6e. Label mode (eye masks only)

Eye masks has `--label-mode` (`lr` for left/right or `union`) with no
equivalent in detection or keypoints.

### 6f. Provenance and lineage policy (detection only)

Detection's prepare script includes `--provenance-policy`
(`warn`/`strict`/`ignore`) and the data card aggregator includes
`--subject-lineage-policy` (`warn`/`require`/`ignore`). Neither keypoints
nor eye masks have these.

---

## 7. Divergences: Registry Schema

### 7a. Review fields

| Field | Detection | Keypoints | Eye Masks |
|-------|-----------|-----------|-----------|
| `review_state` | Yes | Yes | Yes |
| `review_method` | Yes | Yes | Yes |
| `review_intended_use` | Yes | Yes | Yes |
| `review_timestamp_utc` | Yes | Yes | Yes |
| `review_reviewer` | No | Yes (in quality) | Yes |
| `review_notes` | No | Yes (in quality) | Yes |

Eye masks and keypoints have `review_reviewer` and `review_notes` in their
quality tables. Detection does not.

### 7b. Staleness tracking (eye masks only)

The eye mask profile, quality, and performance tables include:

- `source_keypoint_stale_state`
- `source_keypoint_stale_reason`
- `source_keypoint_stale_timestamp_utc`
- `source_keypoint_stale_json`

Detection and keypoints have no staleness tracking.

### 7c. Lifecycle fields (eye masks only)

Eye mask quality and performance tables include:

- `lifecycle_state`
- `lifecycle_reason`

Not present in detection or keypoints.

### 7d. Stage group (eye masks only)

Eye mask tables include a `stage_group` column
(`eye_masks_runs` or `refined_eye_masks_runs`) to distinguish raw from
refined masks. Detection and keypoints do not have this concept in their
profile tables (detection distinguishes source type via `detection_source_type`;
keypoints uses `keypoint_method`).

### 7e. Geometry metrics

Each profile table stores domain-specific geometry percentiles:

**Detection** (`detection_data_profile`):
- `w_norm`, `h_norm`, `area_norm`, `aspect_ratio` (p1, p5, p10, p25, p50, p75, p90, p95, p99)
- `edge_proximity_rate`
- Detection coverage: `frames_with_detections`, `frames_zero_detections`, `detections_per_frame` stats
- Center heatmap (32x32 grid)

**Keypoints** (`keypoint_data_profile`):
- `triangle_area`, `min_angle`, `heading` (p10, p50, p90)
- `confidence_valid_rate`, `geometry_valid_rate`
- `rows_total`, `rows_usable`, `usable_keypoints_total`, `usable_rate`
- `skeleton_id`, `kpt_shape`

**Eye Masks** (`eye_mask_data_profile`):
- `area`, `left_area`, `right_area`, `union_area`, `area_lr_ratio` (p10, p50, p90)
- `major_axis`, `minor_axis`, `aspect_ratio`, `eye_separation` (p10, p50, p90)
- `ellipse_success_rate`, `pair_success_rate`
- `edge_proximity_rate`
- `reviewed_rate`, `excluded_rate`, `exclusion_reasons_json`
- `rows_total`, `rows_usable`, `usable_rate`

Detection uses 9 percentile tiers (p1 through p99). Keypoints and eye masks
use 3 (p10, p50, p90).

---

## 8. Divergences: Auxiliary CLIs

### 8a. Full pipeline orchestrator

| Pipeline | Script | Present? |
|----------|--------|----------|
| Detection | `run_detect_training_pipeline.py` | Yes |
| Keypoints | `run_keypoint_training_pipeline.py` | Yes |
| Eye Masks | (none) | **Missing** |

Detection and keypoints have an orchestrator that chains
prepare -> export -> aggregate data card -> (optionally) train.
Eye masks does not.

### 8b. Training config audit

| Pipeline | Script | Present? |
|----------|--------|----------|
| Detection | `check_detect_training_config.py` | Yes |
| Keypoints | (none) | **Missing** |
| Eye Masks | (none) | **Missing** |

### 8c. Model resolution CLI

| Pipeline | Resolve model | Run with registry model |
|----------|---------------|-------------------------|
| Detection | `resolve_detect_model.py` (standalone) | `run_detect_with_registry_model.py` |
| Keypoints | (bundled) | `run_keypoints_with_registry_model.py` |
| Eye Masks | (bundled) | `run_eye_masks_with_registry_model.py` |

Detection has a separate `resolve_detect_model.py` for model lookup without
inference. Keypoints and eye masks bundle resolution into the run script.

### 8d. Finalize artifacts CLI

| Pipeline | Script | Present? |
|----------|--------|----------|
| Detection | (none) | **Missing** |
| Keypoints | `finalize_keypoint_refinement_artifacts.py` | Yes |
| Eye Masks | `finalize_eye_mask_profile_artifacts.py` | Yes |

### 8e. Quality overview export

| Pipeline | Script | Present? |
|----------|--------|----------|
| Detection | `export_detect_quality_overview.py` | Yes |
| Keypoints | `export_keypoint_quality_overview.py` | Yes |
| Eye Masks | `export_eye_mask_quality_overview.py` | Yes |

All three are present.

### 8f. Review management CLIs

| Capability | Detection | Keypoints | Eye Masks |
|------------|-----------|-----------|-----------|
| Set review status | `set_detect_review_status.py` | `set_keypoint_review_status.py` | (via `review_eye_masks_batch.py`) |
| Accept review | `accept_detect_review.py` | `accept_keypoint_review.py` | (via `review_eye_masks_batch.py`) |
| Show review status | (via registry query) | `show_keypoint_review_status.py` | (via registry query) |
| Batch review | (via `detect_quality_batch.py`) | `review_keypoints_batch.py` | `review_eye_masks_batch.py` |

Detection and keypoints have separate set/accept/show CLIs. Eye masks
combines these into a single `review_eye_masks_batch.py`.

### 8g. Batch inference CLIs

| Pipeline | Script |
|----------|--------|
| Detection | (detection is a pipeline stage, not a standalone batch CLI) |
| Keypoints | `run_keypoints_batch.py` |
| Eye Masks | `run_eye_masks_batch.py` |

### 8h. Backfill utilities (beyond profiles)

| Utility | Detection | Keypoints | Eye Masks |
|---------|-----------|-----------|-----------|
| Review status backfill | `backfill_detect_review_status.py` | (none) | (none) |
| Lineage attrs backfill | (none) | (none) | `backfill_eye_mask_lineage_attrs.py` |
| Confidence backfill | (none) | `backfill_keypoint_confidences.py` | (none) |
| Heading fields backfill | (none) | `backfill_keypoint_heading_fields.py` | (none) |
| Label names backfill | (none) | `backfill_keypoint_label_names.py` | (none) |
| Reason bytes backfill | (none) | `backfill_keypoint_reason_bytes.py` | (none) |
| Stale resolution | (none) | (none) | `resolve_eye_mask_stale.py` |

---

## 9. Registry Maintenance (maintenance.py)

The unified maintenance CLI supports backfill and refresh for all three
types via flags:

| Operation | Detection | Keypoints | Eye Masks |
|-----------|-----------|-----------|-----------|
| Backfill quality | `--backfill-detect-quality` | `--backfill-keypoint-quality` | `--backfill-eye-mask-quality` |
| Backfill performance | `--backfill-detect-performance` | `--backfill-keypoint-performance` | `--backfill-eye-mask-performance` |
| Backfill profiles | (via standalone CLI) | `--backfill-keypoint-profiles` | `--backfill-eye-mask-profiles` |
| Refresh quality | `--refresh-detect-quality` | `--refresh-keypoint-quality` | `--refresh-eye-mask-quality` |
| Refresh performance | `--refresh-detect-performance` | `--refresh-keypoint-performance` | `--refresh-eye-mask-performance` |
| Refresh profiles | (via standalone CLI) | `--refresh-keypoint-profiles` | `--refresh-eye-mask-profiles` |

Detection profile backfill/refresh is handled by its standalone
`backfill_detection_profiles.py` rather than maintenance flags. Keypoints
and eye masks support both standalone CLIs and maintenance flags.

---

## 10. Registry Query Surface (check_training_registry.py)

The `check_training_registry.py` unified viewer supports these views:

| View | Detection | Keypoints | Eye Masks |
|------|-----------|-----------|-----------|
| Quality | `--show-detect-quality` | `--show-keypoint-quality` | `--show-eye-mask-quality` |
| Performance | `--show-detect-performance` | `--show-keypoint-performance` | `--show-eye-mask-performance` |
| Profile | `--show-detect-profile` | `--show-keypoint-profile` | `--show-eye-mask-profile` |

---

## 11. Data Card Schema Differences

### Detection data card includes:
- Composition breakdown (rig, camera, arena, dish, canvas, protocol)
- Genotype distribution histogram
- DPF histogram
- Bbox geometry histograms (w_norm, h_norm, area_norm, aspect_ratio)
- Subject lineage coverage assessment
- Detection source type breakdown

### Keypoints data card includes:
- Composition breakdown (same fields)
- Geometry distributions (triangle_area, min_angle, heading)
- Usable rate per dataset
- Spatial summary and train/val parity metrics
- Skeleton ID and kpt_shape

### Eye masks data card includes:
- Composition breakdown (same fields)
- Geometry distributions (area, left/right area, union area, lr ratio, axes, eye separation)
- Usable rate and review state distribution
- Ellipse/pair success rates
- Exclusion reason breakdown

---

## 12. Resolved Source Dataclass Pattern

Each profile module defines a frozen dataclass for resolved source metadata:

| Pipeline | Class | Key fields |
|----------|-------|------------|
| Detection | `ResolvedDetectionSource` | `detection_path`, `detection_type`, `detect_run`, `refined_run`, `manual_group` |
| Keypoints | `ResolvedKeypointSource` | `keypoint_path`, `keypoint_type`, `keypoint_method`, `keypoint_run`, `refined_run`, `skeleton_id`, `kpt_shape` |
| Eye Masks | `ResolvedEyeMaskSource` | `eye_mask_path`, `stage_group`, `eye_mask_run`, `eye_mask_method`, `source_keypoint_group`, `source_keypoint_run`, `source_crop_run` |

Each also has a `*ProfileWriteResult` dataclass:

| Pipeline | Class | Key fields |
|----------|-------|------------|
| Detection | `DetectionProfileWriteResult` | `run_name`, `source_detection_path`, `source_detection_type`, `profile_summary` |
| Keypoints | `KeypointProfileWriteResult` | `run_name`, `source_keypoint_path`, `source_keypoint_method`, `profile_summary` |
| Eye Masks | `EyeMaskProfileWriteResult` | `run_name`, `source_eye_mask_path`, `source_eye_mask_method`, `profile_summary` |

---

## 13. Zarr Group Hierarchy

### Detection
```
root/
  detect_runs/<run>/
  refined_detect_runs/<run>/
  analysis/detection_profile_runs/<run>/profile_summary
```

### Keypoints
```
root/
  keypoints_runs/<run>/
  refined_keypoints_runs/<run>/   (or keypoints_refined_runs/<run>/)
  analysis/keypoint_profile_runs/<run>/profile_summary
```

### Eye Masks
```
root/
  eye_masks_runs/<run>/
  refined_eye_masks_runs/<run>/
  analysis/eye_mask_profile_runs/<run>/profile_summary
```

The keypoints pipeline has a legacy alias (`keypoints_refined_runs`) handled
via the `REFINED_PARENT_NAMES` tuple. Detection and eye masks do not have
this ambiguity.
