# Schema Audit TODO

## Goal
Inventory where explicit schemas exist, where they are missing, and where they
are stale or incomplete. Covers zarr structure, registry database tables, data
contracts, and Python type definitions.

**Priority focus**: Full pipeline from video import through eye mask
refinement -- every stage that touches the recording zarr.

## Audit date: 2026-02-27

---

## 1. Pipeline Stage Schema Coverage

The table below maps every processing stage to its schema assets. A check mark
means the asset exists and covers this stage; a dash means it does not.

| # | Stage | Writer module | Zarr group | `zarr_structure.md` | `ZARR_SCHEMA` dict (schema.py) | `REQUIRED_ARRAYS` checker | Contract doc | Code-level `ArraySpec` |
|---|-------|--------------|------------|---------------------|-------------------------------|--------------------------|-------------|----------------------|
| 1 | Video import | `capture/import_video.py` | `raw_video/` | YES | arrays listed | -- | -- | -- |
| 2 | Background | `preprocessing/background.py` | `background_runs/` | YES | group only | -- | -- | -- |
| 3 | Detection | `detection/detect_yolo.py` | `detect_runs/` | YES | group only | `check_detection_runs.py` (7 of ~10 arrays) | `detect_batch_analysis_zarr_parallel_agents_contract.md` | -- |
| 4 | Detect quality | `refinement/detect_quality.py` | `detect_runs/.../quality_reports/` | nested, no section | -- | -- | `detect_quality_parallel_agents_contract.md` | -- |
| 5 | Detect refinement | `refinement/refine_detect.py` | `refined_detect_runs/` | YES | attrs listed | `list_incomplete_refined_detect_groups.py` (2 arrays only) | `recording_analysis_pipeline_contract.md` | -- |
| 6 | Crop | `tracking/crop.py` | `crop_runs/` | YES | attrs listed | `check_crop_runs.py` (7 of ~9 arrays) | -- | -- |
| 7 | Keypoints | `detection/detect_keypoints_yolo.py` | `keypoints_runs/` | YES | group only | -- | `detect_keypoints_parity_contract.md` | -- |
| 8 | Keypoint refinement | `refinement/refine_keypoints.py` | `refined_keypoints_runs/` | YES | attrs listed | -- | `keypoint_late_correction_contract.md` | -- |
| 9 | Eye masks | `segmentation/eye_segmentation_yolo.py` | `eye_masks_runs/` | YES | group only | -- | `eye_mask_parity_parallel_agents_contract.md` | -- |
| 10 | Eye mask refinement | `refinement/refine_eye_masks.py` | `refined_eye_masks_runs/` | YES | group only | -- | `eye_mask_training_artifact_contract.md` | -- |
| 11 | ID assignment | `tracking/assign_ids.py` | `id_assignment_runs/` | YES | group only | -- | -- | -- |
| 12 | Stimulus import | `analysis/import_stimulus_to_zarr.py` | `analysis/stimulus_runs/` | YES | -- | -- | `analysis_zarr_creation_contract.md` | -- |

**Key observations:**

- `zarr_structure.md` covers every stage. It is comprehensive and accurate.
- `schema.py`'s `ZARR_SCHEMA` dict only lists group names and some attrs. It
  has **zero array definitions** for any stage except `raw_video`.
- Only 3 of 12 stages have a `REQUIRED_ARRAYS` diagnostic checker, and those
  checkers cover a subset of the documented arrays.
- The "Code-level ArraySpec" column is empty for every stage. This is the gap
  that `stage_arrays.py` would fill.

---

## 2. Per-Stage Array Inventories

Definitive array lists from the stage writers (the source of truth for what
actually gets written). These should be transcribed into `stage_arrays.py`.

### Stage 1: Video Import (`raw_video/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `images_full` | `(n_frames, H, W)` | uint8 | optional |
| `images_ds` | `(n_frames, H_ds, W_ds)` | uint8 | optional (at least one of full/ds) |
| `images_ds_rgb` | `(n_frames, H_ds, W_ds, 3)` | uint8 | optional |
| `original_frame_indices` | `(n_import_frames,)` | int32 | only when frame_step > 1 |

### Stage 2: Background (`background_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `background_full` | `(H, W)` | uint8 | optional |
| `background_ds` | `(H_ds, W_ds)` | uint8 | optional |
| `frame_indices` | `(n_samples,)` | int32 | yes |

### Stage 3: Detection (`detect_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `frame_indices` | `(n_detections,)` | int32 | yes |
| `bbox_norm_coords` | `(n_detections, 4)` | float32 | yes |
| `scores` | `(n_detections,)` | float32 | yes |
| `class_ids` | `(n_detections,)` | int32 | yes |
| `frame_counts` | `(n_frames,)` | int32 | yes |
| `n_detections` | `(n_frames,)` | int32 | yes (legacy alias of frame_counts) |
| `centers_px` | `(n_detections, 2)` | float32 | blob method only |

### Stage 4: Detect Quality (`detect_runs/<run>/quality_reports/<qrun>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `quality_flags` | `(n_frames,)` | int8 | yes |
| `detection_quality_labels` | `(n_detections,)` | int8 | yes |

### Stage 5: Detect Refinement (`refined_detect_runs/<run>/{filtered,interpolated,manual}/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `bbox_norm_coords` | `(n_refined, 4)` | float32 | yes |
| `scores` | `(n_refined,)` | float32 | yes |
| `frame_indices` | `(n_refined,)` | int32 | yes |
| `class_ids` | `(n_refined,)` | int32 | yes |
| `frame_counts` | `(n_frames,)` | int32 | yes |
| `n_detections` | `(n_frames,)` | int32 | yes (legacy alias) |
| `detection_source` | `(n_refined,)` | int8 | yes (0=real, 1=interpolated) |
| `reason_bytes` | `(n_refined, width)` | uint8 | yes |
| `reason` | `(n_refined,)` | string | yes |
| `frame_mapping` | `(n_refined,)` | int32 | legacy alias of frame_indices |

### Stage 6: Crop (`crop_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `roi_images` | `(n_rois, h, w)` | uint8 | yes |
| `roi_coordinates_full` | `(n_rois, 2)` | int32 | yes |
| `roi_coordinates_ds` | `(n_rois, 2)` | int32 | yes |
| `bbox_norm_coords` | `(n_rois, 4)` | float32 | yes |
| `frame_indices` | `(n_rois,)` | int32 | yes |
| `frame_counts` | `(n_frames,)` | int32 | yes |
| `detection_indices` | `(n_rois,)` | int32 | yes |
| `detection_source` | `(n_rois,)` | int8 | yes |

### Stage 7: Keypoints (`keypoints_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `frame_indices` | `(n_rois,)` | int32 | yes |
| `frame_counts` | `(n_frames,)` | int32 | yes |
| `n_rois` | `(n_frames,)` | int32 | legacy alias |
| `detection_indices` | `(n_rois,)` | int32 | yes |
| `keypoints_roi` | `(n_rois, 3, 2)` | float64 | yes |
| `keypoints_img` | `(n_rois, 3, 2)` | float64 | yes |
| `keypoints_norm` | `(n_rois, 3, 2)` | float64 | yes |
| `heading` | `(n_rois,)` | float64 | yes |
| `confidence` | `(n_rois,)` | float64 | yes |
| `keypoint_confidences` | `(n_rois, 3)` | float64 | yes |
| `effective_threshold` | `(n_rois,)` | float64 | yes |
| `effective_se2_radius` | `(n_rois,)` | float64 | yes |
| `detection_success` | `(n_rois,)` | bool | yes |
| `detection_source` | `(n_rois,)` | int8 | yes |
| `heading_finite` | `(n_rois,)` | bool | yes |
| `heading_usable` | `(n_rois,)` | bool | yes |
| `n_keypoints` | `(n_frames,)` | int32 | yes |
| `triangle_angles` | `(n_rois, 3)` | float64 | yes |
| `triangle_angles_raw` | `(n_rois, 3)` | float64 | yes |
| `triangle_area` | `(n_rois,)` | float64 | yes |

### Stage 8: Keypoint Refinement (`refined_keypoints_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `frame_indices` | `(n_rois,)` | int32 | yes |
| `frame_counts` | `(n_frames,)` | int32 | yes |
| `n_rois` | `(n_frames,)` | int32 | legacy alias |
| `detection_indices` | `(n_rois,)` | int32 | yes |
| `detection_source` | `(n_rois,)` | int8 | yes |
| `retune_id` | `(n_rois,)` | int32 | yes (-1 = none) |
| `keypoints_roi` | `(n_rois, 3, 2)` | float64 | yes |
| `keypoints_img` | `(n_rois, 3, 2)` | float64 | yes |
| `keypoints_norm` | `(n_rois, 3, 2)` | float64 | yes |
| `heading` | `(n_rois,)` | float64 | yes |
| `confidence` | `(n_rois,)` | float64 | yes |
| `keypoint_confidences` | `(n_rois, 3)` | float64 | yes |
| `effective_threshold` | `(n_rois,)` | float64 | when present in source |
| `effective_se2_radius` | `(n_rois,)` | float64 | when present in source |
| `triangle_area` | `(n_rois,)` | float64 | yes |
| `min_angle` | `(n_rois,)` | float64 | yes |
| `triangle_angles` | `(n_rois, 3)` | float64 | yes |
| `quality_labels` | `(n_rois,)` | int8 | yes |
| `refined_success` | `(n_rois,)` | bool | yes |
| `source_success` | `(n_rois,)` | bool | yes |
| `flip_corrected` | `(n_rois,)` | bool | yes |
| `heading_finite` | `(n_rois,)` | bool | yes |
| `heading_usable` | `(n_rois,)` | bool | yes |
| `confidence_valid` | `(n_rois,)` | bool | yes |
| `geometry_valid` | `(n_rois,)` | bool | yes |
| `usable_keypoints` | `(n_rois,)` | bool | yes |
| `reason_bytes` | `(n_rois, width)` | uint8 | yes |
| `reason` | `(n_rois,)` | string | yes |
| `failure_indices` | `(n_failures,)` | int32 | yes |

### Stage 9: Eye Masks (`eye_masks_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `masks_roi` | `(n_rois, 2, H, W)` | uint8 | yes |
| `mask_probs_roi` | `(n_rois, 2, H, W)` | float16/float32 | optional |
| `mask_scores` | `(n_rois,)` | float32 | optional |
| `ellipse_params` | `(n_rois, 2, 5)` | float32 | yes |
| `ellipse_success` | `(n_rois, 2)` | bool | yes |
| `eye_separation` | `(n_rois,)` | float32 | yes |
| `detection_source` | `(n_rois,)` | int8 | yes |
| `contour_left_ptr` | `(n_rois,)` | int32 | yes |
| `contour_left_len` | `(n_rois,)` | int32 | yes |
| `contour_right_ptr` | `(n_rois,)` | int32 | yes |
| `contour_right_len` | `(n_rois,)` | int32 | yes |
| `contours_left` | `(n_points, 2)` | float32 | yes |
| `contours_right` | `(n_points, 2)` | float32 | yes |
| `reason` | `(n_rois,)` | string | yes |

### Stage 10: Eye Mask Refinement (`refined_eye_masks_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `masks_roi` | `(n_rois, 2, H, W)` | uint8 | yes |
| `ellipse_params` | `(n_rois, 2, 5)` | float32 | yes |
| `ellipse_success` | `(n_rois, 2)` | bool | yes |
| `eye_separation` | `(n_rois,)` | float32 | yes |
| `retune_id` | `(n_rois,)` | int32 | optional |
| `mask_probs_roi_refined` | `(n_rois, 2, H, W)` | float16 | optional |
| `contour_left_ptr` | `(n_rois,)` | int32 | yes |
| `contour_left_len` | `(n_rois,)` | int32 | yes |
| `contour_right_ptr` | `(n_rois,)` | int32 | yes |
| `contour_right_len` | `(n_rois,)` | int32 | yes |
| `contours_left` | `(n_points, 2)` | float32 | yes |
| `contours_right` | `(n_points, 2)` | float32 | yes |

**Metrics subgroup** (`refined_eye_masks_runs/<run>/metrics/`):
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `area_refined` | `(n_rois, 2)` | float32 | yes |
| `area_source` | `(n_rois, 2)` | float32 | yes |
| `area_zscore` | `(n_rois, 2)` | float32 | yes |
| `area_delta_vs_source` | `(n_rois, 2)` | float32 | yes |
| `centroid_error` | `(n_rois, 2)` | float32 | yes |
| `symmetry_offsets` | `(n_rois, 2)` | float32 | yes |
| `separation_refined` | `(n_rois,)` | float32 | yes |
| `axis_ratio` | `(n_rois, 2)` | float32 | yes |
| `circularity` | `(n_rois, 2)` | float32 | yes |
| `connectivity_flags` | `(n_rois,)` | uint8 | yes |
| `smoothing_flags` | `(n_rois, 2)` | uint8 | yes |
| `pixels_reassigned` | `(n_rois,)` | int32 | yes |
| `probabilities_used` | `(n_rois,)` | bool | yes |
| `filter_flags` | `(n_rois, 2)` | uint8 | yes |
| `reason` | `(n_rois,)` | string | yes |

### Stage 11: ID Assignment (`id_assignment_runs/<run>/`)
| Array | Shape | DType | Required |
|-------|-------|-------|----------|
| `detection_ids` | `(n_detections,)` | int32 | yes |
| `confidence` | `(n_detections,)` | float32 | yes |

---

## 3. Existing Schema Assets (summary)

### Zarr schema

| Asset | Location | Role |
|-------|----------|------|
| `zarr_structure.md` | `src/fisheye/docs/zarr_structure.md` (846 lines) | **Self-declared authoritative spec.** Comprehensive per-stage array/attr docs. |
| `ZARR_SCHEMA` dict | `src/fisheye/shared/zarr/schema.py` | Code-level schema. Lists groups, root attrs, some run-level attrs. **No array specs.** |
| `validate_zarr_structure()` | `src/fisheye/shared/zarr/schema.py:636` | Shallow: group existence + root attrs only. |
| `create_palette_zarr()` | `src/fisheye/shared/zarr/schema.py:136` | Creates initial group tree + root attrs. |
| Per-stage `REQUIRED_ARRAYS` | `diagnostics/check_detection_runs.py`, `check_crop_runs.py`, `list_incomplete_refined_detect_groups.py` | Hardcoded partial checklists. 3 of 12 stages covered. |
| Training zarr validators | `utils/validate_{detect,keypoint,eye_mask}_training_zarr.py` | Per-task export validation. Bespoke, no shared framework. |

### Registry database

| Asset | Location | Role |
|-------|----------|------|
| Migration system (v1-27) | `db.py` lines 1808-1877 | 33 tables, ~35 views, 50+ indexes, FKs + cascades. |
| CHECK constraints | `recording_step_status.status` | Only enum column with enforcement. |
| Triggers | `trg_dataset_lineage_no_self_*` | Self-edge prevention. |
| No standalone documentation | -- | Schema only readable in 6800-line migration code. |

### Contracts

24+ contract documents in `docs/`. See coverage in the stage table (Section 1).
No versioning metadata. No code-to-contract linking.

### Python type definitions

`PoseSchema` (pose), `RefinedDetectResolution`, `ExperimentSetupInfo`,
`MaskBundle` (masks), Pydantic training configs. No typed per-stage array specs.

---

## 4. Known Gaps

### Per-stage
- [ ] `ZARR_SCHEMA` dict has zero array definitions for any stage except
      `raw_video`. Group names + attrs only.
- [ ] `validate_zarr_structure()` does not validate per-stage arrays, dtypes,
      row alignment, or conditional requirements.
- [x] Stage-specific diagnostics now consume canonical `StageSpec` definitions
      from `stage_arrays.py` for stages 1-11.
- [ ] No dedicated diagnostic checker for Stage 12 stimulus import
      (`analysis/stimulus_runs`) yet.
- [ ] Detect quality arrays (`quality_reports/` subgroup) have no dedicated
      section in `zarr_structure.md`.

### Cross-cutting
- [ ] Array expectations defined independently in writer, checker, docs, and
      training export validator. No mechanism to detect drift.
- [x] Code-importable schema exists for stages 1-11 in
      `src/fisheye/shared/zarr/stage_arrays.py`.
- [ ] `schema.py` remains stale for per-stage array definitions and does not
      yet delegate to `stage_arrays.py`.
- [ ] No schema documentation for the registry database outside `db.py` itself.
- [ ] Contracts have no version or freshness metadata.

---

## 5. Implementation Plan

### Priority: stage_arrays.py -- the single source of truth

Create `src/fisheye/shared/zarr/stage_arrays.py` containing the canonical array
spec for every pipeline stage as frozen dataclasses. This is the keystone that
makes everything else cheaper.

```python
@dataclass(frozen=True)
class ArraySpec:
    name: str
    dtype: str
    shape_template: str   # e.g. "(n_detections, 4)"
    required: bool = True
    description: str = ""

@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    zarr_group: str       # e.g. "detect_runs"
    specs: Tuple[ArraySpec, ...]
    subgroups: Dict[str, Tuple[ArraySpec, ...]] = field(default_factory=dict)
```

#### Phase 1: Define specs for the core pipeline (transcription work)
- [x] `RAW_VIDEO_ARRAYS` -- Stage 1
- [x] `BACKGROUND_ARRAYS` -- Stage 2
- [x] `DETECT_ARRAYS` -- Stage 3
- [x] `DETECT_QUALITY_ARRAYS` -- Stage 4
- [x] `REFINED_DETECT_ARRAYS` -- Stage 5 (with `filtered/`, `interpolated/`
      subgroup specs)
- [x] `CROP_ARRAYS` -- Stage 6
- [x] `KEYPOINT_ARRAYS` -- Stage 7
- [x] `REFINED_KEYPOINT_ARRAYS` -- Stage 8
- [x] `EYE_MASK_ARRAYS` -- Stage 9
- [x] `REFINED_EYE_MASK_ARRAYS` -- Stage 10 (with `metrics/` subgroup spec)
- [x] `ID_ASSIGNMENT_ARRAYS` -- Stage 11

Source: the array inventories in Section 2 above, cross-checked against
`zarr_structure.md` and the stage writer code.

#### Phase 2: Generic validation function
- [x] `validate_run(group, stage_spec) -> ValidationResult` that checks array
      presence, dtype, shape rank, and leading-dimension alignment.
- [x] Tests: synthetic zarr validation coverage added for detect, crop,
      keypoints, and eye masks, including missing required arrays, dtype
      mismatches, and optional-array warning behavior.

#### Phase 3: Migrate existing checkers to use stage_arrays.py
- [x] `check_detection_runs.py` -- replace hardcoded `REQUIRED_ARRAYS` with
      import from `DETECT_ARRAYS`.
- [x] `check_crop_runs.py` -- replace with `CROP_ARRAYS`.
- [x] `list_incomplete_refined_detect_groups.py` -- replace with
      `REFINED_DETECT_ARRAYS`.

#### Phase 4: Add checkers for uncovered stages
- [x] Background checker (or add to `validate_zarr_structure`)
- [x] Detect quality checker
- [x] Keypoint checker
- [x] Keypoint refinement checker
- [x] Eye mask checker
- [x] Eye mask refinement checker
- [x] ID assignment checker

#### Phase 5: Reconcile zarr_structure.md
- [x] Diff `stage_arrays.py` specs against `zarr_structure.md` and resolve
      discrepancies (missing arrays, dtype mismatches, optional vs required).
      Completed 2026-02-27: reconciled `raw_video`, `detect`, `crop`,
      `keypoints`, `eye_masks`, and `id_assignment` array tables.
- [x] Update `zarr_structure.md` header to reference `stage_arrays.py` as the
      code-level counterpart. Completed 2026-02-27.
- [x] Add detect quality section to `zarr_structure.md` (currently missing).
      Completed 2026-02-27: added
      `detect_runs/<run>/quality_reports/<qrun>/` section.

#### Phase 6: Retire stale schema.py artifacts
- [x] Remove or mark deprecated: `create_detection_arrays()` (creates 2 of 10
      arrays), `create_tracking_arrays()` (legacy 21-column layout).
      Completed 2026-02-27: both helpers now emit `DeprecationWarning` with
      migration guidance to `stage_arrays.py`.
- [x] Update `ZARR_SCHEMA` dict to reference `stage_arrays.py` specs or
      remove it in favor of the new module.
      Completed 2026-02-27: `ZARR_SCHEMA` now includes
      `array_contract_source=fisheye.shared.zarr.stage_arrays` and a
      legacy-status marker.

### Lower priority (not blocked on stage_arrays.py)

#### Registry data dictionary
- [ ] Script to auto-generate `docs/registry_schema_reference.md` from live DB
      via `PRAGMA table_info()`, `PRAGMA foreign_key_list()`,
      `PRAGMA index_list()`.
- [ ] Include column types, constraints, FK relationships, and index coverage.

#### Contract versioning
- [ ] Add version/status/last-verified header block to each `*_contract.md`.
- [ ] Scan script to report stale contracts.

---

## Related docs
- `src/fisheye/docs/zarr_structure.md` -- authoritative zarr layout spec
- `src/fisheye/shared/zarr/stage_arrays.py` -- code-level stage array contracts
- `src/fisheye/shared/zarr/schema.py` -- legacy zarr schema helpers (metadata/group skeleton)
- `src/fisheye/registry/db.py` -- registry database schema (migrations 1-27)
- `docs/review_status_schema_unification_contract.md` -- review payload schema
- `docs/detection_data_profile_schema_contract.md` -- detection profile schema
- `docs/keypoint_data_profile_schema_contract.md` -- keypoint profile schema
- `docs/eye_mask_data_profile_schema_contract.md` -- eye mask profile schema
- `docs/recording_manifest_contract.md` -- recording metadata contract
- `docs/detection_merged_export_contract.md` -- training export layout
- `docs/eye_mask_training_artifact_contract.md` -- eye mask training layout
- `src/fisheye/docs/citrus_data_structure_documentation.md` -- stimulus/protocol reference
