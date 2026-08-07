# Keypoint Training Data Card Contract
<!-- contract-meta
version: 2
status: active
last_verified: 2026-08-07
-->

Purpose: define a canonical, reproducible summary payload for keypoint (pose)
training datasets with parity to the detect training data-card flow.

Related TODO: `docs/keypoint_multi_skeleton_todo.md`

## Scope

In scope:
- keypoint training data-card schema (`v2`)
- metric naming and aggregation rules
- default plot bundle contract
- skeleton compatibility guardrails

Out of scope:
- model-quality pass/fail thresholds
- UI/dashboard implementation details

## Key Decisions

- Mixed skeleton training sets are invalid.
  - A single training set must resolve to exactly one skeleton identity
    (`skeleton_id` and `kpt_shape`).
  - If multiple skeleton identities are present, fail fast in:
    - `prepare_keypoint_training_from_registry`
    - `export_keypoint_training_zarr`
    - `validate_keypoint_training_zarr`
    - keypoint data-card aggregation
- Graph metrics are schema-derived, not hard-coded to fish-specific landmarks.
  - Compute edge and angle metrics from `pose_schema.skeleton`.
- Metric keys are canonical by index, with optional label aliases.
  - Canonical examples: `edge_0_1`, `angle_0_1_2`
  - Alias examples: `edge_swim_bladder_eye_left`, `angle_eye_left_swim_bladder_eye_right`

## Schema Identity

- `schema_name`: `keypoint_training_data_card`
- `schema_version`: `v2`

Version 2 retains the version 1 quality, geometry, skeleton, spatial, and
lineage fields. It adds recording-grouped acquisition and biological coverage;
version 1 cards remain historical evidence and are not rewritten in place.

## Required Payload Sections (`v2`)

### 1) `selection`

- `dataset_count`
- `rows_pre_gate`
- `rows_post_gate`
- `split_counts` (`train`, `val`, optional `test`)
- `quality_exclusions_by_reason`

### 2) `quality`

- `usable_keypoints_total`
- `usable_keypoints_rate_overall`
- `usable_keypoints_rate_dataset_stats`
- `raw_success_rate_overall`
- `confidence_valid_rate`
- `geometry_valid_rate`
- `flips_corrected_rate`

### 3) `geometry`

- `triangle_area_stats` (+ histogram)
- `min_angle_stats` (+ histogram)
- `heading_stats` (+ histogram)

### 4) `skeleton_graph_metrics`

Derived from `pose_schema.skeleton`:

- `edge_length_norm_stats`
  - one metric object per edge key (`edge_<i>_<j>`)
  - optional alias field per edge
  - histogram per edge
- `angle_stats`
  - one metric object per valid path (`angle_<i>_<j>_<k>`)
  - optional alias field per angle
  - histogram per angle

### 5) `semantic_metrics` (Optional)

- Optional profile-based metrics for known label sets.
- Must never be required for correctness.
- If label profile mismatch occurs, skip and emit warning metadata.

### 6) `spatial`

- center heatmaps per landmark index (and alias when available)
- per-landmark edge proximity rate

### 7) `composition_counts`

- `recording_id`
- `rig_id`
- `camera_id`
- `arena_id`
- `dish_design`
- `canvas_name`
- `protocol_name`
- `keypoint_method`

### 8) `subject_coverage`, `population_coverage`, and lineage parity

Mirror detect data-card lineage sections:

- `subject_coverage`
- `population_coverage`
  - `count_unit` (`recording_subject_observation`)
  - `source_dataset_count`
  - `recording_count`
  - `subject_count`
  - `recording_subject_observation_count`
  - `camera_dataset_counts`
  - `species_counts`
  - `line_strain_counts`
  - `canonical_strain_counts`
  - `genotype_counts`
  - `sex_counts`
  - `pigmentation_phenotype_counts`
  - `pigmentation_phenotype_origin_counts`
  - `melanophore_status_counts`
  - `xanthophore_status_counts`
  - `iridophore_status_counts`
  - `pigment_pattern_status_counts`
  - `optical_transparency_counts`
  - `unknown_counts`
- `genotype_counts`
- `dpf_stats`
- `dpf_histogram`

All population count mappings include an explicit `unknown` bin when the
registry lacks the corresponding value. Species, source strain label,
canonical strain, genotype, and pigmentation are separate axes. Pigmentation
values are read from the resolved trait contract. Their origin is explicit:
recording-scoped observations override curated strain expectations. Readers
must not parse strain labels, filenames, or image appearance to manufacture a
trait. See `docs/recording_subject_trait_contract.md`.

### 9) `train_val_parity`

Minimum parity deltas:

- usable keypoint rate
- confidence validity rate
- triangle area (`p50`, `p90`)
- min angle (`p50`, `p90`)
- lineage mix deltas (genotype, species, line/strain,
  pigmentation phenotype, camera, and DPF)

Split composition is descriptive rather than a quality threshold. For an
independent evaluation, rows must be assigned by recording/subject group before
sampling; frame-level random splitting is not acceptable evidence of
generalization.

### 10) `audit_freshness`

- `canonical_dataset_id_resolved_count`
- `zarr_mtime_mismatch_count`
- `quality_stale_count`
- `source_run_refs`

## Plot Bundle Contract

Default behavior:
- generate plots by default when card is written
- do not auto-open unless explicit `--view` is requested

Default plot set (`v2`):
- usable-rate distribution
- triangle-area distribution
- min-angle distribution
- heading distribution
- landmark heatmap panel
- genotype counts
- DPF histogram

## Storage Convention

Write card next to training artifacts:

- `/nvme1/training/datasets/<set_id>/<set_id>.data_card.json`

Recommended plots directory:

- `/nvme1/training/datasets/<set_id>/<set_id>.data_card.plots/`
