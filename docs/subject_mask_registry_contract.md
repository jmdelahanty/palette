# Subject Mask Registry Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-03-10
-->

## Purpose

Define the registry model for `subject_mask_runs` so Palette can:

- track run-level subject-mask production and freshness,
- track component-level quality and review state for
  `subject_body`, `eyes_union`, and `swim_bladder`,
- preserve the existing step-level recording dashboards,
- keep `refined_eye_masks_runs` representable as a historical or derived
  compatibility artifact.

This is a design contract. It does not by itself implement registry writers,
schema migrations, or review UIs.

## Scope

In scope:

- run-level registry projection for `subject_mask_runs`
- component-level registry projection for subject-mask channels
- review payload conventions for run-level and component-level review
- relationship to `recording_step_status`
- migration behavior for legacy eye-only backfilled subject-mask runs

Out of scope:

- `subject_shape_runs` registry design
- removal of `refined_eye_masks_runs` compatibility support
- model-training registry surfaces
- UI layout details beyond the required state model

## Layered Model

The registry should keep two layers, not one:

1. `recording_step_status`
   Coarse per-dataset step status for dashboards and pipeline freshness.
2. Subject-mask domain tables
   Run-level and component-level quality/review surfaces for queries, gating,
   and reviewer workflows.

This matches the existing split between:

- coarse step ledger writes through
  [`upsert_recording_step_status()`](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/registry/status_ledger.py#L105)
- stage-specific quality tables such as `eye_mask_quality`
  ([registry_schema_reference.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/registry_schema_reference.md#L843))

## Evolution Policy

The registry design must accommodate three runtime cases without another table
redesign:

1. Sparse compatibility runs
   Example: eye-only backfilled `subject_mask_runs`.
2. Dense multi-component runs
   Example: future models that emit body, eyes, and swim bladder together.
3. Component-scoped runs
   Example: future workflows where only one subset of subject-mask components
   is requested or reviewed.

V1 registry behavior must fully support cases 1 and 2.

Case 3 is intentionally deferred at the workflow level, but the schema should
already tolerate it through:

- component rows keyed by `component_name`
- explicit `available`
- run-level `mask_labels_json` and `available_channels_json`

That way, a future component-only run still projects cleanly without inventing
new registry families.

## Relationship To Existing Stages

Canonical stage relationship:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>
  -> subject_shape_runs/<run>
```

Historical and compatibility eye-mask relationship:

```text
crop_runs/<run>
  -> eye_masks_runs/<run>
  -> refined_eye_masks_runs/<run>
```

Registry implication:

- `subject_masks` becomes a coarse recording step
- `refined_subject_masks` is the canonical refined component step
- `refined_eye_masks` remains a separate coarse recording step for historical
  and compatibility artifacts
- component review for `eyes_union` lives under `subject_mask_runs`, not under
  `refined_eye_masks_runs`
- left/right eye review authority for modern runs lives under
  `refined_subject_masks_runs`; `refined_eye_masks_runs` can be projected for
  legacy query and diagnostic compatibility

## Coarse Step Policy

`recording_step_status` remains one row per dataset + step:

- do not create one `recording_step_status` row per subject component
- do not overload `step_name` with values like `subject_masks:eyes_union`

Recommended new/continued step names:

- `subject_masks`
- `refined_eye_masks`
- `subject_shape` (future)

The `subject_masks` row should summarize run-level component state through:

- `review_status_json`
  Run-level review payload, optionally including a compact component summary.
- `details_json`
  Query pointer fields such as:
  - `stage_group`
  - `run_name`
  - `available_components`
  - `component_review_counts`
  - `component_lifecycle_counts`

## Canonical Review Payloads On Zarr Runs

At `subject_mask_runs/<run>.attrs`:

- `subject_mask_review_status`
  Canonical run-level review payload.
- `component_review_statuses`
  JSON object mapping component name -> canonical review payload.

Parent attrs on `subject_mask_runs/`:

- `subject_mask_review_status_latest`
  Run name holding the latest run-level review status.

Canonical review payload keys follow the shared review contract:

- `state`
- `method`
- `intended_use`
- `reviewer`
- `notes`
- `timestamp_utc`

Reference:
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md#L83)

Example:

```json
{
  "subject_mask_review_status": {
    "state": "approved",
    "method": "manual",
    "intended_use": "training",
    "reviewer": "alice",
    "notes": "body and eye channels acceptable",
    "timestamp_utc": "2026-03-10T18:22:00Z"
  },
  "component_review_statuses": {
    "subject_body": {
      "state": "approved",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T18:21:10Z"
    },
    "eyes_union": {
      "state": "approved",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T18:21:20Z"
    },
    "swim_bladder": {
      "state": "needs_review",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T18:21:30Z"
    }
  }
}
```

## Runtime Availability Rule

Component review is constrained by runtime `available_channels` from the
subject-mask run contract:

- if a component channel is unavailable, it is not reviewable
- unavailable is not a negative
- unavailable components should project to registry as `available = false`
  and `lifecycle_state = "na"`

Reference:
- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md#L77)

This is required for eye-only backfilled runs where:

- `subject_body` is unavailable
- `eyes_union` is available
- `swim_bladder` is unavailable

The same rule should apply later to intentionally component-scoped runs. An
unavailable component remains queryable in the registry, but it is not treated
as reviewed, absent, or negative.

## Registry Tables

### 1. `subject_mask_performance`

Purpose:

- one row per `subject_mask_runs/<run>`
- run-level performance, lineage, availability, and overall review state

Primary key:

- `PRIMARY KEY (dataset_id, stage_group, run_name)`

Required identity/context fields:

- `dataset_id`
- `stage_group`
  Expected value in v1: `subject_mask_runs`
- `run_name`
- `run_created_utc`
- `recording_id`
- `zarr_use`
- `method`
- `run_semantics`
- `label_schema_id`
- `mask_labels_json`
- `available_channels_json`
- `source_crop_run`
- `source_background_run`
- `source_background_array`
- `source_dish_mask_array`
- `source_keypoint_group`
- `source_keypoints_run`
- `source_eye_masks_run`
- `source_refined_eye_masks_run`
- `tuning_source`
- `tuning_timestamp`
- `projection_mode`
- `probability_semantics`
- `probabilities_dtype`
- `probabilities_encoding`
- `zarr_mtime_ns`
- `updated_utc`

Required run-level summary fields:

- `total_rois`
- `available_component_count`
- `mask_present_any_rate_json`
  Mapping component -> fraction of rows with any positive pixel.
- `prob_max_p50_json`
  Mapping component -> median per-row `prob_max`.

Required review/lifecycle fields:

- `review_state`
- `review_method`
- `review_intended_use`
- `review_reviewer`
- `review_notes`
- `review_timestamp_utc`
- `lifecycle_state`
- `lifecycle_reason`

Recommended raw snapshots:

- `summary_statistics_json`
- `component_review_statuses_json`

For traditional body-only runs, these fields let the registry answer:

- which artifact family the run belongs to (`run_semantics`)
- which background source was used (`source_background_run`,
  `source_background_array`)
- whether dish gating was applied (`source_dish_mask_array`)
- which saved tuning entry generated the run (`tuning_source`,
  `tuning_timestamp`)
- how `mask_probs_roi` should be interpreted (`probability_semantics`)

### 2. `subject_mask_component_quality`

Purpose:

- one row per run component
- supports filtering/review gates by component rather than whole run

Primary key:

- `PRIMARY KEY (dataset_id, stage_group, run_name, component_name)`

Required identity/context fields:

- `dataset_id`
- `stage_group`
- `run_name`
- `component_name`
- `component_index`
- `run_created_utc`
- `recording_id`
- `zarr_use`
- `method`
- `label_schema_id`
- `source_crop_run`
- `source_keypoint_group`
- `source_keypoints_run`
- `source_eye_masks_run`
- `source_refined_eye_masks_run`
- `projection_mode`
- `zarr_mtime_ns`
- `updated_utc`

Required availability/quality fields:

- `available`
  `0/1`; false means channel is structurally unavailable in this run.
- `rows_total`
- `rows_mask_present`
- `mask_present_rate`
- `prob_max_p10`
- `prob_max_p50`
- `prob_max_p90`

Recommended geometry fields when available:

- `area_px_p10`
- `area_px_p50`
- `area_px_p90`
- `centroid_valid_rate`
- `bbox_valid_rate`

Required review/lifecycle fields:

- `review_state`
- `review_method`
- `review_intended_use`
- `review_reviewer`
- `review_notes`
- `review_timestamp_utc`
- `lifecycle_state`
- `lifecycle_reason`

Recommended snapshot fields:

- `component_review_status_json`

## Lifecycle Semantics

### Run-level lifecycle (`subject_mask_performance`)

Suggested derivation:

- if any available component has `state in {"needs_review", "pending", "review"}`:
  - `lifecycle_state = "in_progress"`
- else if all available components are approved:
  - `lifecycle_state = "approved"`
- else if any available component is rejected:
  - `lifecycle_state = "rejected"`
- else if no review payload exists:
  - `lifecycle_state = null`

### Component-level lifecycle (`subject_mask_component_quality`)

Suggested derivation:

- if `available = false`:
  - `lifecycle_state = "na"`
  - `lifecycle_reason = "component_unavailable"`
- else if `review_state in {"needs_review", "pending", "review"}`:
  - `lifecycle_state = "in_progress"`
- else if `review_state in {"approved", "rejected"}`:
  - `lifecycle_state = review_state`
- else:
  - `lifecycle_state = null`

## Latest Views

Recommended v1 views:

- `subject_mask_performance_latest`
  Latest row per `dataset_id + stage_group`
- `subject_mask_component_quality_latest`
  Latest row per `dataset_id + component_name`
- `subject_mask_component_quality_latest_by_recording`
  Latest row per `recording_id + component_name`

These should mirror the current latest-view pattern used by eye-mask tables.

## Extraction Rules From `subject_mask_runs`

The registry extractor should:

1. Iterate `subject_mask_runs/<run>`.
2. Read:
   - run attrs
   - `available_channels`
   - `metrics/prob_max`
   - `metrics/mask_present`
   - optional geometry metrics if present
3. Emit one run row to `subject_mask_performance`.
4. Emit one component row per label in `mask_labels`.

Required behavior for unavailable channels:

- still emit a component row
- set `available = false`
- set `rows_mask_present = 0`
- set `mask_present_rate = 0.0`
- set review fields to null unless an invalid payload exists
- set `lifecycle_state = "na"`

## Review Application Model

The review surface should be one generic subject-mask reviewer, not one
separate app per component.

Required capabilities:

- load one `subject_mask_runs/<run>`
- switch active component by `component_name`
- skip or disable unavailable channels
- write:
  - run-level `subject_mask_review_status`
  - component-level `component_review_statuses`
- trigger registry sync for:
  - `subject_mask_performance`
  - `subject_mask_component_quality`
  - `recording_step_status` for `step_name = "subject_masks"`

Non-goal:

- replacing the specialized `refined_eye_masks` reviewer, which still owns
  left/right identity and ellipse geometry review.

Deferred extension:

- component-scoped review workflows where an operator intentionally opens only
  one subset of subject-mask components

The registry contract should support that later without schema changes, but v1
does not require separate component-only apps or routing.

## Migration Policy

This contract must preserve compatibility during transition from legacy
eye-mask-first workflows.

Required migration behavior:

1. Keep existing `eye_mask_quality` and `refined_eye_masks` review flows.
2. Allow backfilled `subject_mask_runs` from legacy eye-mask runs.
3. For those backfilled runs:
   - `eyes_union` component rows are reviewable
   - `subject_body` and `swim_bladder` component rows are `available = false`
4. Do not infer run-level approval for unavailable components.
5. Do not collapse `refined_eye_masks` review state into raw `subject_masks`
   review state.

## Recommended Implementation Sequence

1. Add registry tables + latest views:
   - `subject_mask_performance`
   - `subject_mask_component_quality`
2. Implement extraction/sync from `subject_mask_runs`.
3. Add run attr helpers for:
   - `subject_mask_review_status`
   - `component_review_statuses`
4. Build `review_subject_masks.py`.
5. Sync `recording_step_status` for `subject_masks`.
6. Later connect `refined_eye_masks` to `subject_mask_runs` as upstream source
   without removing its own registry tables.

## Related Contracts

- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)
- [recording_step_status_parallel_agents_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/recording_step_status_parallel_agents_contract.md)
- [eye_mask_data_profile_schema_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_mask_data_profile_schema_contract.md)
