# Keypoint Derived Metric Schema Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-03-11
-->

## Purpose

Define a skeleton-aware contract for named keypoint-derived metrics so Palette can
support richer anatomy measurements such as:

- `total_length`
- `tail_length`
- `head_length`
- `eye_span`

without adding ad hoc arrays or registry columns for each new skeleton.

This contract is intentionally layered on top of existing refined-keypoint and
keypoint-profile artifacts:

- schema-driven `edge_distances` remain the generic geometry base layer
- named derived metrics become a second, anatomy-aware layer

This document covers the columnar `derived_metric_values*` arrays only.

It is separate from the run-level `derived_metrics_schema` contract in
[derived_metrics_schema_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/derived_metrics_schema_contract.md),
which describes the semantic meaning of derived arrays and boolean/status gates
for a run.

## Scope

In scope:

- per-run storage contract for derived metric arrays on `refined_keypoints_runs`
- schema/config contract for skeleton-specific metric definitions
- profile-summary aggregation contract for derived metrics

Out of scope:

- registry projection columns for every derived metric
- kinematic time-series modeling
- behavior classification or downstream phenotyping policy

## Current Implementation Status

Implemented:

- metric schema config loading from
  `configs/fisheye/keypoint_metric_schemas/*.json`
- derived metric storage on `refined_keypoints_runs/<run>/`
- recompute on initial refinement write
- recompute on manual keypoint save/clear actions
- one-shot backfill for existing refined runs via
  `scripts/py -m fisheye.utils.backfill_keypoint_derived_metrics`
- keypoint profile aggregation into
  `analysis/keypoint_profile_runs/<run>.attrs["profile_summary"]`

Deferred:

- registry SQL columns for derived metrics
- registry query/report surfaces for derived metrics

This deferral is intentional because different skeletons may define different
metric sets, and the cross-skeleton query policy needs to be settled before
projecting them into registry views.

## Design Principles

1. Keep generic geometry generic.
   `edge_distances` should continue to come directly from `pose_schema.edges`.

2. Add anatomy metrics as named derivations.
   Metrics like `total_length` are semantic measurements, not just unnamed edges.

3. Avoid stage-schema churn.
   New skeletons should not require one new top-level Zarr array per metric.

4. Prefer label-based definitions over fixed indices.
   Metric definitions should resolve keypoints by `keypoint_labels`.

5. Keep registry denormalization selective.
   Full metric payloads should live in run/profile JSON first; only stable,
   query-critical metrics should later be promoted into SQL columns.

## Layered Geometry Model

### Layer 1: Skeleton Edge Metrics

Already supported today via:

- `edge_pairs`
- `edge_distances`
- `edge_distances_norm`
- `edge_distance_valid`
- `edge_distance_labels`

These are automatically derived from `pose_schema.edges`.

### Layer 2: Named Derived Metrics

New layer introduced by this contract:

- `derived_metric_values`
- `derived_metric_values_norm`
- `derived_metric_valid`
- `derived_metric_labels`
- `derived_metric_definitions`

These metrics are anatomy-aware and skeleton-specific.

Examples:

- `total_length = distance(snout_tip, tail_tip)`
- `tail_length = distance(swim_bladder, tail_tip)`
- `head_length = distance(snout_tip, swim_bladder)`
- `eye_span = distance(eye_left, eye_right)`

## Metric Schema Files

Derived metric definitions should live in config files separate from pose
schemas, for example:

- `configs/fisheye/keypoint_metric_schemas/traditional_v2.json`
- `configs/fisheye/keypoint_metric_schemas/traditional_v3.json`

This separation is intentional:

- pose schema defines the skeleton graph
- metric schema defines named anatomical measurements derived from that graph

### Required Metric Schema Fields

Top-level:

- `schema_name`
- `schema_version`
- `skeleton_id`
- `source_pose_schema`
- `metrics`

Each metric entry:

- `name`
- `type`
- `from_label`
- `to_label`
- `units`
- `normalization`
- `description`

### Supported Metric Types (`v1`)

For `v1`, only:

- `distance`

Later possible extensions:

- `path_length`
- `angle`
- `ratio`

### Example Metric Schema (`traditional_v2`)

```json
{
  "schema_name": "traditional_v2_derived_metrics",
  "schema_version": "v1",
  "skeleton_id": "pose_skel_traditional_v2",
  "source_pose_schema": "traditional_v2",
  "metrics": [
    {
      "name": "total_length",
      "type": "distance",
      "from_label": "snout_tip",
      "to_label": "tail_tip",
      "units": "px",
      "normalization": "roi_diagonal",
      "description": "Full body axis distance from snout to tail tip."
    },
    {
      "name": "tail_length",
      "type": "distance",
      "from_label": "swim_bladder",
      "to_label": "tail_tip",
      "units": "px",
      "normalization": "roi_diagonal",
      "description": "Posterior body/tail distance from swim bladder to tail tip."
    },
    {
      "name": "head_length",
      "type": "distance",
      "from_label": "snout_tip",
      "to_label": "swim_bladder",
      "units": "px",
      "normalization": "roi_diagonal",
      "description": "Anterior body distance from snout to swim bladder."
    },
    {
      "name": "eye_span",
      "type": "distance",
      "from_label": "eye_left",
      "to_label": "eye_right",
      "units": "px",
      "normalization": "roi_diagonal",
      "description": "Distance between left and right eye centers."
    }
  ]
}
```

## Refined Run Storage Contract

### Storage Target

Within:

- `refined_keypoints_runs/<run>/`

### Required Arrays

- `derived_metric_values` with shape `(n_rois, n_metrics)` and dtype `float32`
- `derived_metric_values_norm` with shape `(n_rois, n_metrics)` and dtype `float32`
- `derived_metric_valid` with shape `(n_rois, n_metrics)` and dtype `bool`

### Required Attributes

- `derived_metric_schema_id`
- `derived_metric_schema_version`
- `derived_metric_labels`
- `derived_metric_type = "named_keypoint_derivations"`
- `derived_metric_source = "keypoint_metric_schema"`
- `derived_metric_normalization`
- `derived_metric_definitions`

### Attribute Semantics

- `derived_metric_schema_id`
  Example: `traditional_v2_derived_metrics`

- `derived_metric_labels`
  Ordered metric names corresponding to columns in the metric arrays

- `derived_metric_definitions`
  JSON-serializable copy of the metric-schema entries actually applied to the run

- `derived_metric_normalization`
  Example:
  ```json
  {
    "mode": "roi_diagonal",
    "roi_diagonal": 724.08
  }
  ```

### Validity Rules

For a distance metric:

- `derived_metric_valid[row, metric] = true` only if both required keypoints are finite
- invalid metrics must write `NaN` to `derived_metric_values`
- invalid normalized metrics must write `NaN` to `derived_metric_values_norm`

## Relationship to Existing Triangle Metrics

This contract does not replace:

- `triangle_area`
- `min_angle`
- `triangle_angles`

Those remain valid for the current starter head-triangle logic and should keep
being computed from the first three canonical anchor points:

- `swim_bladder`
- `eye_left`
- `eye_right`

For `traditional_v2` and later skeletons:

- triangle metrics remain compatibility metrics
- derived metrics become the extensible anatomy-aware layer

## Computation Timing

Derived metrics should be recomputed whenever refined keypoints are materially
updated, including:

- initial refinement write
- manual correction save
- patching/import of refined keypoints

They should be computed from the refined run, not the raw run.

## Profile Aggregation Contract

`analysis/keypoint_profile_runs/<run>.attrs["profile_summary"]` should gain an
optional section:

```json
{
  "geometry": {
    "triangle_area": {"stats": {}},
    "min_angle": {"stats": {}},
    "heading": {"stats": {}},
    "edge_distance": {},
    "derived_metrics": {
      "schema_id": "traditional_v2_derived_metrics",
      "schema_version": "1.0",
      "labels": ["total_length", "tail_length", "head_length", "eye_span"],
      "normalization": {"mode": "roi_diagonal", "roi_diagonal": 724.08},
      "metrics": [
        {
          "name": "total_length",
          "valid_count": 0,
          "valid_rate": 0.0,
          "stats": {
            "count": 0,
            "min": null,
            "max": null,
            "mean": null,
            "std": null,
            "p10": null,
            "p50": null,
            "p90": null
          },
          "stats_norm": {
            "count": 0,
            "min": null,
            "max": null,
            "mean": null,
            "std": null,
            "p10": null,
            "p50": null,
            "p90": null
          }
        }
      ]
    }
  }
}
```

For each derived metric:

- `valid_count`
- `valid_rate`
- `count`
- `min`
- `max`
- `mean`
- `std`
- `p10`
- `p50`
- `p90`

## Registry Policy

For `v1`, do not add one SQL column per derived metric.

Registry policy:

1. Store derived-metric summaries inside `profile_json`
2. Keep query-critical SQL projection limited to stable legacy fields
3. Promote selected derived metrics later only if there is a clear product need

This avoids schema churn as skeletons evolve.

Until that policy is settled, registry/query tooling should treat derived
metrics as profile-only payloads.

## Traditional V2 Initial Recommendation

For the first `traditional_v2` rollout, define and support these metrics:

- `total_length`
- `tail_length`
- `head_length`
- `eye_span`

Optional later:

- `snout_to_eye_left`
- `snout_to_eye_right`
- `left_eye_to_tail_tip`
- `right_eye_to_tail_tip`

## Traditional V3 Initial Recommendation

`traditional_v3` extends the `traditional_v2` metric set with tail-segment and
pectoral-fin distances:

- `anterior_tail_segment`
- `posterior_tail_segment`
- `right_pectoral_fin_length`
- `left_pectoral_fin_length`
- `right_pectoral_insertion_to_eye`
- `left_pectoral_insertion_to_eye`

These metrics are schema-specific and should remain inside derived-metric arrays
and profile payloads until a skeleton-aware registry/query projection policy is
defined.

## Deferred Items

- metric types beyond simple point-to-point distance
- skeleton-path metrics using more than two anchors
- ratios and normalized anatomical proportions
- registry first-class columns for selected derived metrics
- training/export cards explicitly plotting derived-metric distributions

## Recommended Rollout Order

1. add this contract
2. add metric schema config for `traditional_v2`
3. compute/store derived metric arrays on `refined_keypoints_runs`
4. extend keypoint profile summaries to include derived metrics
5. only then consider registry promotion of selected metrics
