# Derived Metrics Schema Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-11
-->

Purpose: define a run-level metadata contract for derived numeric/vector
measurements and optional boolean/status gates without collapsing every stage
into a single top-level `quality_metrics` abstraction.

## Design Goals

This contract intentionally separates three concerns:

1. Entity schema
   What one row stores before any derived computation.
2. Derived metrics schema
   What measurements are computed from that row.
3. Quality gates
   Optional boolean/status outputs computed from the derived metrics.

This separation matters because keypoints, detect boxes, and masks have
different entity schemas even when some downstream consumer wants "quality"
signals from all of them.

## Scope

The contract is run-level and is intended to work for:

- keypoint ROI runs
- detect bbox runs / refined detect groups
- mask ROI runs

Immediate implementation target:

- `refined_keypoints_runs/<run>.attrs["derived_metrics_schema"]`

## Canonical Placement

When present, writers should store:

- `<run>.attrs["derived_metrics_schema"]`

Version 1 payload shape:

```json
{
  "schema_version": 1,
  "entity_kind": "keypoint_roi",
  "metrics": [],
  "quality_gates": []
}
```

## Reader Rules

- When `derived_metrics_schema` is present, it is the authoritative semantic
  description of the run's derived arrays and quality-gate arrays.
- When it is absent, readers must continue supporting legacy stage-specific
  behavior and fallbacks.
- This contract describes semantics only. It does not require a storage-layout
  rewrite.
- Existing arrays stay where they already live.

## Metadata Backfill

Older refined-keypoint runs may already contain the derived arrays but lack the
run-level `derived_metrics_schema` attr. These runs can be upgraded without
rerunning keypoint prediction or refinement:

```bash
scripts/py -m fisheye.utils.backfill_keypoint_derived_metrics_schema \
  /path/to/archive_analysis.zarr \
  --apply
```

The backfill is metadata-only. It writes `derived_metrics_schema` only when the
run already has the refined-keypoint arrays described by the schema:
`keypoints_roi`, `triangle_area`, `triangle_angles`, `min_angle`, and
`geometry_valid`. It resolves keypoint labels from run-level `keypoint_labels`
first, then `pose_schema`, and skips runs whose labels cannot identify the
swim-bladder/left-eye/right-eye triangle.

## Relationship To Entity Schema

`derived_metrics_schema` does not replace the entity schema.

Examples:

- keypoint runs still use `pose_schema`, `keypoint_labels`, and the keypoint
  coordinate arrays as the entity schema
- detect runs still use bbox arrays such as `bbox_norm_coords` /
  `bbox_img_xyxy`
- mask runs still use `mask_labels`, `available_channels`, `masks_roi`, and
  `mask_probs_roi`

`entity_kind` only tells the reader what kind of per-row entity the metrics are
derived from.

## Relationship To Heading Metadata

For keypoints, heading semantics remain separate:

- `pose_schema.metadata.heading_computation` is still the canonical heading
  definition
- `heading_computation_override` remains the optional run-level override

`derived_metrics_schema` does not replace or redefine heading computation.

## Metric Object Contract

Each entry in `metrics` should declare:

- `name`
- `kind`
- `source.array`
- `source.value_kind`
- optional `source.coordinate_space`
- optional `selectors` such as:
  - `labels`
  - `indices`
  - `mask_value`
- `outputs`, where each output describes the array populated by the metric

Version 1 does not require a single closed enum for every possible
`value_kind`, but writers should use stable machine-readable strings.

Example metric kinds:

- `triangle_3pt`
- `distance_2pt`
- `bbox_area`
- `bbox_aspect_ratio`
- `mask_area_pixels`
- `mask_centroid`

## Quality Gate Contract

Each entry in `quality_gates` should declare:

- `name`
- `kind`
- `output`
- `conditions`

Version 1 quality gates are intended for boolean or status-like outputs such as:

- `geometry_valid`
- `confidence_valid`
- `usable_keypoints`
- `bbox_valid`
- `mask_valid`

Conditions may reference:

- a metric name
- a metric output name
- a literal operation such as `is_finite`, `>=`, `<=`
- a threshold attr path such as `summary_statistics.min_triangle_area` or a
  more stage-specific path like `summary_statistics.refine.min_area`

## Refined Keypoint V1 Example

Current Palette refined-keypoint implementation writes:

- `entity_kind = "keypoint_roi"`
- one metric:
  - `eye_triangle_geometry`
  - `kind = "triangle_3pt"`
  - source array `keypoints_roi`
  - selectors identifying the three triangle keypoints
  - outputs:
    - `triangle_area`
    - `triangle_angles`
    - `min_angle`
- one quality gate:
  - `geometry_valid`
  - conditions referencing:
    - finite `triangle_area`
    - finite `min_angle`
    - `summary_statistics.min_triangle_angle`
    - `summary_statistics.min_triangle_area`
    - optional `summary_statistics.max_triangle_area`

This makes the current eye-triangle geometry explicit without forcing downstream
consumers to guess it from label names or array names alone.

## Detect BBox Compatibility

This contract is compatible with detect/refined-detect runs because their
entity schema is already stable and row-oriented.

Likely future mapping:

- `entity_kind = "detect_bbox"`
- `source.array = "bbox_norm_coords"` or `bbox_img_xyxy`
- metric kinds such as:
  - `bbox_area`
  - `bbox_aspect_ratio`
  - `bbox_center`
- quality gates such as:
  - `bbox_valid`
  - `confidence_valid`

The important point is that bbox semantics stay in the bbox arrays, while
derived metrics and gates become explicit metadata instead of consumer-side
heuristics.

## Mask ROI Compatibility

This contract is compatible with mask runs because the entity schema is already
defined by:

- `mask_labels`
- `available_channels`
- `masks_roi`
- `mask_probs_roi`

Likely future mapping:

- `entity_kind = "mask_roi"`
- metric kinds such as:
  - `mask_area_pixels`
  - `mask_centroid`
  - `mask_bbox`
  - `prob_max`
- quality gates such as:
  - `mask_valid`
  - `component_present`

For masks, `available_channels` remains the entity-schema availability signal;
`derived_metrics_schema` only describes metrics and gates derived from the mask
content.

## Related Documents

- `docs/keypoint_heading_computation_contract.md`
- `docs/keypoint_derived_metric_schema_contract.md`
- `docs/crimson_detect_bbox_read_contract.md`
- `docs/subject_mask_runs_contract.md`
- `src/fisheye/docs/zarr_structure.md`
