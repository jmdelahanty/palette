# Pose Heuristic Profile Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-17
-->

## Purpose

Define a first-pass contract for packaged pose heuristic profiles that are
separate from `pose_schema.metadata`.

This contract exists to keep a clean boundary between:

- skeleton semantics that belong in `pose_schema`
- method-specific or stage-specific heuristic policy that should evolve
  independently

Examples of heuristic policy covered here:

- unlabeled blob assignment rules
- left/right disambiguation fallback behavior
- geometry quality-control thresholds
- temporal heading review thresholds
- flip-detection family selection

## Relationship To `pose_schema`

This contract is intentionally separate from
`pose_schema.metadata.heading_computation`.

Use this split:

- `pose_schema.metadata` for deterministic pose meaning
- heuristic profiles for detector/review policy

Do not copy heuristic thresholds or blob-assignment rules into
`pose_schema.metadata`.

## Scope

In scope:

- packaged heuristic profile files under `configs/fisheye/pose_heuristics/`
- profile shape for method-specific heuristic defaults
- first-pass selection keys for matching a profile to a skeleton

Out of scope:

- run-level override attrs
- registry projection of heuristic profiles
- mandatory runtime adoption by every existing tool
- Crimson consumer behavior

## Packaged Config Location

Packaged defaults should live at:

- `configs/fisheye/pose_heuristics/<method>/<pose_schema_name>.json`

Examples:

- `configs/fisheye/pose_heuristics/traditional_pose/traditional_v1.json`
- `configs/fisheye/pose_heuristics/traditional_pose/traditional_v2.json`

This first-pass path is keyed by:

- method family
- packaged pose schema name

The file may also carry `skeleton_id` for validation and future stricter
selection.

## Selection Model (`v1`)

First-pass profile resolution is intentionally simple.

Recommended packaged-default lookup:

1. select method family
2. resolve pose schema name
3. load `configs/fisheye/pose_heuristics/<method>/<pose_schema_name>.json`
4. if present, optionally validate `skeleton_id`
5. if absent, no heuristic profile is available

This contract does not yet define a run-level override mechanism.

## Design Principles

1. Keep pose meaning and heuristic policy separate.
2. Prefer explicit named families over implicit magic booleans.
3. Fail closed for unknown profile families or unsupported sections.
4. Allow one skeleton to support multiple method families.
5. Keep `v1` narrow and additive.

## Required Top-Level Fields

- `profile_name`
- `profile_version`
- `method`
- `source_pose_schema`
- `skeleton_id`

### Field meanings

#### `profile_name`

Human-readable packaged profile identifier.

Example:

- `traditional_pose_traditional_v1`

#### `profile_version`

String or integer profile contract version.

`v1` examples in this repo use:

- `"v1"`

#### `method`

Method family this profile applies to.

Examples:

- `traditional_pose`
- `manual_review`
- `retry_pose`

#### `source_pose_schema`

Packaged pose schema name this profile targets.

Examples:

- `traditional_v1`
- `traditional_v2`

#### `skeleton_id`

Expected skeleton identity for validation.

For legacy schemas that do not yet carry explicit `skeleton_id`, use the
documented fallback form:

- `pose_schema:<name>`

Examples:

- `pose_schema:traditional_v1`
- `pose_skel_traditional_v2`

## Optional Sections (`v1`)

All optional sections are independent. Missing sections mean:

- no packaged default is defined for that policy family

Supported optional sections in `v1`:

- `label_requirements`
- `blob_assignment`
- `geometry_qc`
- `heading_qc`
- `flip_detection`
- `notes`

## `label_requirements`

Declares which labels the heuristic profile expects to exist.

### Fields

- `required_labels`
- optional `core_assignment_labels`

### Semantics

- `required_labels` lists labels the profile assumes are present on the run
- `core_assignment_labels` lists labels used by unlabeled assignment logic
  when only a subset of the full skeleton participates

## `blob_assignment`

Describes how a producer maps unlabeled candidates onto semantic labels.

### Required fields

- `family`

### Optional fields

- `bladder_vertex_rule`
- `left_right_rule`
- `fallback_rule`

### Supported `family` values in `v1`

- `triangle_3blob`

### Supported rule values in `v1`

- `bladder_vertex_rule`
  - `smallest_angle`
- `left_right_rule`
  - `heading_relative`
- `fallback_rule`
  - `image_x_order`

Unknown values should be treated as unsupported.

## `geometry_qc`

Packaged default thresholds for geometry acceptance or review.

Supported `v1` fields:

- `min_triangle_angle_deg`
- `max_triangle_angle_deg`
- `min_triangle_area_px`
- `max_triangle_area_px`

These are defaults, not canonical pose meaning.

## `heading_qc`

Packaged default thresholds for heading-based review heuristics.

Supported `v1` fields:

- `temporal_outlier_threshold_deg`
- `max_frame_gap`

These do not define heading semantics. Heading semantics still come from
`pose_schema.metadata.heading_computation`.

## `flip_detection`

Selects the heuristic family used for left/right eye flip correction.

### Required fields

- `family`

### Supported `family` values in `v1`

- `traditional_eye_flip_v1`

## Example: `traditional_v1`

```json
{
  "profile_name": "traditional_pose_traditional_v1",
  "profile_version": "v1",
  "method": "traditional_pose",
  "source_pose_schema": "traditional_v1",
  "skeleton_id": "pose_schema:traditional_v1",
  "label_requirements": {
    "required_labels": ["swim_bladder", "eye_left", "eye_right"],
    "core_assignment_labels": ["swim_bladder", "eye_left", "eye_right"]
  },
  "blob_assignment": {
    "family": "triangle_3blob",
    "bladder_vertex_rule": "smallest_angle",
    "left_right_rule": "heading_relative",
    "fallback_rule": "image_x_order"
  },
  "geometry_qc": {
    "min_triangle_angle_deg": 10.0,
    "max_triangle_angle_deg": 90.0,
    "min_triangle_area_px": 100.0,
    "max_triangle_area_px": null
  },
  "heading_qc": {
    "temporal_outlier_threshold_deg": 120.0,
    "max_frame_gap": 3
  },
  "flip_detection": {
    "family": "traditional_eye_flip_v1"
  }
}
```

## Example: `traditional_v2`

```json
{
  "profile_name": "traditional_pose_traditional_v2",
  "profile_version": "v1",
  "method": "traditional_pose",
  "source_pose_schema": "traditional_v2",
  "skeleton_id": "pose_skel_traditional_v2",
  "label_requirements": {
    "required_labels": [
      "swim_bladder",
      "eye_left",
      "eye_right",
      "snout_tip",
      "tail_tip"
    ],
    "core_assignment_labels": ["swim_bladder", "eye_left", "eye_right"]
  },
  "blob_assignment": {
    "family": "triangle_3blob",
    "bladder_vertex_rule": "smallest_angle",
    "left_right_rule": "heading_relative",
    "fallback_rule": "image_x_order"
  },
  "geometry_qc": {
    "min_triangle_angle_deg": 10.0,
    "max_triangle_angle_deg": 90.0,
    "min_triangle_area_px": 100.0,
    "max_triangle_area_px": null
  },
  "heading_qc": {
    "temporal_outlier_threshold_deg": 120.0,
    "max_frame_gap": 3
  },
  "flip_detection": {
    "family": "traditional_eye_flip_v1"
  }
}
```

## Writer Guidance

- packaged heuristic profiles should be committed under
  `configs/fisheye/pose_heuristics/`
- do not embed these profiles into `pose_schema.metadata`
- do not duplicate run-local tuned thresholds into packaged defaults unless they
  are intended to be shared defaults

## Reader Guidance

- consumers should load heuristic profiles only when they need method policy
- consumers should continue to use `pose_schema.metadata.heading_computation`
  for semantic heading interpretation
- if a required heuristic profile is missing, the consumer should either:
  - fall back to existing hardcoded legacy behavior during migration, or
  - fail with a clear unsupported-profile error

## Open Questions

- should detection and review use separate method namespaces, or share one
  packaged family when the policy is identical?
- should run-level heuristic overrides be added later, or should stage-local
  attrs remain the only override surface?
- should some geometry defaults move into a generic review-policy contract
  instead of pose-adjacent heuristic profiles?

## Related Docs

- [pose_schema_heuristics_split_proposal.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_schema_heuristics_split_proposal.md)
- [keypoint_heading_computation_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_heading_computation_contract.md)
- [keypoint_multi_skeleton_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_multi_skeleton_todo.md)
