# Pose Schema vs Pose Heuristics Split Proposal

Purpose: define a clean split between skeleton semantics that belong in
`pose_schema.metadata` and stage-specific heuristic policy that should live in a
separate config surface.

This is a design proposal, not an active contract.

## Why This Split Is Needed

Recent heading work clarified an important distinction:

- some metadata describes what a skeleton means
- some logic is only a detector, review, or tuning policy

Those are not the same thing.

`pose_schema.metadata.heading_computation` is a good example of true skeleton
semantics. It should be stable across raw detection, refined review, Crimson,
export, and later analysis.

By contrast, rules such as:

- how to assign three unlabeled blobs to bladder/left-eye/right-eye
- which triangle angle thresholds are acceptable
- when a frame-to-frame heading jump should be flagged
- how to detect or correct eye flips

are heuristic policy. They may change while the skeleton itself remains the
same.

If we store both classes of logic in `pose_schema.metadata`, we blur the line
between:

- canonical pose meaning
- operational algorithm behavior

That makes schema reuse harder and turns detector tuning into a breaking schema
change.

## Design Goals

- keep `pose_schema` as the canonical definition of skeleton structure and
  deterministic pose semantics
- avoid baking detector or review thresholds into the skeleton itself
- allow multiple methods or stages to share one skeleton while using different
  heuristics
- make downstream consumers safe to implement without copying ad hoc fish logic
- keep room for future skeletons beyond the current traditional 3-point and
  5-point flows

## Non-Goals

- move every runtime decision into metadata
- make all detector logic generic immediately
- define every future heuristic profile field in this proposal
- replace current run-level tuning attrs that are already stage-local

## Proposed Split

### `pose_schema.metadata` should contain

Only information that is part of the meaning of the skeleton or deterministic
geometry derived from labeled keypoints.

Examples:

- `heading_computation`
- explicit landmark-role groupings when several consumers must agree on them
- canonical orientation origin or body-frame conventions derived from labels
- semantic dependency sets for downstream rendering or editing

Rule of thumb:

- if a value should remain true when we swap detectors, review tools, or batch
  thresholds, it is a candidate for `pose_schema.metadata`

### Heuristic profiles should contain

Method-specific or stage-specific policy for interpreting imperfect detections,
reviewing outputs, or applying thresholds.

Examples:

- traditional 3-blob assignment rules
- left/right disambiguation heuristics for unlabeled proposals
- eye-flip detection policy
- geometry acceptance thresholds
- temporal heading outlier thresholds
- review ranking or auto-triage policy

Rule of thumb:

- if a value could change during tuning while the skeleton identity remains the
  same, it should not be part of `pose_schema.metadata`

## Decision Table

| Item | Belongs in `pose_schema.metadata` | Belongs in heuristic profile |
| --- | --- | --- |
| `heading_computation` | yes | no |
| label dependency list for heading preview | yes | no |
| `traditional_v1` blob-to-label assignment rule | no | yes |
| min triangle angle threshold | no | yes |
| temporal heading outlier threshold | no | yes |
| left/right eye swap correction policy | no | yes |
| deterministic landmark-role grouping | yes | no |

## Proposed Heuristic Surface

Do not overload `pose_schema.metadata` for heuristic policy.

Instead, introduce a sibling packaged config surface such as:

- `configs/fisheye/pose_heuristics/<method>/<skeleton>.json`

Example:

- `configs/fisheye/pose_heuristics/traditional_pose/traditional_v1.json`
- `configs/fisheye/pose_heuristics/traditional_pose/traditional_v2.json`

This keeps heuristics namespaced by both:

- detection or review method
- skeleton identity

That is important because one skeleton may support multiple producer families.

## Suggested Heuristic Profile Shape

Initial sketch:

```json
{
  "version": 1,
  "skeleton_id": "pose_skel_traditional_v1",
  "method": "traditional_pose",
  "blob_assignment": {
    "family": "triangle_3blob",
    "bladder_vertex_rule": "smallest_angle",
    "left_right_rule": "heading_relative"
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
  "flip_policy": {
    "family": "traditional_eye_flip_v1"
  }
}
```

This proposal does not require this exact field set. The important part is the
separation of concerns.

## Read/Write Guidance

### Writers

- keypoint and refined-keypoint writers should continue to write semantic pose
  meaning into `pose_schema.metadata`
- writers should not copy detector thresholds into `pose_schema.metadata`
- stage-local tuning attrs may remain on the run when they are truly run-local

### Readers

- downstream consumers such as Crimson should trust
  `pose_schema.metadata.heading_computation` for heading semantics
- consumers should not infer pose meaning from detector heuristics
- detector or review tools may load heuristic profiles when their algorithm
  needs them, but should treat those as method policy, not pose meaning

## Migration Guidance

Short term:

- keep heading semantics in `pose_schema.metadata.heading_computation`
- use packaged heuristic profiles for shared traditional detector/tuner
  defaults where the policy is genuinely method-level rather than run-local
- keep stage-local tuning attrs when operators need one-off overrides

Current state:

- packaged profiles now exist under
  `configs/fisheye/pose_heuristics/traditional_pose/`
- `detect_keypoints_traditional` and `keypoint_tuner` load the packaged
  `traditional_v1` profile for blob assignment and geometry-QC defaults
- the raw traditional detector/tuner still target the starter 3-point layout,
  so packaged `traditional_v2` is available for future skeleton-aware readers
  rather than selected automatically by those tools

Medium term:

- move remaining reusable traditional review/refine policy into packaged
  heuristic configs where it should be shared
- move reusable QC threshold defaults into heuristic profiles
- keep run-level tuning attrs only for explicit local deviation

Long term:

- reserve `pose_schema.metadata` for stable cross-consumer semantics
- treat heuristic profiles as replaceable algorithm policy that can evolve
  independently of skeleton identity

## Practical Boundary Test

Before adding a new field to `pose_schema.metadata`, ask:

1. Would this still be true if we changed detection method?
2. Would a downstream reader need the same meaning even without our detector?
3. Is this derived deterministically from labeled keypoints or skeleton roles?

If the answer is mostly yes, it probably belongs in `pose_schema.metadata`.

If the answer is mostly no, it probably belongs in a heuristic profile or a
stage-local tuning surface.

## Open Questions

- should heuristic profiles support run-level overrides, or should run-local
  tuning remain ordinary stage attrs?
- should some current geometry metrics move to a semantic role surface if more
  consumers need them?
- should review heuristics and detection heuristics share one profile file or
  remain separate families?

## Related Docs

- [pose_heuristic_profile_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_heuristic_profile_contract.md)
- [keypoint_heading_computation_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_heading_computation_contract.md)
- [keypoint_multi_skeleton_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_multi_skeleton_todo.md)
- [keypoint_derived_metric_schema_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_derived_metric_schema_contract.md)
- [traditional_v2_keypoint_migration_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/archive/traditional_v2_keypoint_migration_design.md)
