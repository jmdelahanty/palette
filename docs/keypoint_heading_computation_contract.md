# Keypoint Heading Computation Contract

<!-- contract-meta
status: active
last_verified: 2026-04-09
-->

Purpose: define explicit, machine-readable metadata for how keypoint runs
compute heading and heading-arrow geometry.

This contract exists so downstream consumers such as Crimson do not infer
heading semantics from hardcoded label names or fixed 3-point fish logic.

## Scope

This contract covers:

- how a keypoint run declares heading semantics
- where that metadata lives
- how readers resolve precedence between run metadata and skeleton metadata
- how UI/editing tools determine whether an edited keypoint affects heading

This contract does not cover:

- unlabeled blob-assignment policy
- geometry acceptance thresholds
- heading temporal outlier policy
- flip-detection family selection

Those are heuristic-policy concerns, not heading semantics. See:

- `docs/pose_heuristic_profile_contract.md`

This contract does not require every keypoint run to define heading semantics.
Runs may explicitly disable heading computation or omit it entirely.

## Placement And Precedence

### Canonical location

The canonical definition of heading semantics belongs to the skeleton:

- `pose_schema.metadata.heading_computation`

This is the preferred source of truth because heading definition is part of
what the skeleton means, not primarily a property of one specific run.

### Optional run-level override

Runs may optionally carry:

- `keypoints_runs/<run>.attrs["heading_computation_override"]`
- `refined_keypoints_runs/<run>.attrs["heading_computation_override"]`

This exists only for explicit per-run override or disable behavior during
migration or exceptional cases. It is not the canonical definition.

### Deprecated transitional alias

During migration, readers may also tolerate:

- `keypoints_runs/<run>.attrs["heading_computation"]`
- `refined_keypoints_runs/<run>.attrs["heading_computation"]`

New writers should not rely on this alias. It is a compatibility bridge only.

### Precedence rules

Readers should resolve heading metadata in this order:

1. run attr `heading_computation_override`
2. `pose_schema.metadata.heading_computation`
3. deprecated run attr `heading_computation`
4. heading semantics unavailable

Override rule:

- if `heading_computation_override` exists and `enabled == false`, that
  disables heading semantics for the run even if `pose_schema.metadata`
  defines them

Recommendation for new writers:

- write the canonical payload into `pose_schema.metadata.heading_computation`
- use `heading_computation_override` only when a run must explicitly differ
  from the skeleton default

## Metadata Shape

Version 1 payload:

```json
{
  "version": 1,
  "enabled": true,
  "origin": {
    "op": "midpoint",
    "labels": ["eye_left", "eye_right"]
  },
  "direction_from": {
    "op": "keypoint",
    "label": "swim_bladder"
  },
  "direction_to": {
    "op": "midpoint",
    "labels": ["eye_left", "eye_right"]
  },
  "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"]
}
```

Required top-level fields:

- `version`: integer, currently `1`
- `enabled`: bool
- `origin`: point-expression object when `enabled == true`
- `direction_from`: point-expression object when `enabled == true`
- `direction_to`: point-expression object when `enabled == true`
- `dependent_keypoints`: explicit ordered list of labels when `enabled == true`

Minimal disabled payload:

```json
{
  "version": 1,
  "enabled": false
}
```

## Point Expressions

Version 1 required reader support:

### `keypoint`

```json
{ "op": "keypoint", "label": "swim_bladder" }
```

Semantics:

- use the coordinates of exactly one labeled keypoint

### `midpoint`

```json
{ "op": "midpoint", "labels": ["eye_left", "eye_right"] }
```

Semantics:

- use the midpoint of exactly two labeled keypoints

### Reserved for future expansion

Future versions may add expressions such as:

```json
{ "op": "centroid", "labels": ["snout", "eye_left", "eye_right"] }
```

Version 1 writers should not emit unsupported ops. Version 1 readers should
fail closed for unknown ops instead of inventing fallback behavior.

## Explicit Dependency Rule

`dependent_keypoints` must be explicit.

Consumers should use it to answer:

- does editing this keypoint affect candidate heading?
- should the dashed candidate-heading preview be recomputed?

Version 1 writer rule:

- `dependent_keypoints` should be the de-duplicated union of labels referenced
  by `origin`, `direction_from`, and `direction_to`

Version 1 reader rule:

- readers may validate that `dependent_keypoints` matches that union
- readers should still trust the explicit list for dependency gating even if
  they also warn about a mismatch

## Heading Scalar Semantics

Version 1 fixes the scalar heading formula; it is not another configurable
subfield.

Given resolved points:

- `A = evaluate(direction_from)`
- `B = evaluate(direction_to)`

the stored heading semantics are:

```text
dx = B.x - A.x
dy = B.y - A.y
heading_deg = atan2(-dy, dx) in degrees
```

Implications:

- `0` degrees points to image +x (right)
- positive rotation is counter-clockwise in the usual math sense
- image y increases downward, so the formula negates `dy`

Readers and writers should evaluate this using an isotropic pixel coordinate
space such as:

- `keypoints_img`
- `keypoints_roi`

Do not compute candidate heading from `keypoints_norm` directly, because
non-square normalization can distort angles.

## Consumer Guidance

### Rendering

If heading metadata resolves and `enabled == true`:

- solid arrow: render from stored `heading` when finite
- arrow origin: evaluate `origin`

If heading metadata is absent or disabled:

- rendering stored `heading` is still allowed when present
- candidate-heading preview should be disabled unless the consumer explicitly
  enters a temporary legacy-compatibility mode

## Related Docs

- `docs/pose_heuristic_profile_contract.md`
- `docs/pose_schema_heuristics_split_proposal.md`

### Editing / dashed candidate preview

When a row is being edited:

1. determine whether the edited labels intersect `dependent_keypoints`
2. if not, do not recompute candidate heading
3. if they do intersect, recompute candidate heading only when all labels
   required by `origin`, `direction_from`, and `direction_to` are available and
   finite
4. if any required point is missing/invalid, suppress the dashed preview

This prevents edits to non-heading landmarks from changing heading previews.

## Writer Guidance

New keypoint/refined-keypoint runs that store meaningful `heading` should write
their canonical heading definition into `pose_schema.metadata.heading_computation`.

Only write `heading_computation_override` when a run must intentionally diverge
from the skeleton-level definition or explicitly disable heading semantics.

Legacy compatibility:

- old runs without this metadata remain valid
- consumers may temporarily fall back to the legacy 3-point fish rule for
  archives whose labels are exactly the starter skeleton
- new tooling should not depend on that legacy fallback

## Examples

### Example 1: current 3-point fish

Labels:

- `swim_bladder`
- `eye_left`
- `eye_right`

Metadata:

```json
{
  "version": 1,
  "enabled": true,
  "origin": {
    "op": "midpoint",
    "labels": ["eye_left", "eye_right"]
  },
  "direction_from": {
    "op": "keypoint",
    "label": "swim_bladder"
  },
  "direction_to": {
    "op": "midpoint",
    "labels": ["eye_left", "eye_right"]
  },
  "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"]
}
```

Meaning:

- arrow origin is between the eyes
- heading direction runs from swim bladder to eye midpoint
- editing any of those 3 points affects candidate heading

### Example 2: future skeleton with extra non-heading keypoints

Labels:

- `swim_bladder`
- `eye_left`
- `eye_right`
- `dorsal_fin_origin`
- `tail_tip`

Metadata:

```json
{
  "version": 1,
  "enabled": true,
  "origin": {
    "op": "midpoint",
    "labels": ["eye_left", "eye_right"]
  },
  "direction_from": {
    "op": "keypoint",
    "label": "swim_bladder"
  },
  "direction_to": {
    "op": "midpoint",
    "labels": ["eye_left", "eye_right"]
  },
  "dependent_keypoints": ["swim_bladder", "eye_left", "eye_right"]
}
```

Meaning:

- `dorsal_fin_origin` and `tail_tip` do not affect heading
- editing them should not trigger a dashed candidate-heading preview

### Example 3: explicit no-heading run

```json
{
  "version": 1,
  "enabled": false
}
```

Meaning:

- no candidate heading preview
- consumers should not infer heading semantics from label names

## Recommended Adoption Path

1. New keypoint and refined-keypoint runs should carry canonical
   `pose_schema.metadata.heading_computation` via their packaged pose schema.
2. Existing runs can be backfilled in place with
   `scripts/py -m fisheye.utils.backfill_keypoint_heading_computation`.
3. Use `heading_computation_override` only for explicit run-level differences
   or disable behavior.
4. Update Crimson and other consumers to resolve skeleton metadata first, with
   run override precedence.
5. Keep legacy hardcoded 3-point fallback only as a temporary compatibility
   mode for old archives.

## Related Documents

- `src/fisheye/docs/zarr_structure.md`
- `src/fisheye/docs/provenance_workflow.md`
- `docs/keypoint_multi_skeleton_todo.md`
- `docs/crimson_palette_integration_acceptance_checklist.md`
