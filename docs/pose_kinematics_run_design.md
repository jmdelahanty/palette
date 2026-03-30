# Pose Kinematics Run Design

Date anchored: 2026-03-06

Purpose: define where richer skeleton-derived metrics should live once Palette
supports larger fish skeletons (for example tail landmarks or pectoral fin
landmarks), without overloading `track_kinematics` or forcing stimulus-response
analysis to become the first place that geometry is computed.

## Executive Summary

The current architecture already has a good base layer:

- `tracking_runs` resolves identity
- `analysis/track_kinematics_runs` computes generic whole-animal motion

That base layer should remain intentionally narrow:

- position
- heading
- speed
- displacement
- turning
- generic per-track summaries

Future skeleton-derived metrics should not all be added directly to
`track_kinematics`. Instead, the next analysis layer should be:

- `analysis/pose_kinematics_runs/<run>/`

This layer should consume refined keypoints plus track identity and produce
per-frame, identity-resolved geometric features derived from arbitrary skeleton
labels and segments.

Examples:

- tail segment angles
- tail curvature
- lateral tail displacement
- pectoral fin angles or spread
- body bend metrics
- skeleton-region validity / coverage metrics

Higher-level summaries such as tail-beat frequency, tail-beat amplitude, or
stimulus-locked tail responses should sit downstream of `pose_kinematics`,
either in dedicated runs or in `stimulus_response`.

## Why `track_kinematics` Should Stay Narrow

`track_kinematics` is strongest when it stays skeleton-agnostic and stable
across many data sources. Its role is to answer:

- where the tracked subject is
- how fast it is moving
- what direction it is facing
- how those quantities change over time

That remains valuable whether the recording has:

- no keypoints
- a starter 3-point skeleton
- a richer body + tail skeleton
- future fin landmarks

If we put every future tail or fin metric into `track_kinematics`, the base
contract becomes:

- harder to reason about
- more brittle across skeleton revisions
- full of optional arrays whose meaning depends on the active skeleton

That is the wrong place for skeleton-specific geometry.

## Proposed Layering

Recommended stack:

```text
arena_assignment
  -> tracking_runs
  -> analysis/track_kinematics_runs
  -> analysis/pose_kinematics_runs
  -> analysis/swim_bout_runs           (base movement bouts)
  -> analysis/tail_beat_runs           (future, optional)
  -> analysis/stimulus_response_runs
```

Important nuance:

- `eye_angle_runs` remains a specialized sibling analysis rather than being
  forced into `pose_kinematics`, because it depends on refined eye masks and
  ellipse fits, not just skeleton nodes.

So the more complete downstream picture is:

```text
tracking_runs -----------------------------> track_kinematics_runs
refined_keypoints_runs + tracking + track_kinematics -> pose_kinematics_runs
refined_eye_masks_runs + keypoints + track_kinematics -> eye_angle_runs

track_kinematics_runs -> swim_bout_runs
pose_kinematics_runs  -> tail_beat_runs (future)

stimulus_runs + track_kinematics_runs
           + swim_bout_runs
           + eye_angle_runs (optional)
           + pose_kinematics_runs (optional)
           -> stimulus_response_runs
```

## Contract Boundary

### `track_kinematics_runs`

This run should continue to own:

- track identity-aligned frame indices
- positions in pixels / millimeters
- heading
- delta heading
- angular velocity
- speed
- acceleration
- displacement
- cumulative distance

It should not become the catch-all home for:

- tail joint geometry
- fin geometry
- arbitrary node-to-node angles
- skeleton-specific oscillation metrics

### `pose_kinematics_runs`

This run should own:

- skeleton-derived geometry for one selected keypoint lineage
- label-based or segment-based per-frame metrics
- track-aligned outputs that remain valid when skeleton size grows from `K=3`
  to `K=N`

It should be the canonical place for metrics that depend on:

- named landmarks
- named edges / segments
- joint definitions
- body-region groupings

## Input Contract for `pose_kinematics_runs`

Direct inputs should be:

- one exact `tracking_runs/<run>`
- one exact `analysis/track_kinematics_runs/<type>/<run>`
- one exact `keypoints_runs/<run>` or `refined_keypoints_runs/<run>`
- skeleton metadata (`skeleton_id`, `pose_schema`, `kpt_shape`)

Recommended required attrs:

- `source_tracking_run`
- `source_track_kinematics_run`
- `source_keypoints_run` or `source_refined_keypoints_run`
- `source_skeleton_id`
- `source_pose_schema_sha256` if available

That keeps lineage explicit and prevents downstream consumers from inferring
which skeleton geometry was used.

## Recommended Storage Shape

The storage pattern should follow the existing analysis style:

```text
analysis/pose_kinematics_runs/<run>/
  attrs...
  tracks/
    id_<track_id>/
      frame_indices
      validity/
      segments/
      joints/
      regions/
      summaries/
```

Recommended principle:

- do not flatten every future metric into one giant top-level namespace

Instead, organize by semantic unit:

- `segments/`: orientation, angular velocity, length
- `joints/`: bend angles, curvature proxies
- `regions/`: tail, trunk, pectoral fins, dorsal fin, etc.
- `validity/`: which landmarks or regions were usable per frame
- `summaries/`: stable rollups for the run or track

This is more extensible than adding arrays such as:

- `tail_tip_angle_deg`
- `tail_base_angle_deg`
- `pectoral_left_angle_deg`
- `pectoral_right_angle_deg`

directly at the same level as generic track motion arrays.

## What Belongs in `pose_kinematics` Versus Downstream Runs

### Belongs in `pose_kinematics_runs`

Per-frame or near-primitive geometry:

- segment orientations
- joint angles
- curvature proxies
- tail midline displacement
- fin spread / fin angles
- per-region validity masks

### Belongs in downstream summary runs

Metrics that depend on temporal interpretation, event detection, or a stimulus
window:

- tail-beat frequency
- tail-beat amplitude
- tail-beat burst segmentation
- fin-beat frequency
- bout-phase tail metrics
- stimulus-locked tail or fin response summaries

These should either become:

- dedicated runs such as `analysis/tail_beat_runs`

or be consumed directly by:

- `analysis/stimulus_response_runs`

## Multi-Skeleton Implications

This design assumes the multi-skeleton direction already outlined in:

- [`docs/keypoint_multi_skeleton_todo.md`](./keypoint_multi_skeleton_todo.md)

Key requirement:

- all `pose_kinematics` computation must be label-based, not fixed-index based

That means:

- resolve landmarks by semantic label
- resolve segments and joints from `pose_schema`
- fail clearly when a requested metric requires labels not present in the
  current skeleton

Examples:

- a starter 3-point skeleton can still produce only whole-body motion
- an extended body + tail skeleton can produce tail-angle metrics
- a future fin-augmented skeleton can add pectoral metrics without redefining
  the `track_kinematics` contract

## Relationship to Existing Specialized Analyses

### `eye_angle_runs`

`eye_angle_runs` is already the right example of a specialized derived analysis:

- it consumes upstream aligned data
- computes its own domain-specific per-frame metrics
- persists a separate run with explicit provenance

`pose_kinematics_runs` should follow that architectural pattern, not try to
replace `eye_angle_runs`.

### `swim_bout_runs`

Base swim-bout detection should remain allowed to consume `track_kinematics`
alone. Later, richer body metrics can be joined in downstream analyses without
making basic bout detection depend on a large skeleton.

### `stimulus_response_runs`

`stimulus_response` should remain a consumer, not a geometry producer. It can
optionally consume `pose_kinematics_runs` when a protocol calls for tail or fin
metrics, but it should not be the first stage that computes those primitives.

## Suggested Next Design Decisions

Before any code is written, the next concrete choices should be:

1. Decide whether the canonical run name is exactly `pose_kinematics_runs`.
2. Define the minimum stable semantic groups under each track:
   `segments`, `joints`, `regions`, `validity`, `summaries`.
3. Decide whether tail-beat frequency belongs in `pose_kinematics` summaries or
   in a separate `tail_beat_runs` layer.
4. Define which metrics are considered:
   - skeleton-primitive
   - summary
   - stimulus-specific

## Recommendation

Adopt this boundary now, even before richer skeletons land:

- `track_kinematics` stays the generic motion layer
- `pose_kinematics_runs` becomes the future home for skeleton-derived geometry
- downstream runs consume those outputs as needed

That gives Palette a stable place to grow from a 3-point skeleton into richer
body, tail, and fin analyses without repeatedly redefining the base
track-kinematics contract.
