# Track Identity Target Architecture

Date anchored: 2026-03-06

Purpose: define the target model for spatial assignment and temporal identity
in fisheye so the pipeline can correctly support:

- one subject in one dish
- multiple subjects in one dish
- multiple dishes in one camera view
- multiple subjects across multiple dishes

This document is the proposed follow-on to
[`track_assignment_id_status.md`](./archive/track_assignment_id_status.md)
(archived; captured the pre-`tracking_runs` state — see
`single_subject_per_arena_tracking_contract.md` for current behavior).

## Executive Summary

The pipeline should stop using one integer namespace for both spatial
containment and temporal identity.

Target end state:

- `arena_id` means "which spatial container is this detection in?"
- `track_id` means "which subject trajectory does this detection belong to?"
- `dish_id` remains an external acquisition/registry identifier, not a local
  vision/tracking label
- `subject_id` remains optional experimental metadata, not something inferred
  automatically from image tracking alone
- when present in the registry, `subject_id` is the canonical biological
  identity key; legacy acquisition `fish_id` is only an alias/source name

In that model:

- the current arena-assignment stage is `arena_assignment_runs`
- `tracking_runs` becomes the true producer of `track_id`
- `track_kinematics_runs` consumes `tracking_runs`, not arena labels

## Design Goals

1. Make the ID model correct for both simple and multi-subject workflows.
2. Keep spatial and temporal semantics separate.
3. Make downstream analysis consume true tracks rather than ROI buckets.
4. Preserve a clean place for acquisition metadata such as `dish_id`.
5. Keep the contract simple enough that a trivial single-subject tracker and a
   full multi-subject tracker can share the same downstream interface.

## Entity Model

### `arena_id`

Local spatial container identifier within one archive/run.

Meaning:

- dish
- chamber
- lane
- subdish
- any other tuned analysis ROI that acts as a containment region

Properties:

- local to one recording/archive
- integer or compact string namespace is fine, but it should be one namespace
  per assignment run
- can carry metadata such as `arena_type`, `arena_name`, `dish_id`, geometry,
  and parent relationships

Non-goals:

- not a subject identity
- not a registry primary key

### `track_id`

Local temporal identity for one observed subject trajectory within one archive
and one tracking run.

Meaning:

- one fish over time
- one tracked animal over time

Properties:

- belongs to a tracking run, not to the detection stage directly
- should be stable across frames within that run
- should normally belong to exactly one `arena_id` for dish-based experiments

Non-goals:

- not a spatial ROI label
- not a dish identifier
- not an acquisition subject identifier unless explicitly linked later

### `dish_id`

External acquisition/registry identifier for a physical dish.

Properties:

- comes from acquisition metadata or registry data
- may be absent
- may map onto one arena or multiple sub-arenas depending on the experiment

### `subject_id`

External experimental identity for a biological subject when known.

Properties:

- canonical registry biological identity key
- should be modeled as metadata linkage, not as the runtime tracking namespace
- may be absent or ambiguous
- should not be required for `track_id` to exist

### `fish_id`

Legacy acquisition/snapshot name for the same biological concept.

Properties:

- may still appear in source metadata or provenance snapshots
- should map into canonical `subject_id` during registry normalization
- should not be treated as a second long-term identity namespace

## Scenario Matrix

### 1. One fish in one dish

- one `arena_id`
- one `track_id`
- optional one `dish_id`

### 2. Multiple fish in one dish

- one `arena_id`
- multiple `track_id`s
- optional one `dish_id`

This is the main case the current model does not represent correctly.

### 3. Multiple dishes in one camera view, one fish per dish

- multiple `arena_id`s
- one `track_id` per arena
- multiple optional `dish_id`s if acquisition metadata is available

### 4. Multiple dishes in one camera view, multiple fish per dish

- multiple `arena_id`s
- multiple `track_id`s per some arenas
- optional `dish_id` attached to arena metadata

### 5. Online target/chaser workflows

These should not reuse fish `track_id` semantics.

- keep `chaser_index`, `target_id`, or another dedicated namespace
- do not treat online target series as fish identity tracks

## Proposed Stage Model

### Stage A: Arena Definition

Purpose: define the spatial containers available in the recording.

Likely sources:

- `subdish_mask_tuning`
- single-dish masks
- chamber/lane ROI definitions
- future explicit arena-definition metadata

Suggested persisted metadata:

- `arena_id`
- `arena_type`
- `arena_name`
- geometry or mask definition
- optional `dish_id`
- optional parent relationships

This may continue to live under `analysis_metadata` or move into a more
explicit `arena_metadata` namespace later.

### Stage B: Arena Assignment

Target name: `arena_assignment_runs`

Purpose: map each detection row to a spatial container.

This is what the current arena-assignment stage actually does today.

Detection-level outputs:

- `arena_ids`
- optional `assignment_confidence`
- optional `assignment_reason`

Run-level attrs:

- `source_detect_run`
- `source_refined_run`
- `source_arena_definition`
- `assignment_method`
- arena metadata snapshot

Important rule:

- unassigned detections should remain representable, but as an explicit arena
  assignment state such as `-1` or a null-style encoding, not as a track

### Stage C: Tracking

Target name: `tracking_runs`

Purpose: map each detection row to a temporal subject identity.

Detection-level outputs:

- `track_ids`
- `arena_ids`
- optional `track_confidence`
- optional `track_status` or `tracking_reason`

Run-level attrs:

- `source_detect_run`
- `source_refined_run`
- `source_arena_assignment_run`
- `tracking_method`
- tracker parameters

This stage is where `track_id` becomes real.

#### Two tracking strategies should share the same contract

1. `single_subject_per_arena`

Use when the experiment guarantees at most one subject per arena.

Behavior:

- one `track_id` is created for each occupied `arena_id`
- if multiple simultaneous detections occur in one arena, the run should warn
  or fail depending on strictness settings

This strategy handles the simple current workflows cleanly without pretending
that arena labels are track identities.

Concrete first-pass contract:

- [`single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)

2. `multi_subject_within_arena`

Use when multiple subjects may coexist in one arena.

Behavior:

- tracker resolves temporal identity inside each arena
- multiple `track_id`s may map to the same `arena_id`

The key point is that both strategies write the same `tracking_runs` contract.

### Stage D: Track Kinematics

Target name: `analysis/track_kinematics_runs`

Purpose: compute per-track kinematics and summaries.

Inputs:

- `tracking_runs/<run>/track_ids`
- `tracking_runs/<run>/arena_ids`
- lineage to detect/refined-detect/keypoint stages as needed

Output semantics:

- group by `track_id`
- expose `arena_id` as track metadata and filtering context
- persist lineage to the exact `tracking_run` and `arena_assignment_run`

Important rule:

- `track_kinematics` should not infer tracks by grouping arena labels
- if true tracking is unavailable, a trivial tracker should run first

## Recommended Namespace And Naming

### Use these names consistently

- `arena_id`: spatial container identity
- `track_id`: temporal subject identity
- `dish_id`: external acquisition dish identifier
- `subject_id`: external biological subject identifier
- `fish_id`: legacy acquisition alias for `subject_id`, not a parallel
  registry namespace

### Avoid these overloaded usages

- using `track_id` to mean dish/ROI/subdish assignment
- using `dish_id` to mean local tuned ROI index
- using online target indices as fish `track_id`

## Proposed Zarr Contract Sketch

### `arena_assignment_runs/<run>/`

Required arrays:

- `arena_ids`

Suggested optional arrays:

- `assignment_confidence`
- `assignment_reason`
- `n_detections_per_arena`

Suggested attrs:

- `source_detect_run`
- `source_refined_run`
- `source_arena_definition`
- `assignment_method`
- `arena_definitions`

### `tracking_runs/<run>/`

Required arrays:

- `track_ids`
- `arena_ids`

Suggested optional arrays:

- `track_confidence`
- `tracking_reason`
- `frame_indices`
- `detection_indices`

Suggested attrs:

- `source_detect_run`
- `source_refined_run`
- `source_arena_assignment_run`
- `tracking_method`
- `tracking_parameters`

### `analysis/track_kinematics_runs/offline/<run>/`

Required lineage attrs:

- `source_tracking_run`
- `source_arena_assignment_run`

Suggested summary arrays:

- `track_ids`
- `track_arena_ids`

Per-track groups continue to live under:

- `tracks/id_<track_id>`

But those groups should represent real trajectories, not ROI buckets.

## Behavioral Rules

### One track should usually belong to one arena

For dish-based experiments, the default expectation should be:

- each `track_id` maps to exactly one `arena_id`

If a tracker emits one `track_id` spanning multiple arenas, that should be
treated as suspicious unless the workflow explicitly allows arena transitions.

### Lineage should be source-bound, not latest-bound

Both arena assignment and tracking should resolve inputs by exact provenance:

- `source_detect_run`
- `source_refined_run`
- `source_arena_assignment_run`

Downstream consumers should stop loading "latest compatible enough" runs.

### Observation rows are not identity edits

Detection and crop rows are observations. Arena assignment and tracking are
interpretation layers over those rows.

For review workflows, correcting an arena or track mistake should mutate the
assignment/tracking review run, not the bound observation rowset. The detailed
mutable-run policy is documented in
[`mutable_review_runs_contract.md`](./mutable_review_runs_contract.md).

### Unassigned is not a track

`arena_id == -1` or equivalent may be a useful diagnostic state.

But:

- unassigned rows should not become public `track_id`s by accident
- track construction should decide explicitly whether to drop, quarantine, or
  separately summarize unassigned detections

## Migration Direction

Because backward compatibility is not the priority here, the clean target
migration is:

1. Keep `arena_assignment_runs` as the spatial containment stage.
2. Stop describing arena assignment as tracking.
3. Make `tracking_runs` the required producer of `track_id`.
4. Update `track_kinematics` to require `source_tracking_run`.
5. Keep online target/chaser identity out of the fish `track_id` namespace.

## Immediate Implementation Consequences

If we adopt this design, the first concrete code tasks become clearer:

1. Use "arena assignment" consistently for the spatial containment stage.
2. Decide whether the current `tracking_runs` codepath is salvageable or should
   be replaced with a new tracker contract.
3. Change `track_kinematics` so it no longer groups directly by current
   assignment labels.
4. Add a trivial `single_subject_per_arena` tracking mode so simple workflows
   still work with the new contract.
5. Audit downstream tools so filtering and plotting can use both `track_id` and
   `arena_id` explicitly.

## Bottom Line

The durable model is:

- arena assignment answers "where is this detection?"
- tracking answers "which subject is this over time?"
- track kinematics answers "what did this tracked subject do?"

That separation handles all of the workflow shapes discussed so far without
forcing dish labels, ROI labels, and subject identities into the same integer
namespace.
