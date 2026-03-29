# Track Kinematics and Swim Bout Status

Date anchored: 2026-03-06

Purpose: document the current implementation status of track-level kinematics,
distance/speed summaries, heading handling, and downstream swim-bout analysis so
we can decide what to stabilize next.

## Executive Summary

The current stack has a real upstream producer:

- [`src/fisheye/analysis/track_kinematics.py`](../src/fisheye/analysis/track_kinematics.py)

It writes rich per-track motion outputs under:

- `analysis/track_kinematics_runs/online/<run>/`
- `analysis/track_kinematics_runs/offline/<run>/`

For the current one-subject-per-arena workflow, the stack is conceptually in the
right place:

- `tracking_runs` now supplies real `track_id`
- `track_kinematics` is the canonical producer of speed, heading, and distance
  traces
- swim bouts sit downstream of track kinematics rather than raw arena labels

But the implementation is not yet fully coherent:

- speed and distance outputs are more standardized than downstream turning
  summaries
- there are two parallel swim-bout paths with overlapping responsibilities
- `track_kinematics` now materializes multiple tracks correctly, but downstream
  consumers are still mostly oriented around the current single-subject-per-
  arena workflow
- swim-bout mirroring currently copies one bout run into every track subgroup
  without proving that the bout data is actually track-specific

So the stack is usable for current single-track use, but it should not be
considered settled.

One important architectural boundary is now clearer than it was earlier in the
refactor:

- future skeleton-derived metrics should not all land in `track_kinematics`

See:

- [`pose_kinematics_run_design.md`](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)

That proposal keeps `track_kinematics` as the generic motion producer and makes
room for tail / fin / richer body geometry in a separate analysis layer.

## Current Producer: `track_kinematics`

Main module:

- [`src/fisheye/analysis/track_kinematics.py`](../src/fisheye/analysis/track_kinematics.py)

Core numeric helper:

- [`src/fisheye/analysis/compute_speed.py`](../src/fisheye/analysis/compute_speed.py)

Current offline lineage:

- detect / refined-detect rowset
- arena assignment
- tracking run
- track kinematics

Offline `track_kinematics` now resolves:

- one exact detection lineage
- one exact keypoint lineage
- one exact `tracking_runs/<run>`
- one exact `source_arena_assignment_run`

That is the right architectural direction. `track_kinematics` is no longer
grouping arena IDs directly in the main path.

## What `track_kinematics` Computes Today

### Speed and Distance

For each track, the current implementation computes four speed series:

- `speed_raw_*`
- `speed_filtered_*`
- `speed_smoothed_*`
- `speed_averaged_*`

It also computes three displacement series:

- `displacement_raw_*`
- `displacement_filtered_*`
- `displacement_smoothed_*`

And one cumulative distance series:

- `cumulative_distance_*`

These are stored in both pixel and millimeter space when calibration is
available.

Important behavior in
[`compute_track_speed(...)`](../src/fisheye/analysis/compute_speed.py):

- only consecutive frames contribute to displacement and speed
- non-consecutive frame jumps are treated as zero movement
- displacements larger than `500 px` are treated as suspicious and excluded
- optional hysteresis removes micro-jitter before smoothing
- temporal smoothing can be moving-average or Savitzky-Golay

This is conservative and generally reasonable for current data quality:

- it avoids inventing distance through gaps
- it avoids accumulating obvious teleports or resets

But it also means:

- the stack is not modeling missing-frame motion
- gap-heavy tracks will systematically undercount movement

### 2. Future skeleton-derived metrics need a separate home

The current module already covers generic per-track motion well enough for:

- position
- heading
- turning
- speed
- displacement
- acceleration

It should not become the first-class home for:

- tail segment angles
- tail curvature
- pectoral fin spread
- arbitrary joint or region geometry

Those metrics should be designed under:

- [`pose_kinematics_run_design.md`](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)

This keeps the base track contract stable while richer keypoint skeletons are
introduced.

### Heading

For each track, the current implementation stores:

- `heading_degrees`
- `heading_radians`
- `delta_heading_degrees`
- `angular_velocity_deg_s`
- `smoothed_heading_degrees`
- `smoothed_heading_radians`
- `heading_per_second_degrees`
- `heading_per_second_resultant`

The smoothed heading logic uses circular averaging over sine/cosine components,
which is the right basic approach for angular data.

The new turning arrays make successive-sample heading changes explicit, but the
downstream stack still does not yet standardize how turning should be summarized
or consumed in later stimulus-response analyses.

### Acceleration

For each track, the current implementation also stores:

- `acceleration_px`
- `acceleration_mm`
- `smoothed_acceleration_px`
- `smoothed_acceleration_mm`

These are derived from the smoothed speed trace rather than directly from the
position trace.

## What Is Missing or Weak in the Kinematics Layer

### 1. Distance semantics are conservative but not explicitly tiered

The current stack already distinguishes:

- raw displacement
- hysteresis-filtered displacement
- smoothed displacement

But downstream consumers can still easily confuse:

- "distance actually traversed in the raw signal"
- "distance after anti-jitter filtering"
- "distance after temporal smoothing"

The code is internally consistent, but the product contract is not yet explicit
enough about which of these should be treated as:

- canonical swim distance
- QC/debug only
- preferred input for bout detection

## Recently Fixed: Multi-Track Materialization in `build_track_datasets(...)`

Location:

- [`src/fisheye/analysis/track_kinematics.py`](../src/fisheye/analysis/track_kinematics.py)

Previous problem:

- the `for track_id in unique_ids:` loop computes basic per-track speed inputs
- but the smoothed heading, acceleration smoothing, per-second heading summary,
  final `tracks[int(track_id)] = {...}`, and `summary` append happen after the
  loop exits

That meant the function only materialized the last track processed.

Why this was not more obvious:

- current workflows often operate with one fish per dish and effectively one
  offline track
- in that case, "last track only" happens to be the only track

What changed:

- the multi-track loop was corrected, so all tracks are now persisted
- focused unit coverage now exercises multiple tracks in one run

What still matters:

- most downstream tools still effectively assume one selected track at a time
- the broader multi-track consumer contract is still not fully standardized

## Swim Bout Analysis: Current Split

There are currently two downstream bout paths.

### Path A: `detect_bouts_multi_level`

Main module:

- [`src/fisheye/analysis/detect_bouts_multi_level.py`](../src/fisheye/analysis/detect_bouts_multi_level.py)

This is the newer track-kinematics-based bout detector.

It:

- loads one selected `track_kinematics` track
- reads all four speed levels
- detects bouts on each level separately
- writes hierarchical results under `analysis/swim_bout_runs/<run>/`

Speed levels:

- `speed_raw`
- `speed_filtered`
- `speed_smoothed`
- `speed_averaged`

This path is useful because it treats bout detection as a consumer of the
canonical track kinematics artifact.

### Path B: `swim_bout_statistics`

Main module:

- [`src/fisheye/analysis/swim_bout_statistics.py`](../src/fisheye/analysis/swim_bout_statistics.py)

This is an older parallel path wrapping `EnhancedBoutAnalyzer`.

It is still valuable because it adds:

- global bout statistics
- per-trial bout summaries based on stimulus events
- calibration-aware unit overrides

But it is not the same contract as `detect_bouts_multi_level`.

So right now the repo has:

- one path centered on speed-level bout segmentation from track kinematics
- another path centered on summary statistics and trial segmentation

These are related, but not unified.

## Real Correctness Issue: Swim-Bout Mirroring Is Not Track-Aware

Location:

- [`_mirror_swim_bouts_to_tracks(...)`](../src/fisheye/analysis/track_kinematics.py)

Current behavior:

- resolve one `analysis/swim_bout_runs/<run>`
- iterate through all `tracks/id_*` groups in the current track-kinematics run
- copy the same bout payload into every track subgroup

This is only safe if the selected swim-bout run truly corresponds to that exact
track for every destination track, which the current code does not prove.

That means the mirrored per-track `swim_bouts/` subgroups are currently best
understood as convenience copies, not trustworthy track-specific lineage.

This should not be treated as final architecture.

## Where the Stack Is Strong Today

For the current one-fish-per-dish workflow, the stack already has several strong
points:

- `tracking_runs` now gives `track_kinematics` a correct upstream identity
  contract
- motion outputs are much richer than a simple "speed only" product
- circular heading smoothing is implemented rather than naive linear averaging
- distance computation is conservative in the presence of gaps and teleports
- multi-level bout detection is already using the track-kinematics product as an
  upstream source

So this is not a placeholder stack. It is a real analysis layer with a few
important unresolved seams.

## Where the Stack Is Not Yet Settled

The main unresolved issues are:

1. `track_kinematics` has a real multi-track implementation bug.
2. Bout analysis still exists in two partially overlapping paths.
3. Mirrored bout data inside `track_kinematics` is not currently track-safe.
4. The contract does not yet clearly separate:
   - canonical kinematic outputs
   - QC/debug variants
   - preferred downstream inputs for bout and stimulus-response analysis

## Recommended Fix Order

### 1. Fix `build_track_datasets(...)`

This is the highest-priority runtime fix.

Reason:

- it is a correctness bug, not just a design preference
- it affects any archive with more than one resolved track

### 2. Stop treating mirrored swim-bout data as authoritative

Either:

- remove the automatic mirror for now, or
- redesign it so a swim-bout run must explicitly bind to one exact
  `track_kinematics` run and one exact `track_id`

### 3. Decide how turning metrics should be summarized downstream

The per-frame turning arrays now exist in `track_kinematics`, but later
consumers still need a standard contract for:

- mean turning rate
- mean absolute turning rate
- onset turning rate
- binned turning summaries for stimulus-response analysis

### 4. Choose one canonical bout producer

Recommended direction:

- keep `detect_bouts_multi_level` as the canonical track-kinematics consumer for
  per-track bout segmentation
- keep `swim_bout_statistics` only if it becomes a summary/aggregation layer on
  top of canonical bout runs rather than a parallel producer with overlapping
  authority

### 5. Clarify canonical semantics for distance

The contract should explicitly say which series is the preferred one for:

- track-level distance summaries
- bout detection
- downstream stimulus-response metrics

## Bottom Line

The current speed/distance/heading stack is real and useful, especially for the
current one-subject-per-arena operating mode.

But it is not yet stable enough to treat as finished architecture.

The next work should not start by redesigning everything. It should:

1. fix the multi-track bug in `track_kinematics`
2. stop or redesign unsafe bout mirroring
3. add first-class turning metrics
4. collapse the two bout paths into a clearer producer/consumer split

## Related Docs

- [`tracking_runs_contract_status.md`](./tracking_runs_contract_status.md)
- [`single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)
- [`track_identity_target_architecture.md`](./track_identity_target_architecture.md)
- [`analysis_post_detection_workflow_status.md`](./analysis_post_detection_workflow_status.md)
- [`stimulus_response_run_design.md`](./stimulus_response_run_design.md)
