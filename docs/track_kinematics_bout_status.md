# Track Kinematics and Swim Bout Status

Date anchored: 2026-03-06

Last reviewed: 2026-04-26

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
- `track_kinematics` now materializes multiple tracks correctly, but many
  downstream consumers are still mostly oriented around the current
  single-subject-per-arena workflow
- swim-bout mirroring currently copies one bout run into every track subgroup
  without proving that the bout data is actually track-specific
- `stimulus_response` expands sparse track data to dense arrays, but distance
  summaries must preserve `track_kinematics` gap semantics rather than
  recomputing path distance across missing-frame gaps

So the stack is usable for current single-track use, but it should not be
considered settled.

One important architectural boundary is now clearer than it was earlier in the
refactor:

- future skeleton-derived metrics should not all land in `track_kinematics`

See:

- [`pose_kinematics_run_design.md`](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)

## 2026-04-26 Review Update

The current design direction is sound:

- `track_kinematics` should remain the canonical generic movement producer.
- `detect_bouts_multi_level` should become the canonical per-track bout
  segmentation producer.
- `swim_bout_statistics` should become a downstream summary or aggregation
  layer, not a second competing bout source of truth.
- `stimulus_response` should consume movement, stimulus, bouts, and optional eye
  angle or pose-derived metrics without redoing identity resolution.

The highest-priority correctness issue is now downstream gap handling, not basic
track materialization. `track_kinematics` deliberately treats non-consecutive
frames conservatively when computing frame path distance and cumulative path
distance. Any consumer that expands sparse tracks to dense frame arrays must
either consume those source path-distance arrays or reproduce the same
consecutive-frame rules. It should not compute distance by taking `np.diff(...)`
across only the valid positions in a time window, because that can invent
movement across gaps.

The next contract for this gap-handling work is
[`track_validity_timeline_design.md`](./track_validity_timeline_design.md). That
doc defines the planned per-track validity arrays, transition reason codes,
swim-bout metric implications, and future plot overlays for invalid detections
or movement gaps.

`detect_bouts_multi_level` now uses that direction for new outputs. Bout tables
write explicit `path_length_*`, `net_displacement_*`,
`observed_duration_s`, `valid_transition_fraction`, and `gap_censored` fields
instead of the older ambiguous `distance = mean_speed * duration` estimate.
Path length is summed from `track_kinematics` frame path-distance arrays and is
not back-estimated from speed when those source arrays are unavailable.

The second-priority correctness issue remains swim-bout mirroring. Mirrored
`swim_bouts/` groups inside a `track_kinematics` run should not be treated as
authoritative unless the mirrored bout run proves both:

- the same source `track_kinematics` run
- the same destination `track_id`

Until that is enforced, `analysis/swim_bout_runs/<run>/` should be treated as
the canonical bout artifact.

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

It also computes three frame path-distance series:

- `frame_path_distance_raw_*`
- `frame_path_distance_filtered_*`
- `frame_path_distance_smoothed_*`

And one cumulative path-distance series:

- `cumulative_path_distance_*`

These are stored in both pixel and millimeter space when calibration is
available.

This schema is intentionally strict after the path-distance cleanup: current
consumers expect `frame_path_distance_*` and `cumulative_path_distance_*`, not
the earlier `displacement_*` or `cumulative_distance_*` names. Existing canary
runs written with the old names should be regenerated from `track_kinematics`
rather than silently read through compatibility fallbacks.

Important behavior in
[`compute_track_speed(...)`](../src/fisheye/analysis/compute_speed.py):

- only consecutive frames contribute to frame path distance and speed
- non-consecutive frame jumps are treated as zero movement
- frame path-distance increments larger than `500 px` are treated as suspicious and excluded
- optional hysteresis removes micro-jitter before smoothing
- temporal smoothing can be moving-average or Savitzky-Golay
- moving-average smoothing can be `centered` or `causal`

This is conservative and generally reasonable for current data quality:

- it avoids inventing distance through gaps
- it avoids accumulating obvious teleports or resets

But it also means:

- the stack is not modeling missing-frame motion
- gap-heavy tracks will systematically undercount movement
- centered smoothing can leak future motion backward in time, so onset-sensitive
  bout segmentation should use `speed_filtered` or causal smoothing

### Tuning Hysteresis and Bout Segmentation

For bout tuning, treat `track_kinematics` parameters and swim-bout segmentation
parameters as separate candidate runs:

- `track_kinematics` controls the source speed traces, including hysteresis
  thresholds and smoothing window.
- `detect_bouts_multi_level` consumes those traces and writes bout intervals for
  `speed_raw`, `speed_filtered`, `speed_smoothed`, and `speed_averaged`.
- the swim-bout run's `default_level` declares which speed subgroup downstream
  consumers should use when they do not explicitly request a level.

For recordings where the smoothed trace over-broadens bouts, prefer detecting
and displaying bouts from `speed_filtered`. In this mode, hysteresis has already
zeroed non-motion, so the bout threshold should be a small epsilon rather than a
biological speed cutoff:

```bash
scripts/py -m fisheye.analysis.track_kinematics <archive.zarr> \
  --offline-only \
  --hysteresis-high-px 8 \
  --hysteresis-low-px 4 \
  --hysteresis-min-frames 3 \
  --smooth-seconds 0.10 \
  --smoothing-alignment causal \
  --offline-run-name tk_hyst8_low4_s010

scripts/py -m fisheye.analysis.detect_bouts_multi_level <archive.zarr> \
  --track-kinematics-run tk_hyst8_low4_s010 \
  --run-name bouts_tk_hyst8_low4_s010_filtered \
  --threshold-mm 0.01 \
  --default-level filtered \
  --boundary-mode threshold \
  --overwrite
```

For iterative tuning, create one named pair of runs per candidate rather than
rewriting one generic run in place. When regenerating the same candidate after a
schema or implementation change, reuse the candidate name with
`detect_bouts_multi_level --overwrite` so the explorer stays clean and does not
mix stale and current derived bout runs. A practical first sweep is:

```text
hysteresis_high_px=4, hysteresis_low_px=2, smooth_seconds=0.05
hysteresis_high_px=6, hysteresis_low_px=3, smooth_seconds=0.05
hysteresis_high_px=8, hysteresis_low_px=4, smooth_seconds=0.05
hysteresis_high_px=8, hysteresis_low_px=2, smooth_seconds=0.05
```

Then compare candidates in the Marimo track explorer with:

```bash
scripts/py -m marimo run apps/marimo/track_kinematics_explorer.py -- \
  --zarr-path <archive.zarr>
```

The explorer treats the `track_kinematics` run as the top-level selection. After
selecting a track run, it lists only swim-bout runs whose metadata says they
were derived from the same `source_track_kinematics_run` and `track_id`. That
keeps candidate comparisons aligned with the actual dependency graph instead of
requiring operators to manually pair run names.

`detect_bouts_multi_level` stores two boundary concepts when
`--boundary-mode local_minimum` is enabled:

- `core_start_*` / `core_end_*`: the threshold-crossing or peak-width bout core
- `start_*` / `end_*`: the expanded onset/offset found by searching for nearby
  local speed minima within `--boundary-window-s`

The explorer overlays `start_time_s` to `end_time_s`, so local-minimum mode
should visually capture the full rise and decay around the core threshold
segment while still preserving the core fields for stricter quantitative
analyses.

The explorer exposes each stored speed-level subgroup as its own derived
swim-bout candidate while keeping the Zarr storage hierarchical. Selecting a
candidate such as `filtered` or `smoothed` reads that candidate's `bouts` and
`inter_bout_intervals` tables directly from
`analysis/swim_bout_runs/<run>/<speed_level>/`. Its histogram panel is a
pandas/Plotly view over those persisted fields, not a new recomputation from the
speed trace. Current histogram metrics include bout duration, observed bout
duration, path length, net displacement, and inter-bout interval.

Downstream per-bout heading metrics should not be added back into
`analysis/swim_bout_runs`. They belong in linked
`analysis/bout_kinematics_runs` outputs so the segmentation candidate remains an
immutable source artifact. See
[bout_kinematics_run_design.md](bout_kinematics_run_design.md).

For the current 2026-01-28 arena 2 canary review, `tk_hyst4_low2_s005` is the
preferred default candidate when it exists. This is a review default for the
current tuning pass, not a repository-wide conclusion that those thresholds are
optimal for all recordings. Pass `--run-path <run>` to open a different
candidate explicitly.

The explorer writes performance events to
`/tmp/palette_track_kinematics_explorer_perf.jsonl` by default. Use
`--performance-log <path>` to choose another JSONL file or
`--performance-log none` to disable logging.

On the 2026-01-28 arena 2 canary, the first performance log showed that
candidate switching is dominated by time-series figure construction, not Zarr
IO. A 517-bout overlay spent about `49 s` in `build_timeseries_figure`, while
Zarr loading took about `0.2 s`. The explorer now batches swim-bout overlays
into one translucent Plotly bar trace instead of drawing one `vrect` layout shape
per bout; the same 517-bout candidate measured about `0.01 s` for
`build_timeseries_figure` after that change. WebGPU-style renderers should
remain a future viewer-backend evaluation, not the first fix for the current
Plotly app.

### 2. Future skeleton-derived metrics need a separate home

The current module already covers generic per-track motion well enough for:

- position
- heading
- turning
- speed
- frame path distance
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

- raw frame path distance
- hysteresis-filtered frame path distance
- smoothed frame path distance

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

1. Stimulus-response distance summaries must preserve the gap-aware movement
   semantics produced by `track_kinematics`.
2. Bout analysis still exists in two partially overlapping paths.
3. Mirrored bout data inside `track_kinematics` is not currently track-safe.
4. Downstream consumers are not yet fully standardized for multi-track use.
5. The contract does not yet clearly separate:
   - canonical kinematic outputs
   - QC/debug variants
   - preferred downstream inputs for bout and stimulus-response analysis

## Recommended Fix Order

### 1. Preserve gap-aware distance in downstream consumers

This is now the highest-priority runtime fix.

Reason:

- `track_kinematics` already defines conservative distance semantics
- downstream consumers should not invent distance through missing-frame gaps
- stimulus-response summaries will otherwise disagree with the canonical
  movement artifact

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

1. fix downstream distance handling so consumers preserve gap-aware movement
2. stop or redesign unsafe bout mirroring
3. add first-class turning summaries for stimulus-response use
4. collapse the two bout paths into a clearer producer/consumer split
5. move future skeleton/body-specific metrics to `pose_kinematics_runs` or
   shape-specific analysis layers rather than expanding `track_kinematics`

## Related Docs

- [`tracking_runs_contract_status.md`](./tracking_runs_contract_status.md)
- [`single_subject_per_arena_tracking_contract.md`](./single_subject_per_arena_tracking_contract.md)
- [`track_identity_target_architecture.md`](./track_identity_target_architecture.md)
- [`analysis_post_detection_workflow_status.md`](./analysis_post_detection_workflow_status.md)
- [`stimulus_response_run_design.md`](./stimulus_response_run_design.md)
