# GoodCopBadCop Escape/Freeze Canary Checklist
<!-- design-meta
status: draft
last_updated: 2026-06-23
-->

Purpose: define the first one-recording canary for a chaser-centric
escape/freeze diagnostic. This is the defensive-channel companion to the
existing CRA relocation and near-field analyses. The first goal is inspection,
not inference: generate a per-trial artifact for one recording that a human can
review before any response classifier or group statistics are locked.

## Current Data Constraint

For the current 32-fish GoodCopBadCop cohort, imported stimulus state supports
trial definition from controller logs, but active pursuit exists only for
`chaser_index=0`.

Observed in the expanded cohort scan:

```text
chaser0_trials: 8-16 per recording
chaser1_trials: 0 for all recordings
```

Therefore the first implementation must label the active-chase diagnostic as:

- US validation: does the aggressive chaser evoke escape/freeze-like behavior?
- Within-aggressive trend: does response mode change over trial ordinal?
- Not estimable from active pursuit in this cohort: aggressive-minus-benign
  active-chase specificity, because there are no benign pursuit trials.

The benign/static identity conditioned read remains a later extension.

## Storage Contract

Write the canary as a component under the selected chaser-distance run:

```text
analysis/chaser_distance_runs/<run>/chaser_escape_freeze/<component>/
```

Do not create a new top-level run family. The chaser-distance run is the parent
because it already owns the registered fish/chaser frame alignment.

Suggested component name for the first canary:

```text
canary_chaser0_escape_freeze_v1
```

## Stable Inputs

Use existing run-local data. Do not restart from raw H5 positions.

- `analysis/chaser_distance_runs/<run>/...`
  - fish centroid in registered arena coordinates
  - chaser positions in the same registered frame
  - frame ids, stimulus frame ids, timestamps
  - fish/chaser validity masks
  - fish-to-chaser distance in mm
- `analysis/stimulus_runs/<run>/tracking_data/chaser_states`
  - `chase_sequence_active`
  - `chase_trial_id`
  - `is_chasing`
  - `trial_state`
  - `loom_phase`
  - `loom_mode`
  - `chase_speed_mm_per_s`
  - chaser position fields
  - distance-to-target fields
- optional existing kinematics arrays if available, but the canary can compute
  frame-to-frame speed from registered fish positions.

Hard preconditions:

- Use registered arena-local canvas pixel positions from the selected
  chaser-distance run, then convert position deltas to mm with
  `pixels_per_mm_projector`.
- Define trials from controller state: `chase_sequence_active` and
  `chase_trial_id`.
- Do not infer pursuit bouts from fish or chaser kinematics.
- Fail loudly if frame alignment between chaser-distance arrays and
  `chaser_states` cannot be established.

## Canary Recording

Default canary candidate:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr
```

Known active-trial structure for this canary:

```text
chaser_index=0: 10 active pursuit trials
chaser_index=1: 0 active pursuit trials
```

## Trigger-Distance Metadata Finding

The canary zarr contains an explicit starting-distance value in both protocol
metadata and per-frame chaser state:

```text
analysis/stimulus_runs/stimulus_external_ipc_20260616_01
  attrs["protocol_json"]
    steps[0].parameters.chasers[0].initial_distance_mm = 20.0
    steps[0].parameters.chasers[1].initial_distance_mm = 20.0

analysis/stimulus_runs/stimulus_external_ipc_20260616_01/tracking_data/chaser_states
  initial_distance_mm = 20.0 for active chaser-0 pursuit frames
```

Nearby protocol parameters in the same metadata:

```text
positioning_distance_mm = 30.0
proximity_threshold_mm = 1.5
size_scaling_start_distance_mm = 20.0
size_scaling_full_distance_mm = 3.0
chaser_radius_mm = 2.0
target_radius_mm = 1.0
stop_at_target_edge = true
```

The initial canary used `trigger_radius_mm = 5.0`. That is likely too late for
response classification because the chaser is already close to target contact.
The chaser and target radii imply an edge-stop distance near 3 mm, so a 5 mm
trigger effectively aligns to near-contact rather than to the beginning of the
proximal aversive approach.

For the current canary, comparing first threshold crossing within each active
pursuit trial:

```text
trial 1: 20 mm at 0.20 s, 5 mm at 1.76 s
trial 2: 20 mm at 0.79 s, 5 mm never crossed
trial 3: 20 mm at 1.48 s, 5 mm at 2.05 s
trial 4: starts inside 20 mm and 5 mm
trial 5: 20 mm at 0.81 s, 5 mm at 1.26 s
trial 6: starts inside 20 mm, 5 mm at 0.26 s
trial 7: 20 mm at 0.02 s, 5 mm at 0.47 s
trial 8: starts inside 20 mm, 5 mm at 0.16 s
trial 9: starts inside 20 mm, 5 mm at 0.15 s
trial 10: starts inside 20 mm and 5 mm
```

Implemented cleanup:

- Default `trigger_radius_mm` is resolved from
  `chaser_states.initial_distance_mm` for the selected chaser when present.
- It falls back to `protocol_json.steps[*].parameters.chasers[chaser_index].initial_distance_mm`
  when the per-frame field is absent.
- Passing `--trigger-radius-mm` is treated as an intentional override; if no
  metadata is available, the canary falls back to the old 5 mm default and
  records a warning.
- The component stores the resolved value and source in config/provenance attrs
  and each trial row, for example:
  - `trigger_radius_mm = 20.0`
  - `trigger_radius_source = "chaser_states.initial_distance_mm"`
  - `trigger_radius_override = false`
- If a trial starts already inside the resolved trigger radius, use bout onset
  as the trigger and record `trigger_source = "bout_onset_already_inside_radius"`.
  This is not a failure; it means the chaser began pursuit from inside the
  nominal starting-distance threshold for that trial.
- If the first finite distance sample in a trial is already inside the radius,
  but earlier trial frames have invalid distance, record
  `trigger_source = "first_valid_already_inside_radius"` instead of
  `proximity`. That label means the 20 mm crossing was not observed in the
  offline distance trace, so the trigger is a conservative first-observable
  point rather than a true threshold-crossing event.

## Phase 1: Unclassified Diagnostic Component

Goal: write enough per-trial data and static/interactive artifacts to inspect
whether the transform and trial windows are behaviorally sensible.

Implementation checklist:

- Resolve the selected chaser-distance run and source stimulus run.
- Load dense fish centroid, chaser position, distance, validity, frame, and
  timestamp arrays from the chaser-distance run.
- Load `chaser_states` from the linked stimulus run.
- Align `chaser_states` to camera frames using existing stimulus-frame to
  camera-frame mapping logic.
- Filter to `chaser_index=0`.
- Define trials as contiguous stretches of `chase_sequence_active == true`,
  grouped by `chase_trial_id` when available.
- Store trial boundaries:
  - `trial_id`
  - `trial_ordinal`
  - `start_frame`
  - `end_frame`
  - `start_time_s`
  - `end_time_s`
  - `duration_s`
  - `trigger_frame_proximity`
  - `trigger_frame_bout_onset`
  - `trigger_distance_mm`
  - `trigger_radius_mm`
  - `trigger_radius_source`
  - `trigger_source`
- Compute chaser heading per frame:
  - prefer finite-difference of chaser position for the canary
  - smooth lightly if needed
  - hold the last valid heading through low-speed frames
  - record the low-speed/held-heading mask
- Implement the chaser-centric transform:
  - translate fish position by chaser position
  - rotate so chaser travel direction points to `+y`
  - keep original radius and angle for debugging
- Implement the stage-1 fish-centered diagnostic:
  - translate chaser position by fish position
  - do not rotate by fish heading yet
  - use arena/math axes: `+x` is right and corresponds to `0 deg`; `+y` is up
    and corresponds to `90 deg`
  - store `chaser_x_fish_centered_mm` and `chaser_y_fish_centered_mm` in the
    per-frame trial trajectory table
  - render `escape_freeze_fish_centered_diagnostic_png` with the fish at the
    origin, 3 mm/20 mm rings, trigger marker, and candidate response labels
- Validate the transform with a synthetic unit test:
  - scripted chaser path
  - scripted fish trajectory moving directly away
  - expected transformed trajectory points upward/away with increasing radius
- For each trial, compute unclassified response metrics:
  - response-window frame count
  - freeze-window frame count
  - baseline speed over the pre-trigger baseline window
  - mean speed in response window
  - max windowed speed, labeled diagnostic only
  - net displacement in mm
  - radial excursion in mm
  - path length in mm
  - full-trial fish path length in mm
  - full-trial fish net displacement in mm
  - full-trial mean/max speed
  - full-trial low-speed fraction
  - cumulative absolute distance-to-chaser change, labeled diagnostic only
  - escape bearing in chaser frame
  - fraction of freeze-window frames below candidate freeze speed
  - longest low-speed run in seconds
  - tracking dropout fraction in each window
- Add candidate canary labels using full-trial fish path length:
  - default `escape_path_threshold_mm = 40.0`
  - `candidate_response_class = "escape_attempt"` when full-trial fish path
    length is at or above the threshold
  - `candidate_response_class = "not_escape"` below threshold
  - visual labels may render `not_escape` as `not_escape / freeze_candidate`
    to reflect the likely behavior without treating freeze as a locked class
  - keep `classification_locked = false` until this rule is reviewed across
    more recordings/fish
  - do not classify on distance-to-chaser change alone because a stationary fish
    can show large distance changes when the chaser moves toward it
- Render `escape_freeze_response_class_bar_png`:
  - x-axis labels: `escape_attempt`, `not_escape / freeze_candidate`
  - y-axis: trial count
  - annotate each bar with count and percent
  - title/caption must state the full-trial path threshold and that the
    classifier is not cohort-locked
- Render `escape_freeze_trial_outcome_timeline_png`:
  - x-axis: time in recording, using each trial's trigger time for the marker
  - show light horizontal spans for each active chase bout duration
  - marker shape/color: red triangle for `escape_attempt`, gray circle for
    `not_escape / freeze_candidate`
  - annotate each marker with trial ordinal
  - title/caption must state that labels are candidate labels from the full-trial
    path threshold, not cohort-locked classifications
- Render fish-centered polar approach summaries:
  - `escape_freeze_fish_centered_polar_approach_png`: transparent frame points
    pooled across active pursuit, colored by trial ordinal
  - `escape_freeze_fish_centered_polar_density_png`: polar-binned frame density
    using the same fish-centered coordinates
  - polar convention: `0 deg` is right, `90 deg` is up
  - draw distance rings at near-contact, intermediate distances, and the
    metadata-derived chaser starting/trigger radius
- Store all metrics without assigning `escape/freeze/none`.

Suggested component arrays/tables:

```text
trials/
  trial_id
  trial_ordinal
  chaser_index
  start_frame
  end_frame
  trigger_frame_proximity
  trigger_frame_bout_onset
  trigger_distance_mm
  duration_s
  valid_response_fraction
  valid_freeze_fraction

trial_metrics/
  trial_id
  baseline_mean_speed_mm_s
  response_mean_speed_mm_s
  response_max_windowed_speed_mm_s
  net_displacement_mm
  radial_excursion_mm
  path_length_mm
  escape_bearing_deg
  freeze_low_speed_fraction
  longest_low_speed_run_s

trial_trajectories/
  trial_id
  relative_frame
  camera_frame_id
  time_from_trigger_s
  fish_x_chaser_frame_mm
  fish_y_chaser_frame_mm
  distance_mm
  fish_speed_mm_s
  chaser_heading_deg
  heading_held
  valid
```

Trajectory coordinate metadata:

- `fish_x_chaser_frame_mm`, `fish_y_chaser_frame_mm`, and `bearing_deg` are in
  the `chaser_centric_mm` frame:
  - `+y` is chaser heading forward;
  - `+x` is right relative to chaser heading;
  - `bearing_deg = 0` is chaser heading forward;
  - positive `bearing_deg` is right relative to chaser heading.
- `chaser_x_fish_centered_mm` and `chaser_y_fish_centered_mm` are in the
  `fish_centered_world_mm` diagnostic frame:
  - fish at origin;
  - `+x` right;
  - `+y` up;
  - no fish-heading rotation.

## Phase 2: Canary Figures

Generate figures from stored canary data, not from a separate visualization-only
calculation.

Required canary artifacts:

- Per-trial chaser-centric diagnostic PNG:
  - one row per trial
  - chaser at origin
  - chaser heading up
  - distance rings
  - trajectory onset and end markers
  - paired distance-from-chaser trace
  - trial ordinal and trial id labels
- Speed-vs-displacement scatter PNG:
  - one point per active pursuit trial
  - x = `net_displacement_mm` or `radial_excursion_mm`
  - y = `response_mean_speed_mm_s`
  - trial ordinal encoded by color or label
  - no classifier threshold drawn unless explicitly supplied by config
- Trial metric table view in Palette Explorer or a static HTML/PNG summary.

Optional canary artifacts:

- Interactive Plotly version of the per-trial diagnostic.
- Population trend placeholder for one recording:
  - response metrics vs trial ordinal
  - labeled as single-recording diagnostic only.

## Phase 3: Classifier Locking, Not Part Of First Canary

Do not classify trials in the first pass unless thresholds are explicitly
provided as config.

Later classifier config should include:

- response window seconds
- freeze window seconds
- baseline window seconds
- proximity trigger radius
- net displacement threshold
- radial excursion threshold
- response mean speed threshold
- freeze speed threshold or baseline fraction
- minimum freeze duration
- directed-away bearing bounds
- held-out split assignment if used

Thresholds must be recorded in component attrs and provenance. They must not be
chosen by optimizing a group p-value.

## Phase 4: Future Group Export

Only after the canary diagnostic is trusted, add export tables:

```text
goodcopbadcop_escape_freeze_trials
goodcopbadcop_escape_freeze_trial_metrics
goodcopbadcop_escape_freeze_fish_summary
```

For the current cohort, group summaries should expose:

- active-chase US validation for chaser 0
- within-aggressive trial-ordinal trend
- explicit `benign_active_pursuit_available = false`
- no active-pursuit aggressive-minus-benign Wilcoxon

## Acceptance Criteria For The Canary

- One selected recording writes a complete
  `chaser_escape_freeze/canary_chaser0_escape_freeze_v1` component.
- Trial count matches controller state from `chase_sequence_active` and
  `chase_trial_id`.
- Synthetic transform test passes.
- Per-trial diagnostic figure is inspectable and visually aligned:
  chaser at origin, heading up, fish trajectory in the chaser frame.
- Speed-vs-displacement scatter is produced without classifier thresholds by
  default.
- Output clearly labels this as diagnostic/US-validation, not conditioned
  specificity.
- The component records measurement caveats:
  100 FPS, no C-start kinematics, no sub-30 ms latency claims, windowed speed
  only.
