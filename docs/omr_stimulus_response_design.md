# OMR Metrics in Stimulus Response Runs

Status: design plus initial single-fish implementation.

## Scope

OMR responsiveness should live inside `analysis/stimulus_response_runs/<run>`.
The stimulus-response run is the broad derived-analysis category for
stimulus-aligned behavior. OMR is one metric family inside that category,
primarily for `MOVING_GRATING` steps.

This means:

- Do not create a separate top-level `analysis/omr_response_runs` family for
  the first implementation.
- Reuse the existing `stimulus_response_runs` source contracts: stimulus run,
  track kinematics run, optional swim-bout run, and future optional eye/shape
  runs.
- Add OMR-specific outputs under moving-grating step groups and aggregate OMR
  summaries under the run's `global/` group.

## Literature Grounding

The literature does not define one universal OMR index. It uses several related
metrics depending on assay geometry, whether fish are freely swimming or
head-fixed, and whether the analysis is group-level, per-fish, per-trial, or
per-bout.

| Source | Relevant convention | Implication for Palette |
|--------|---------------------|--------------------------|
| Muto et al. 2005, PLOS Genetics, "Forward Genetic Analysis of Visual Behavior in Zebrafish" | High-throughput OMR screening used group movement in a racetrack-style assay and ratio-style group indices, not the same thing as a per-fish `N_correct / N_total` bout metric. | Do not treat the Muto screen as the direct schema for individual fish. It supports having a direction-normalized OMR index, but not only a bout fraction. |
| Severi et al. 2014, Neuron, "Neural Control and Modulation of Swimming Speed in the Larval Zebrafish" | OMR speed modulation is tied to bout structure, especially bout duration and interbout interval, not only instantaneous speed. | OMR outputs should preserve per-bout and IBI-compatible fields, not just whole-recording drift. |
| Naumann et al. 2016, Cell, "From Whole-Brain Data to Functional Circuit Models: The Zebrafish Optomotor Response" | Uses virtual-reality OMR behavior to connect stimulus speed/direction to circuit models. | Gain and stimulus-locked temporal traces remain important; these overlap with the existing `grating/` metrics. |
| Holman et al. 2023, PLOS Computational Biology, translational OMR model | Defines simulation and analysis metrics such as OMR ratio, fish velocity, stimulus velocity, and direction-dependent components. | Support displacement/velocity projection metrics alongside bout metrics. |
| Bahl and Engert 2020, Nature Neuroscience / related Engert-lab OMR decision work | Head-fixed decision assays often summarize performance as trial or bout direction correctness. | Palette can support fraction-correct analogs, but should label them as bout/trial choice metrics rather than displacement metrics. |
| Krishnan et al. 2025, Science Advances, "Attentional switching in larval zebrafish" | Trial-level performance score and responsiveness categories are useful for distributional analyses, including responder/non-responder questions. | Bimodality analysis should be downstream and configurable; Palette should store values needed for threshold-free export. |
| Marques et al. 2018, Current Biology, zebrafish locomotor repertoire | Bout classes such as forward scoots and routine turns are central to interpreting OMR engagement. | Bout classification should be linkable later, but OMR v1 should not require a classifier. |

References:

- Muto et al. 2005: https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.0010066
- Holman et al. 2023: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010924
- Severi et al. 2014: https://doi.org/10.1016/j.neuron.2014.06.022
- Naumann et al. 2016: https://doi.org/10.1016/j.cell.2016.02.001
- Bahl and Engert 2020: https://doi.org/10.1038/s41593-019-0534-9
- Krishnan et al. 2025: https://doi.org/10.1126/sciadv.ads4994
- Marques et al. 2018: https://doi.org/10.1016/j.cub.2018.09.002

## Existing Palette Surfaces

`stimulus_response_runs` already has the right parent shape:

```text
analysis/stimulus_response_runs/<run>/
  frames/
  global/
  steps/
    step_<i>/
      per_fish/
      per_bout/
      grating/
        per_frame/
        per_fish/
        time_series/
        per_bout/
```

Existing moving-grating outputs already include:

- `grating/per_frame/alignment_angle_deg`
- `grating/per_frame/alignment_cos`
- `grating/per_frame/speed_along_grating_mm_s`
- `grating/per_fish/mean_alignment_cos`
- `grating/per_fish/speed_weighted_alignment`
- `grating/per_fish/optomotor_gain`
- `grating/per_fish/drift_along_grating_mm`
- `grating/time_series/optomotor_gain`
- `grating/per_bout/mean_alignment_cos`

OMR should build on these, but it needs clearer responsiveness outputs:

- direction-normalized displacement indices,
- bout fraction-correct analogs,
- per-bout displacement projection scores,
- windowed OMR summaries suitable for distribution and bimodality analysis,
- explicit detector-vs-estimator provenance.

## Detector vs Estimator Hygiene

Bout detection and physical measurement must stay separate.

Detector sources:

- Bout boundaries come from `analysis/swim_bout_runs/<run>`.
- If a bout run uses `speed_exponential` or another detector response, that
  signal only defines event timing.
- The detector response must not be used as the physical speed, distance, or
  displacement estimator.

Estimator sources:

- Per-frame and per-bout displacement should be measured from physical
  position arrays in mm.
- Per-frame path length should use the physical movement source recorded by
  `track_kinematics`, preferably hysteresis-filtered path increments.
- Speed thresholds for time-moving metrics should use a physical speed trace,
  not the exponential detector response.

The output schema should record:

- `source_track_kinematics_run`
- `source_bout_run`, if used
- `bout_detector_signal`, for example `speed_exponential`
- `movement_metric_source_level`, for example `filtered`
- `position_source_array`, for example `positions_mm`
- `position_anchor`, for example `body_origin`, `eye_midpoint`,
  `swim_bladder`, or `bbox_origin_legacy`

The `position_anchor` field matters because older track-kinematics runs may
store positions derived from a detection bounding-box origin. That is useful
for debugging but not ideal as an analysis-grade biological anchor. OMR runs
should either use a biologically meaningful position source or clearly mark
the anchor semantics.

## Stimulus Direction Handling

OMR metrics need a stimulus direction vector in camera/world metric space:

```text
s_hat[t] = normalized direction of grating drift in camera/mm coordinates
```

Current static moving-grating support resolves direction from step parameters
such as `orientation_degrees`, then applies
`camera_to_projector_offset_deg`. This is enough for non-reactive grating
steps once calibration is validated.

Direction values must be interpreted in the same coordinate space as the fish
trajectory. Citrus stores grating direction in stimulus/projector convention,
while OMR projections are computed from camera-space mm positions. The
`camera_to_projector_offset_deg` CLI/attr name is retained for compatibility,
but operationally it is the angular correction applied to the stored stimulus
direction before comparing it with camera-space motion. The corrected value is
wrapped to `[0, 360)` before persistence.

The current moving-grating canary was recorded with an inverted projector
orientation: Citrus `0 deg` moved left in the camera view, not right. That run
must therefore use `camera_to_projector_offset_deg = 180.0`, so step directions
are interpreted as camera-space left/right correctly. A rig where Citrus
directions are already camera-space should use `0.0`.

For the first implementation:

- Static moving-grating steps may store direction as step attrs:
  `grating_direction_deg`, `stimulus_direction_xy`, and
  `stimulus_direction_source = "static_step_params"`.
- Reactive moving-grating steps should be excluded or marked unsupported unless
  frame-level direction is available.

Future reactive-compatible schema:

```text
steps/step_<i>/grating/per_frame/
  stimulus_direction_xy        float32[n_fish, n_step_frames, 2] or float32[n_step_frames, 2]
  stimulus_active              bool[n_step_frames]
  stimulus_epoch_id            int32[n_step_frames]
```

For single-direction static steps, repeating the direction per frame is not
required. The implementation can broadcast from attrs. If frame-varying
direction is introduced later, the same OMR formulas still apply.

### Canary Direction Audit

On 2026-05-03, the moving-grating canary
`2026-01-28T19-22-28Z_arena_1_DefaultScreen_analysis.zarr` was audited against
Crimson video playback. The fish visually moved to camera-left while the
grating moved left. This supports using the 180 deg projector correction for
this recording:

```text
Citrus/projector 0 deg + correction 180 deg => camera-left
Citrus/projector 180 deg + correction 180 deg => camera-right
```

The regenerated OMR run records:

```text
analysis/stimulus_response_runs/
  stimulus_response_tk_hyst4_low2_latch_s005_omr_canary

step 0 stimulus_direction_deg = 180.0
step 4 stimulus_direction_deg = 0.0
```

The visual/path interpretation and the unweighted bout-count interpretation
diverge in this canary; see the bout-fraction section below. That divergence
does not by itself invalidate the 180 deg direction correction.

## Metric Definitions

Let:

```text
dx_t = position_mm[t] - position_mm[t - 1]
d_t  = ||dx_t||
s_t  = stimulus direction unit vector at frame t
p_t  = dot(dx_t, s_t)
```

Only valid transitions are included. A transition is valid when both adjacent
positions are valid, the stimulus is active, the track identity is stable, and
the transition does not cross an unsupported gap.

### 1. Path-Normalized Displacement OMR Index

Formula:

```text
omr_path_index = sum(p_t) / sum(d_t)
```

Range: `[-1, +1]`

Chance level: `0`

Interpretation:

- `+1`: all path displacement is in the grating direction.
- `0`: no directional bias, or equal positive/negative displacement.
- `-1`: all path displacement is opposite the grating direction.

Edge cases:

- If `sum(d_t) <= eps`, store `NaN` in numeric arrays and record
  `no_movement` in reason/quality metadata.
- If coverage is below threshold, compute the value but mark quality as
  `low_coverage` unless the user chooses strict invalidation.

This is the preferred whole-step freely swimming OMR index for Palette v1.

### 2. Net-Displacement Direction Index

Formula:

```text
net_dx = position_mm[last_valid] - position_mm[first_valid]
omr_net_direction_index = dot(net_dx, mean_s_hat) / ||net_dx||
```

Range: `[-1, +1]`

Chance level: `0`

Interpretation:

This measures the direction of net displacement, not how much path was spent
following. It ignores tortuosity, so it should be reported alongside
`path_length_mm` and `net_displacement_mm`.

Edge cases:

- Undefined if `||net_dx|| <= eps`.
- For steps with changing stimulus direction, use the path-normalized index
  instead unless a clear epoch split exists.

### 3. Bout-Fraction OMR Performance

For each bout:

```text
b_i = position_mm[bout_end] - position_mm[bout_start]
q_i = dot(b_i, s_i) / ||b_i||
```

Classify:

```text
correct   if q_i >  projection_deadzone
opposing  if q_i < -projection_deadzone
ambiguous otherwise
```

Metrics:

```text
bout_fraction_correct_classified = N_correct / (N_correct + N_opposing)
bout_fraction_correct_all        = N_correct / N_total_bouts
bout_choice_index                = (N_correct - N_opposing) / (N_correct + N_opposing)
```

Ranges:

- `bout_fraction_correct_*`: `[0, 1]`, chance `0.5`
- `bout_choice_index`: `[-1, +1]`, chance `0`

Edge cases:

- If no classifiable bouts exist, store `NaN`.
- Keep `N_total_bouts`, `N_correct`, `N_opposing`, and `N_ambiguous` so
  downstream analyses can choose their own denominator.

This is the freely swimming analog of head-fixed fraction-correct or
performance-score metrics.

Important limitation: this is an **unweighted bout-count** metric. Every
classifiable bout contributes one vote, regardless of bout path length,
displacement magnitude, duration, or whether it is a small corrective movement.
It can therefore disagree with the visually obvious trajectory direction and
with path-weighted OMR metrics. In the 2026-01-28 moving-grating canary, the
corrected step-0 grating direction is camera-left (`180 deg`) and the fish's
whole-step net displacement is leftward, but the unweighted bout count reports
31 aligned bouts and 110 opposing bouts. The path-weighted step metric remains
slightly positive because the larger leftward movements offset many smaller
rightward/corrective bout windows.

Conclusion: `bout_fraction_correct_classified` and `bout_choice_index` should
be interpreted as bout-choice diagnostics, not as the primary freely swimming
OMR response. Palette should keep these fields, but plots and downstream
summaries should preferentially expose weighted bout metrics and step-level
path/net displacement metrics when the question is "did the fish move with the
grating?"

Implemented weighted bout additions:

```text
bout_path_index =
    sum(per_bout.parallel_displacement_mm)
    / sum(per_bout.bout_path_length_mm)

bout_fraction_correct_weighted_by_path =
    sum(path_length_mm for aligned bouts)
    / sum(path_length_mm for aligned or opposing bouts)

bout_fraction_correct_weighted_by_displacement =
    sum(bout_displacement_mm for aligned bouts)
    / sum(bout_displacement_mm for aligned or opposing bouts)
```

These should be computed from physical position/path arrays within detector
bout boundaries, not from the detector response signal.

`bout_path_index` includes all finite per-bout physical path, including
near-zero ambiguous bouts. Weighted correct fractions only include
aligned/opposing classifiable bouts in the denominator because they are
fraction-correct style metrics.

### 4. Per-Bout OMR Score

Formula:

```text
per_bout_omr_score = dot(bout_displacement_mm, s_i) / ||bout_displacement_mm||
```

Range: `[-1, +1]`

Chance level: `0`

Required fields:

- `bout_id`
- `fish_id`
- `start_frame`, `end_frame`, and any core/interpolated boundaries available
- `bout_displacement_mm`
- `bout_path_length_mm`
- `parallel_displacement_mm`
- `per_bout_omr_score`
- `correct_label` encoded as `1`, `0`, `-1`, with `0` meaning ambiguous
- `projection_deadzone`

The displacement should be measured from the physical estimator signal within
the bout boundary. Do not integrate the detector response.

### 5. Time-Weighted OMR Fraction

Define moving frames from a physical speed trace:

```text
moving_t = physical_speed_mm_s[t] >= moving_threshold_mm_s
correct_t = moving_t and p_t > projection_speed_deadzone_mm_s * dt
opposing_t = moving_t and p_t < -projection_speed_deadzone_mm_s * dt
```

Metrics:

```text
time_fraction_correct_classified =
    sum(dt for correct_t) / sum(dt for correct_t or opposing_t)

time_choice_index =
    (sum(dt for correct_t) - sum(dt for opposing_t))
    / sum(dt for correct_t or opposing_t)
```

Use this as a complementary metric. It can be sensitive to threshold choice and
should always record `moving_threshold_mm_s` and the physical speed source.

### 6. Windowed OMR Metrics

Compute the same OMR metrics over windows:

- non-overlapping windows,
- sliding windows with configurable stride,
- full step windows,
- optional full eligible-recording windows.

Recommended default windows:

- `10 s`
- `30 s`
- `60 s`
- full step

Windowed values are essential for bimodality checks because a fish may switch
between responsive and non-responsive epochs within one recording.

Palette also stores onset-anchored early-response windows separately from the
regular non-overlapping window table:

```text
early_windows/
  window_length_s
  actual_window_length_s
  omr_path_index
  omr_net_direction_index
  parallel_displacement_mm
  path_length_mm
  bout_path_index
  bout_fraction_correct_weighted_by_path
  bout_fraction_correct_weighted_by_displacement
  time_choice_index
  n_aligned_bouts
  n_opposing_bouts
  n_ambiguous_bouts
```

The default early windows are `5 s` and `10 s`, both measured from grating
step onset. These capture the "first response strength" that can be obscured
when a long epoch includes later corrective or arena-edge behavior.

### 7. Arena-Axis Occupancy and Opportunity Metrics

Long grating epochs can make displacement-only OMR indices hard to interpret.
If a fish starts near, or quickly reaches, the stimulus-forward side of the
arena, it may have little remaining space to keep accumulating forward
displacement. Palette therefore also projects fish position onto the stimulus
axis:

```text
position_axis_mm(t) = dot(position_mm(t) - arena_center_mm, s_hat)
position_axis_norm(t) = position_axis_mm(t) / arena_axis_extent_mm
```

Interpretation:

- `position_axis_norm ~= +1`: fish is near the stimulus-forward side.
- `position_axis_norm ~= 0`: fish is near the arena center along the stimulus
  axis.
- `position_axis_norm ~= -1`: fish is near the opposite side.

Step-level outputs include:

- `start_position_axis_mm`, `end_position_axis_mm`, `mean_position_axis_mm`
- `start_position_axis_norm`, `end_position_axis_norm`,
  `mean_position_axis_norm`
- `fraction_time_correct_side`
- `available_forward_space_at_start_mm`,
  `available_backward_space_at_start_mm`
- `available_forward_space_at_start_norm`,
  `available_backward_space_at_start_norm`
- `opportunity_normalized_parallel_displacement`

`opportunity_normalized_parallel_displacement` divides the step displacement
along the stimulus axis by the available arena space in the direction actually
traveled. It is a complement to, not a replacement for, `omr_path_index`.

Normalized fields require arena geometry. If `arena_center_mm` or
`arena_axis_extent_mm` cannot be resolved from calibration metadata, Palette
stores `NaN` in numeric arrays rather than inventing a radius. Zarr attrs and
provenance use JSON `null` for missing optional numeric metadata, never
JSON-invalid `NaN`.

### 8. First Classified Bout Latencies

For each moving-grating step, Palette stores the first bout whose physical
displacement is classifiable relative to the stimulus direction:

```text
first_aligned_bout_latency_s =
    (first_aligned_bout_start_frame - step_start_frame) / fps
```

Bout labels use the same per-bout OMR score and `projection_deadzone` as the
bout-fraction metrics:

- aligned/correct if `per_bout_omr_score > projection_deadzone`
- opposing if `per_bout_omr_score < -projection_deadzone`
- ambiguous otherwise

Stored fields:

- `first_aligned_bout_id`, `first_aligned_bout_start_frame`,
  `first_aligned_bout_latency_s`, `first_aligned_bout_score`
- `first_opposing_bout_id`, `first_opposing_bout_start_frame`,
  `first_opposing_bout_latency_s`, `first_opposing_bout_score`
- `first_classified_bout_id`, `first_classified_bout_start_frame`,
  `first_classified_bout_latency_s`, `first_classified_bout_score`

Missing events are stored as `-1` for IDs/frames and `NaN` for latency/score
arrays. These latency fields are useful for long trials where cumulative
displacement can saturate against arena boundaries.

## Storage Layout

### Step-Level OMR

For each `MOVING_GRATING` step:

```text
analysis/stimulus_response_runs/<run>/
  steps/step_<i>/
    grating/
      omr/
        attrs:
          method_version
          stimulus_direction_source
          detector_estimator_policy
          projection_deadzone
          moving_threshold_mm_s
          position_anchor
          arena_center_mm
          arena_axis_extent_mm
          arena_geometry_source
        per_fish/
          fish_id
          omr_path_index
          omr_net_direction_index
          parallel_displacement_mm
          net_displacement_mm
          path_length_mm
          valid_transition_count
          coverage_fraction
          bout_fraction_correct_classified
          bout_fraction_correct_all
          bout_choice_index
          bout_path_index
          bout_fraction_correct_weighted_by_path
          bout_fraction_correct_weighted_by_displacement
          bout_parallel_displacement_sum_mm
          bout_path_length_sum_mm
          bout_displacement_sum_mm
          time_fraction_correct_classified
          time_choice_index
          start_position_axis_norm
          end_position_axis_norm
          mean_position_axis_norm
          fraction_time_correct_side
          available_forward_space_at_start_norm
          available_backward_space_at_start_norm
          opportunity_normalized_parallel_displacement
          first_aligned_bout_latency_s
          first_opposing_bout_latency_s
          first_classified_bout_latency_s
          quality_flag
        per_bout/
          fish_id
          bout_id
          start_frame
          end_frame
          per_bout_omr_score
          parallel_displacement_mm
          bout_displacement_mm
          bout_path_length_mm
          correct_label
          quality_flag
        windows/
          window_id
          fish_id
          start_frame
          end_frame
          start_time_s
          end_time_s
          window_length_s
          omr_path_index
          bout_fraction_correct_classified
          time_choice_index
          mean_position_axis_norm
          fraction_time_correct_side
          coverage_fraction
          n_bouts
          quality_flag
        early_windows/
          window_id
          fish_id
          start_frame
          end_frame
          window_length_s
          actual_window_length_s
          omr_path_index
          bout_path_index
          bout_fraction_correct_weighted_by_path
          time_choice_index
          n_aligned_bouts
          n_opposing_bouts
          n_ambiguous_bouts
          quality_flag
```

The existing `grating/per_frame`, `grating/per_fish`, and `grating/per_bout`
groups remain valid. The `grating/omr` subgroup is where responsiveness-index
variants live.

### Run-Level OMR Summary

For cross-step summaries inside the same run:

```text
analysis/stimulus_response_runs/<run>/
  global/
    omr/
      per_fish/
        fish_id
        eligible_step_count
        eligible_window_count
        omr_path_index_mean
        omr_path_index_weighted_by_path
        bout_fraction_correct_classified
        bout_choice_index
        time_choice_index
        mean_fraction_time_correct_side
        mean_start_position_axis_norm
        mean_end_position_axis_norm
        mean_mean_position_axis_norm
        first_aligned_bout_latency_s_min
        total_path_length_mm
        total_parallel_displacement_mm
        total_bouts
        coverage_fraction
        quality_flag
```

This keeps OMR under the broad `stimulus_response_runs` category while still
making cross-recording export straightforward.

### Run-Level OMR Visualization Artifacts

Each OMR-capable `stimulus_response_runs/<run>` can persist a compact review
snapshot and interactive spec under:

```text
analysis/stimulus_response_runs/<run>/visualizations/
  stimulus_response_omr_summary_png
  stimulus_response_omr_summary_interactive/
  stimulus_response_omr_bout_trajectory_png
  stimulus_response_omr_bout_trajectory_interactive/
```

Generate these either during analysis:

```bash
scripts/py -m fisheye.analysis.stimulus_response <analysis.zarr> \
  ... \
  --write-zarr-artifacts
```

or after the fact:

```bash
scripts/py -m fisheye.analysis.plot_stimulus_response_omr <analysis.zarr> \
  --run <stimulus_response_run>
```

The PNG is a review artifact, not a data source. It summarizes:

- signed direction indices by moving-grating step,
- arena-axis start/mean/end occupancy and correct-side fraction,
- first classified/aligned/opposing bout latency,
- the smallest available windowed `omr_path_index` trace per step.

The interactive spec stores source paths for each `grating/omr` group so
Crimson, marimo, or export tooling can render richer views directly from the
canonical arrays. Missing numeric metrics may remain `NaN` in arrays, but all
artifact attrs and specs must serialize as strict JSON.

The bout-trajectory artifact is the first spatial bout review view. It reads
`positions_mm` and heading arrays from the source `track_kinematics` run, then
colors each `grating/omr/per_bout` segment by OMR label:

- aligned/correct bouts: green,
- opposing bouts: orange/red,
- ambiguous bouts: gray.

It also draws stimulus direction and the arena outline when calibration-backed
arena geometry is available. The initial artifact deliberately omits a
yaw/heading time trace; this keeps the review surface focused on whether bout
locations, trajectories, and OMR labels make sense spatially.

## Bimodality and Export Hooks

The implementation should make these rows easy to flatten later:

- one row per fish per recording from `global/omr/per_fish`,
- one row per fish per grating step from `steps/step_i/grating/omr/per_fish`,
- one row per fish per window from `steps/step_i/grating/omr/windows`,
- one row per bout from `steps/step_i/grating/omr/per_bout`.

Useful export columns:

- `recording_id`
- `zarr_path`
- `stimulus_response_run`
- `fish_id`
- `step_index`
- `stimulus_mode`
- `stimulus_direction_deg`
- `stimulus_speed_mm_s`
- `window_start_time_s`, if windowed
- `omr_path_index`
- `bout_fraction_correct_classified`
- `bout_choice_index`
- `time_choice_index`
- `coverage_fraction`
- `n_bouts`
- `dpf`, `strain`, `clutch`, `recording_date`, when available from metadata

Palette should not hardcode a responder/non-responder threshold. Downstream
analysis can fit mixture models, use empirical thresholds, or compare known
responsive/non-responsive controls.

## Validation Plan

Unit tests should cover synthetic motion:

| Scenario | Expected signed index | Expected fraction correct |
|----------|-----------------------|---------------------------|
| Fish moves exactly with stimulus | `+1` | `1.0` |
| Fish moves exactly opposite stimulus | `-1` | `0.0` |
| Fish moves perpendicular to stimulus | `0` | `0.5` or ambiguous, depending on deadzone policy |
| Fish does not move | `NaN` plus `no_movement`, no crash | `NaN` |
| Tracking gap crosses a window | Exclude invalid transition | Coverage drops |
| Stimulus direction reverses between steps | Step-local signs flip correctly | Step-local correctness follows direction |

Real-data validation should include:

- a moving-grating recording visually judged as responsive,
- a recording or epoch visually judged as weak/non-responsive,
- a calibration sanity check that grating direction is not flipped by 180 deg,
- a comparison of path-normalized OMR against existing
  `grating/per_fish/speed_weighted_alignment` and `drift_along_grating_mm`.

## Open Questions

1. Does `orientation_degrees` always mean drift direction, not bar orientation,
   in the Citrus moving-grating protocol?
2. Is the projector-to-camera angular transform fully determined by the stored
   homography and pixels-per-mm metadata, or do we still need per-rig offsets?
3. Which physical position anchor should OMR v1 use when both track kinematics
   positions and subject-shape body-frame origins are available?
4. Should low-coverage OMR values be computed with a warning flag, or replaced
   by `NaN` under a strict quality policy?
5. What projection deadzone should define an ambiguous bout: fixed cosine
   threshold, fixed angular threshold, or data-driven threshold?
6. Should windowed metrics be non-overlapping by default, sliding by default,
   or should both be materialized?
7. Should baseline/gray-screen epochs get OMR groups with `stimulus_active =
   false`, or should OMR groups only exist for active moving-grating steps?

## Implementation Direction

The moving-grating OMR implementation now lives in
`src/fisheye/analysis/stimulus_response_omr.py` and is dispatched by
`src/fisheye/analysis/stimulus_response.py`. It:

- computes OMR metrics for static moving-grating steps,
- consumes the same dense track representation as existing grating metrics,
- consumes optional `swim_bout_runs` for bout boundaries,
- writes `steps/step_<i>/grating/omr/`,
- writes `global/omr/per_fish/` summaries,
- records detector-vs-estimator provenance, direction mapping provenance,
  window lengths, and strict-JSON-safe optional metadata in local attrs,
- adds unit tests in `tests/unit/fisheye/test_stimulus_response.py`.

Do not add cross-recording Parquet export in the first implementation. The Zarr
schema should make that export easy later.
