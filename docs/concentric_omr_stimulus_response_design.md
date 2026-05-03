# Concentric OMR Metrics in Stimulus Response Runs

Status: design only. No implementation yet.

## Scope

Concentric-grating responses should remain inside
`analysis/stimulus_response_runs/<run>`. They are a metric family for
`CONCENTRIC_GRATING` steps, analogous to translational OMR under
`steps/step_<i>/grating/omr/`, but the geometry is different enough that the
metrics should not reuse translational OMR names without qualification. Escape
or looming-response detection is explicitly out of scope for this metric
family; use the dedicated `LOOMING_DOT` pathway for that behavior class.

The existing `concentric_grating/` implementation is a v0 centering analysis:
it computes distance to a center, heading relative to the center, and radial
and tangential speed components. The next implementation should add a
first-class radial OMR group under:

```text
analysis/stimulus_response_runs/<run>/
  steps/step_<i>/
    concentric_grating/
      per_frame/       # existing centering/polar decomposition
      per_fish/        # existing centering summaries
      time_series/     # existing binned centering summaries
      omr/             # proposed radial OMR metrics
        per_bout/
        per_fish/
        windows/
        early_windows/
```

## Dual Use: Probe vs Centering Utility

Concentric gratings have two valid roles in Palette recordings:

- **Primary radial-flow probe**: the concentric grating is the experimental
  stimulus. In this mode, graded per-bout, per-fish, windowed, and early-window
  radial OMR metrics are the primary outputs.
- **Centering utility**: the concentric grating is used to move or keep the fish
  near a target region before another protocol component. In this mode, the
  same per-bout and radial decomposition metrics are still useful, but the most
  important output is whether the epoch achieved its positioning goal.

The schema should not fork the underlying metric definitions. A
`radial_omr_score` still means "path-normalized movement aligned with the local
radial stimulus direction" in both cases. The interpretation and summary layer
change by protocol intent:

- primary-stimulus analyses emphasize responsiveness distributions, windows,
  and bout-level alignment;
- centering-utility analyses emphasize target-annulus occupancy, final radius,
  and time-to-target.

Record the intended use when available:

```text
concentric_grating_role = "primary_stimulus" | "centering_utility" | "unknown"
```

Current Citrus recordings do not encode this role. Palette should default
existing recordings to `unknown` and should not infer intent from step names.
Future Citrus protocols should add role metadata as a per-step field.

## Citrus Metadata Requirements

Current Citrus protocol snapshots provide most geometry needed for radial OMR,
but not role/target-annulus intent.

### Existing Recoverable Fields

Citrus stores protocol JSON in H5 at:

```text
/protocol_snapshot/protocol_definition_json
```

For `CONCENTRIC_GRATING` steps, Palette can recover:

| Palette concept | Citrus source | Notes |
|-----------------|---------------|-------|
| stimulus mode | `steps[].stimulus_mode_str == "CONCENTRIC_GRATING"` | Use protocol snapshot, not event `details_json`. |
| authored polarity | `steps[].parameters.is_expanding` | `true` = intended expanding, `false` = intended contracting. |
| pixel spatial frequency | `steps[].parameters.spatial_freq_rpp` | radial cycles per projector/canvas pixel; actual renderer input. |
| pixel speed | `steps[].parameters.speed_pps` | projector/canvas px/s; actual renderer input. |
| physical spatial frequency | `steps[].parameters.spatial_freq_cycles_per_mm` | portable copy; validate against pixel units and projector scale. |
| physical speed | `steps[].parameters.speed_mm_per_sec` | portable copy; validate against pixel units and projector scale. |
| duty cycle | `steps[].parameters.duty_cycle` | ring duty fraction. |
| foreground/background colors | `line_color_imgui_*`, `bg_color_imgui_*` | contrast is not explicit unless Palette defines a luminance model. |

Temporal frequency is derived:

```text
temporal_frequency_hz =
    abs(speed_mm_per_sec * spatial_freq_cycles_per_mm)

actual_rendered_temporal_frequency_hz =
    abs(speed_pps * spatial_freq_rpp)
```

No explicit phase history is serialized. Do not attempt phase-resolved
analysis from existing H5 logs.

### Authored vs Rendered Polarity Caveat

Palette should preserve two concepts:

```text
stimulus_radial_polarity_authored =
    "expanding" if is_expanding else "contracting"

stimulus_radial_sign_authored =
    +1 if is_expanding else -1
```

The Citrus agent reported a shader-sign caveat: the renderer computes a wave
from `radius * frequency + phase`, and `phase` increases when `is_expanding`
is true. Mathematically, constant phase contours may therefore move toward
smaller radius. Existing metadata unambiguously records authored intent, but
actual rendered motion polarity should be validated before Palette treats
authored polarity as observed polarity.

Until validation is complete, write:

```text
stimulus_radial_polarity_source = "citrus_authored_is_expanding"
stimulus_radial_polarity_validated = false
```

After validation, add:

```text
stimulus_radial_polarity_observed = "expanding" | "contracting"
stimulus_radial_sign_observed = +1 | -1
stimulus_radial_polarity_validated = true
```

Radial OMR metrics should use the validated observed polarity when available.
If only authored polarity is available, the run must record that limitation in
attrs.

### Stimulus Center

Current Citrus concentric gratings do not have explicit per-step center fields.
The renderer uses the active arena texture center:

```text
center_x = texture_width / 2
center_y = texture_height / 2
```

Existing H5 logs can recover the center from:

```text
/stimulus_coordinates/<arena>/custom_coordinates.attrs.texture_center_x
/stimulus_coordinates/<arena>/custom_coordinates.attrs.texture_center_y
/calibration_snapshot/arena_geometry.attrs.arena_region_width_px / 2
/calibration_snapshot/arena_geometry.attrs.arena_region_height_px / 2
/calibration_snapshot/arena_geometry.attrs.arena_region_center_in_canvas_x_px
/calibration_snapshot/arena_geometry.attrs.arena_region_center_in_canvas_y_px
```

Coordinate frame is `stimulus_canvas_px` / active arena texture px, top-left
origin, `+x` right, `+y` down. To map to camera pixels, invert the calibration
homography whose forward mapping is camera-view px to final display canvas px.
Relative millimetres from center can use:

```text
/calibration_snapshot/<camera_id>.attrs.pixels_per_mm_projector
```

Citrus does not currently export a full tank/world-mm transform, so Palette
should record center coordinate source and conversion method explicitly.

### Missing Fields To Add In Citrus

Future Citrus should add optional per-step fields:

```json
{
  "stimulus_role": "primary_stimulus",
  "center_coordinate_frame": "arena_relative_canvas_px",
  "center_x_px": -1,
  "center_y_px": -1,
  "center_source": "arena_texture_center",
  "target_radius_min_mm": 8.0,
  "target_radius_max_mm": 14.0,
  "target_radius_source": "protocol_manual",
  "centering_success_fraction_threshold": 0.8
}
```

Use `center_x_px = -1`, `center_y_px = -1`, and `center_source =
"arena_texture_center"` only as an explicit sentinel convention, not as an
implicit inference.

### Palette Zarr Landing Zone

Palette import should materialize these fields into
`analysis/stimulus_runs/<run>` rather than requiring every downstream consumer
to re-parse H5:

- per-step table/attrs: step index, mode, duration, role, polarity, center,
  frequency, speed, target annulus;
- events: copy event timing fields, but do not rely on `details_json` for full
  stimulus params;
- root/run attrs: copy full protocol snapshot JSON or protocol hash and path;
- calibration: copy arena geometry, homography, projector pixels/mm, and
  direction/coordinate mapping attrs;
- stimulus coordinates: copy texture width/height/origin/center.

Existing Palette Zarrs that did not copy protocol/calibration metadata need
metadata enrichment from the raw Citrus H5 or a clean re-import. No new
acquisition is required for polarity, frequency, and center when the raw H5
contains protocol and calibration snapshots.

## Literature Grounding

Concentric gratings are radial optic-flow stimuli. Unlike translational
gratings, the local stimulus direction depends on the fish's position relative
to the stimulus center. This makes metric conventions less standardized than
for translational OMR, and it creates additional confounds from starting radius
and arena boundaries.

| Source | Relevant point | Palette implication |
|--------|----------------|---------------------|
| Maaswinkel and Li 2003, Vision Research | Zebrafish OMR strength and direction depend on stimulus spatial and temporal frequency. | Store concentric grating frequency/speed parameters even if v1 metrics only use motion polarity. |
| Wang et al. 2019, BMC Biology | Zebrafish pretectum/tectum encode specific binocular optic-flow patterns relevant to OKR/OMR. | Treat radial/concentric responses as optic-flow responses, not as fixed-axis OMR. |
| Matsuda and Kubo 2021, Frontiers in Neural Circuits | Review of optic-flow circuitry identifies pretectum as a primary center for optic-flow processing driving OKR/OMR. | Use optic-flow terminology and keep radial-flow metrics separate from translational OMR. |

References:

- Maaswinkel and Li 2003: https://pubmed.ncbi.nlm.nih.gov/12505601/
- Wang et al. 2019: https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-019-0648-2
- Matsuda and Kubo 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC8334359/

## Coordinate System

Use camera/world millimetres, matching the physical position arrays consumed by
`stimulus_response`.

For fish position `p = (x, y)` and stimulus center `c = (cx, cy)`:

```text
r_vec = p - c
r = ||r_vec||
r_hat = r_vec / r                  # outward from center to fish
theta_hat = (-r_hat_y, r_hat_x)    # CCW tangent
```

The radial basis is undefined when `r < radial_singularity_epsilon_mm`. Those
frames/transitions should be marked invalid for radial/tangential metrics, not
forced to zero.

Stimulus polarity is represented as an authored intent and, after validation,
an observed/rendered polarity:

```text
stimulus_radial_sign_* = +1 for expanding / outward ring motion
stimulus_radial_sign_* = -1 for contracting / inward ring motion
```

Physical radial quantities should use outward-positive naming:

```text
radial_displacement_mm = r_end - r_start
radial_speed_outward_mm_s > 0 means moving away from center
radial_speed_outward_mm_s < 0 means moving toward center
```

Stimulus-aligned quantities should multiply by the best available stimulus
radial sign:

```text
effective_stimulus_radial_sign =
    stimulus_radial_sign_observed if validated else stimulus_radial_sign_authored

stimulus_aligned_radial_displacement_mm =
    effective_stimulus_radial_sign * radial_displacement_mm
```

This preserves physical interpretability while making "with the stimulus" a
positive response for both expanding and contracting steps. If the sign is only
authored and not observed/validated, downstream analyses should treat the
directional OMR scores as provisional.

## Detector vs Estimator Hygiene

The detector-vs-estimator rule from translational OMR applies unchanged.

- Bout boundaries come from `analysis/swim_bout_runs/<run>`.
- Detector response traces such as exponential-convolved speed are event
  detectors only.
- Radial displacement, tangential displacement, path length, speed, and position
  occupancy must be measured from physical estimator sources: `positions_mm`,
  gap-aware physical transition/path arrays, and physical speed traces.
- The output attrs must record `source_track_kinematics_run`,
  `source_bout_run`, `bout_detector_signal`, `movement_metric_source_level`,
  `position_source_array`, and `position_anchor`.

## Metric Definitions

### Per-Transition Physical Components

For valid adjacent frames `t-1 -> t`, compute `dx = p_t - p_{t-1}`. Evaluate
`r_hat` and `theta_hat` at the midpoint position when possible. This avoids
large endpoint-angle artifacts for curved paths.

```text
radial_step_mm = dx dot r_hat_mid
tangential_step_mm = dx dot theta_hat_mid
path_step_mm = ||dx||
stimulus_aligned_radial_step_mm =
    effective_stimulus_radial_sign * radial_step_mm
```

Skip transitions that cross tracking gaps or have invalid radial basis.

### Per-Bout Radial Displacement

Store both endpoint and transition-summed forms:

```text
radial_displacement_endpoint_mm = r_end - r_start
radial_displacement_integrated_mm = sum(radial_step_mm)
stimulus_aligned_radial_displacement_mm =
    effective_stimulus_radial_sign * radial_displacement_integrated_mm
```

The integrated form is the preferred path-aware estimator; the endpoint form is
useful for sanity checks and for interpreting whether the fish ended closer to
or farther from the center.

### Per-Bout Radial OMR Score

```text
radial_omr_score =
    stimulus_aligned_radial_displacement_mm / path_length_mm
```

Range is `[-1, +1]` when `path_length_mm > 0`.

- `+1`: movement fully aligned with local radial stimulus motion.
- `0`: no net radial component, often tangential/orbiting motion.
- `-1`: movement fully opposite local radial stimulus motion.

Also store:

```text
radial_net_direction_score =
    stimulus_aligned_radial_displacement_mm / net_displacement_mm
```

This is more sensitive to endpoint displacement and less robust for tortuous
paths. The path-normalized score should be the preferred default.

### Per-Bout Tangential Displacement

```text
tangential_displacement_mm = sum(tangential_step_mm)
tangential_bias_score = tangential_displacement_mm / path_length_mm
```

Counterclockwise is positive in camera/mm coordinates after applying the local
polar basis. Tangential bias is not OMR by itself; it distinguishes "not radial"
from "not moving".

### Per-Fish Whole-Step Radial OMR

For each fish and concentric-grating step:

```text
omr_path_index =
    sum(stimulus_aligned_radial_step_mm) / sum(path_step_mm)

omr_net_direction_index =
    effective_stimulus_radial_sign * (r_end - r_start) / net_displacement_mm

bout_fraction_correct_classified =
    n_bouts_with_radial_omr_score_gt_deadzone /
    n_bouts_with_abs_score_gt_deadzone

time_fraction_correct_classified =
    time_moving_with_stimulus_radial_speed_gt_deadzone /
    time_moving_classified
```

Chance levels:

- Signed/path indices: `0`.
- Fraction-correct metrics: `0.5`.
- Tangential bias: `0`.

Edge cases:

- No valid movement/path: store `NaN`, not zero.
- No classified bouts: store `NaN`.
- Fish at center singularity for most frames: set quality flag and store `NaN`
  for radial basis-dependent metrics.

### Windowed and Early-Window Radial OMR

Mirror the translational OMR window structure:

- non-overlapping windows: 10 s, 30 s, 60 s defaults;
- early windows from step onset: 5 s, 10 s defaults;
- all windows record actual frame bounds, coverage, and quality flags.

These are important because concentric responses can be strongly onset-biased,
especially if a fish starts close to a boundary or responds with a single large
early bout.

### Epoch-Level Centering Success

For centering-utility epochs, add a cheap, explicit success summary. This should
live beside radial OMR metrics rather than replacing them.

Target annulus parameters should come from stimulus/protocol metadata when
available, or from analysis parameters recorded in attrs:

```text
target_radius_min_mm
target_radius_max_mm
target_radius_source =
    "stimulus_params" | "analysis_param" | "not_configured"
```

For each fish:

```text
end_radius_in_target_annulus =
    target_radius_min_mm <= end_radius_mm <= target_radius_max_mm

mean_radius_in_target_annulus =
    target_radius_min_mm <= mean_radius_mm <= target_radius_max_mm

fraction_time_in_target_annulus =
    time_in_target_annulus / valid_time

centering_success =
    end_radius_in_target_annulus
    or fraction_time_in_target_annulus >= centering_success_fraction_threshold
```

If no target annulus is configured, write `target_radius_source =
"not_configured"` and store `centering_success` as false or absent with a
quality flag. Do not infer a target radius silently from observed fish behavior.

These fields are not a claim about OMR responsiveness. They are operational
QC/utility outputs answering "did this epoch position the fish where the next
protocol step expected it?"

## Position-Dependent Baseline

Radial metrics have a stronger geometry bias than translational OMR. A fish near
the center has more available outward space; a fish near the edge has less.

Default for v1:

- Report uncorrected radial OMR metrics.
- Store starting, ending, and mean radius for each step/window/bout.
- Store normalized radius when arena radius is known.
- Store available inward and outward room at the start.
- Add quality flags for near-center singularity and near-boundary occupancy.

Do not subtract a baseline by default in v1. Baseline correction should be an
analysis-time choice until we have enough recordings to evaluate null models.

Supported future options:

- restrict to an annulus, for example `0.25 <= radius_norm <= 0.75`;
- subtract gray-screen or baseline-step radial drift;
- compare within fish across expanding and contracting conditions.

## Escape Handling Is Out of Scope

Concentric OMR should not include escape-candidate classification. In Palette,
concentric gratings are intended for radial-flow responsiveness and centering
utility, not for evoking or classifying escape responses. If a future protocol
uses looming-like stimuli, the analysis should use a dedicated `LOOMING_DOT`
metric family with its own response criteria rather than overloading
`concentric_grating/omr/`.

## Proposed Output Schema

```text
steps/step_<i>/concentric_grating/omr/
  attrs:
    method_version
    coordinate_system = "camera_mm_polar_about_stimulus_center"
    stimulus_center_mm
    stimulus_center_source
    stimulus_center_coordinate_frame
    stimulus_radial_polarity_authored = "expanding" | "contracting"
    stimulus_radial_sign_authored = +1 | -1
    stimulus_radial_polarity_observed = "expanding" | "contracting" | null
    stimulus_radial_sign_observed = +1 | -1 | null
    stimulus_radial_polarity_source
    stimulus_radial_polarity_validated
    spatial_freq_rpp
    speed_pps
    spatial_freq_cycles_per_mm
    speed_mm_per_sec
    temporal_frequency_hz
    actual_rendered_temporal_frequency_hz
    radial_singularity_epsilon_mm
    projection_deadzone
    projection_speed_deadzone_mm_s
    baseline_correction = "none"
    concentric_grating_role = "primary_stimulus" | "centering_utility" | "unknown"
    target_radius_min_mm
    target_radius_max_mm
    target_radius_source
    centering_success_fraction_threshold
    detector_vs_estimator attrs...

  per_bout/
    fish_id
    bout_id
    start_frame
    end_frame
    start_radius_mm
    end_radius_mm
    mean_radius_mm
    start_radius_norm
    radial_displacement_endpoint_mm
    radial_displacement_integrated_mm
    stimulus_aligned_radial_displacement_mm
    tangential_displacement_mm
    path_length_mm
    net_displacement_mm
    radial_omr_score
    radial_net_direction_score
    tangential_bias_score
    omr_label                 # +1 aligned, -1 opposing, 0 ambiguous
    valid_radial_basis

  per_fish/
    fish_id
    omr_path_index
    omr_net_direction_index
    bout_fraction_correct_classified
    bout_choice_index
    time_fraction_correct_classified
    time_choice_index
    tangential_bias_index
    start_radius_mm
    end_radius_mm
    mean_radius_mm
    start_radius_norm
    end_radius_norm
    mean_radius_norm
    available_outward_space_at_start_mm
    available_inward_space_at_start_mm
    coverage_fraction
    end_radius_in_target_annulus
    mean_radius_in_target_annulus
    fraction_time_in_target_annulus
    time_to_target_annulus_s
    centering_success
    quality_flag

  windows/
    window_id
    fish_id
    start_frame
    end_frame
    window_length_s
    actual_window_length_s
    omr_path_index
    bout_fraction_correct_classified
    time_choice_index
    tangential_bias_index
    mean_radius_norm
    fraction_time_in_target_annulus
    centering_success
    coverage_fraction
    quality_flag

  early_windows/
    same structure as windows, fixed from step onset
```

## Visualization Plan

Add a radial OMR review plot analogous to the translational OMR bout trajectory
plot:

- draw arena outline and stimulus center;
- draw one or more reference rings/annuli;
- plot fish trajectory in mm;
- color bout segments by `omr_label`;
- draw local radial arrows at bout start points;
- annotate expanding/contracting polarity, starting radius, and target-annulus
  success when configured.

This plot should be stored under:

```text
analysis/stimulus_response_runs/<run>/visualizations/
  stimulus_response_concentric_omr_summary_png
  stimulus_response_concentric_omr_bout_trajectory_png
```

The PNG is review/QC only. The numeric arrays remain canonical.

## Implementation Plan

1. Extract current concentric helpers from `stimulus_response.py` into
   `stimulus_response_concentric.py`.
2. Add `stimulus_response_concentric_omr.py` for radial OMR metrics.
3. Keep `stimulus_response.py` as orchestration/writer/CLI only.
4. Implement synthetic tests for radial, tangential, inward, outward, center
   singularity, centering-success, and gap handling.
5. Add canary run writing only after synthetic tests pass.
6. Add review plots after numeric schema is stable.

## Validation Plan

Synthetic tests:

- fish moving exactly outward during expanding stimulus: `omr_path_index = +1`;
- fish moving exactly inward during expanding stimulus: `omr_path_index = -1`;
- fish moving exactly inward during contracting stimulus: `omr_path_index = +1`;
- fish moving exactly tangentially: radial index `0`, tangential bias non-zero;
- fish starts at center: radial metrics `NaN` with center-singularity quality;
- centering-utility epoch ending inside target annulus:
  `centering_success = true`;
- centering-utility epoch without configured target annulus:
  target-specific outputs are marked unavailable rather than inferred;
- tracking gap inside bout/window: transition across the gap is excluded;

Real-data checks:

- Compare expanding and contracting steps separately for primary-stimulus
  recordings.
- For centering-utility recordings, verify final radius, target-annulus
  occupancy, and time-to-target against visual inspection.
- Plot OMR indices against starting radius to detect geometry-driven artifacts.
- Inspect early-window metrics for large first-bout effects.

## Open Questions

1. Can Citrus/protocol metadata identify whether a concentric-grating step is a
   primary stimulus, a centering utility, or unknown? Current answer: not for
   old recordings; add optional per-step metadata for future protocols.
2. For centering-utility epochs, what target annulus should be considered
   successful, and should that come from protocol metadata or analysis params?
3. What should be the first default baseline strategy: uncorrected with radius
   covariates, annulus restriction, gray-screen subtraction, or within-fish
   expanding/contracting contrast?
4. Which position anchor should be preferred for radial OMR when both
   track-kinematics positions and subject-shape body origins are available?
5. Validate the Citrus shader polarity: does `is_expanding=true` render
   outward or inward motion? Until this is validated, Palette should label OMR
   signs as authored-intent based.
6. Do current Palette Zarr imports preserve enough Citrus protocol/calibration
   metadata for concentric OMR, or do existing Zarrs need H5-backed enrichment?
