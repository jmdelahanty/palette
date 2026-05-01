# Palette Analytics Math Primer
<!-- primer-meta
status: draft
last_verified: 2026-04-28
audience: new users
-->

Purpose: give new Palette users a compact explanation of the coordinate
systems, common formulas, and behavioral analytics used by the current analysis
stack. This is a conceptual primer, not the schema authority. For exact storage
contracts, follow the links at the end.

## Mental Model

Palette separates four layers:

| Layer | What it means | Example outputs |
| --- | --- | --- |
| Observations | What was measured from a frame or ROI | detections, keypoints, masks, ellipse fits |
| Coordinate frames | How measurements are interpreted geometrically | image pixels, ROI pixels, fish body frame |
| Derived traces | Time-series signals computed from observations | speed, heading, eye angles, vergence |
| Event summaries | Biological windows or events computed from traces | swim bouts, per-bout kinematics, stimulus responses |

The key rule is: **do not confuse the measured source with the interpreted
analysis**. A refined mask or keypoint is an authority about pixels or
landmarks. Eye angle, body frame, speed, and bout metrics are derived analysis
products that should be reproducible from their recorded sources.

## Coordinate Systems

### Image Coordinates

Most stored pixel coordinates use image-style axes:

```text
+x points right
+y points down
origin is the top-left corner of the image or ROI
```

This convention is natural for images, but it is not the same as the usual math
plane where positive `y` points upward.

### ROI Coordinates vs Image Coordinates

Palette uses both full-image and ROI-local coordinates.

| Coordinate space | Meaning | Common use |
| --- | --- | --- |
| Full image pixels | Coordinates in the original image or downsampled image | detections, crop origins, arena positions |
| ROI-local pixels | Coordinates inside a crop | keypoint model outputs, mask-local geometry |
| Physical units | Pixels converted with calibration | speed and distance in mm or mm/s |

A coordinate must be interpreted with its declared coordinate space. The same
number can mean different locations if one array is ROI-local and another is
full-image.

### Math Angles From Image Vectors

When Palette turns an image vector into an angle, it flips the image `y`
component:

```text
angle_deg = atan2(-dy, dx)
```

Implications:

| Angle | Meaning |
| --- | --- |
| `0 deg` | points toward image `+x`, to the right |
| `90 deg` | points upward in the image |
| `-90 deg` | points downward in the image |
| positive rotation | counter-clockwise in math coordinates |

This is why angle code often looks slightly different from image drawing code.

## Fish Body Frame

The fish body frame is Palette's fish-relative coordinate system. It lets us
say "forward", "left", "right", "inward", and "outward" without rotating the
image itself.

Current default semantic anchors:

| Anchor | Current definition |
| --- | --- |
| Forward polarity | `swim_bladder -> midpoint(eye_left, eye_right)` |
| Left/right polarity | labeled `eye_left` and `eye_right` |
| Forward axis | unit vector from posterior/body anchor toward head/anterior |
| Left axis | perpendicular unit vector pointing anatomical left |

For a vector `v_xy` measured in the same coordinate space as the axes:

```text
forward_coordinate = dot(v_xy, forward_axis_xy)
left_coordinate    = dot(v_xy, left_axis_xy)
heading_deg        = atan2(-forward_axis_y, forward_axis_x)
```

Interpretation:

| Quantity | Interpretation |
| --- | --- |
| positive forward coordinate | ahead of the local origin |
| negative forward coordinate | behind the local origin |
| positive left coordinate | anatomical left |
| negative left coordinate | anatomical right |

The body frame can come from keypoints, masks, centerlines, or future splines.
The downstream math should consume the declared body-frame arrays and metadata,
not assume one estimator forever.

## Tracking And Movement

`analysis/track_kinematics_runs` is the generic movement layer. It should answer
"where did this tracked animal move?" without needing to know the details of
eye masks, body masks, or stimulus protocols.

### Sparse Tracks

Tracks are often sparse. A fish may only have valid observations on some
frames. Palette therefore keeps frame indices and validity information instead
of assuming every array row is the same as every video frame.

Important consequence: missing frames are not the same as zero movement.

### Speed

For consecutive valid frames:

```text
frame_path_distance = distance(position_t, position_t_minus_1)
speed = frame_path_distance / delta_time
```

If frames are not consecutive, current track-kinematics semantics avoid
inventing movement across the gap. Downstream consumers should reuse
`frame_path_distance_*` and `cumulative_path_distance_*` rather than recomputing
distance by differencing only the valid samples.

### Path Length vs Net Displacement

These are different biological measurements.

| Metric | Formula idea | Interpretation |
| --- | --- | --- |
| Path length | sum of stepwise distances | how much ground the fish covered |
| Net displacement | distance from start position to end position | how far the fish ended from where it started |
| Speed | path increment divided by time | instantaneous or filtered movement rate |

A fish can swim in a loop with high path length and low net displacement.

### Raw, Filtered, Smoothed, Averaged

Current track-kinematics runs may expose multiple speed traces:

| Trace | Intended meaning |
| --- | --- |
| `speed_raw_*` | direct frame-to-frame speed before jitter suppression |
| `speed_filtered_*` | speed after hysteresis/micro-motion filtering |
| `speed_smoothed_*` | temporally smoothed speed |
| `speed_averaged_*` | window-averaged speed |

For onset-sensitive bout detection, `speed_filtered` or causal smoothing is
usually safer than centered smoothing because centered smoothing can leak future
motion backward in time.

Acceleration should always be interpreted relative to its source speed trace.
Current track-kinematics runs therefore expose
`speed_derivatives/<speed_level>/acceleration_*` and
`speed_derivatives/<speed_level>/smoothed_acceleration_*`. For example,
`speed_derivatives/speed_filtered/acceleration_mm` is the derivative of
`speed_filtered_mm`, while `speed_derivatives/speed_smoothed/acceleration_mm`
is the derivative of `speed_smoothed_mm`. The older flat
`acceleration_*` arrays are compatibility aliases for the `speed_smoothed`
derivative and should not be treated as source-agnostic acceleration.

For a future track-kinematics schema bump, the cleaner target is to group each
speed trace with its derived quantities under
`movement/speed/<raw|filtered|smoothed|averaged>/`. That would make the source
trace, path-distance increment, and acceleration products co-located instead of
split across flat speed arrays and a derivative sibling group. New consumers
should be designed so they can prefer this v2 layout when it appears.

## Swim Bouts

`analysis/swim_bout_runs` stores bout segmentation candidates. These candidates
say "these time intervals are candidate swim bouts for this speed trace and
parameter set."

`analysis/bout_kinematics_runs` stores measurements computed for those bouts.
Changing segmentation should make a new swim-bout candidate. Changing
measurements should make a new bout-kinematics candidate.

### Threshold Bouts

A simple bout detector asks whether a selected speed signal is above a
threshold:

```text
in_bout = speed_signal > threshold
```

Then it applies duration, gap, and boundary rules. Because video is sampled at
frames, the primary boundaries are frame-discrete. Some runs also store
interpolated boundary annotations for better visualization of threshold
crossings between frames.

### Peak-Event Bouts

Peak-event detection is a separate way to segment bouts. It asks whether the
signal contains a sufficiently prominent local peak, then defines a bout window
around that peak. This is useful when small valleys inside one biological bout
would otherwise split the bout too aggressively.

### Exponential Response Traces

Some bout candidates use a causal exponential response to a source speed trace:

```text
k(t) = exp(-t / tau), t >= 0
```

This is a detector response trace, not a replacement for measured fish speed.
Longer `tau` values broaden the response and extend the decay tail. Shorter
`tau` values track the source speed more closely.

### Detector vs Estimator

Bout segmentation has two distinct jobs:

- the detector finds event windows from a selected speed-like signal
- the estimator measures physical quantities from the least-distorting movement
  source available, usually `speed_filtered` and its matching
  `frame_path_distance_filtered_*` arrays

The causal exponential response is a detector signal. Its peak height, onset,
and decay shape depend on `tau`, so it should not be interpreted as measured
fish speed. When an exponential candidate is useful for finding bouts, physical
fields such as path length, net displacement, mean speed, and physical peak
speed should still be measured from the declared movement source inside the
detected window.

Duration has the same detector-vs-estimator split. `duration_s` in
`swim_bout_runs` is the duration of the stored detector boundary. If the
detector signal has a long tail, that duration can be a detector-envelope
duration rather than physical active-motion duration.

Schema v7 `bout_kinematics_runs` writes this stricter physical duration under
`movement/per_bout_metrics/`. It preserves detector-window duration in
`detector_*` fields and measures physical active duration by slicing near the
detector-defined bout and finding first/last above-threshold samples on a
declared physical speed source, usually `speed_filtered`.

## Eye Angles

`analysis/eye_angle_runs` stores eye geometry interpreted in the fish body
frame. Current v5 run attrs retain the gaze family as the historical preferred
biological viewing surface:

```text
preferred_angle_family = "gaze"
preferred_eye_axis     = "ellipse_major"
gaze_angle_source      = "ellipse_minor_derived_from_resolved_major_axis"
```

This means the canonical eye-orientation axis is the resolved ellipse major
axis. The apparent look/gaze axis is the perpendicular minor-axis direction
derived from that resolved major axis, rather than independently disambiguated.
Output schema v7 also includes `eye_angle_variant_schema`, which lets UI
consumers choose between eye-frame, gaze, nasal-gaze, major-axis, centroid, and
legacy representations without hardcoding field groups. For UI angle-trace
selectors, use `eye_angle_variant_schema.default_representation` rather than
inferring a default from `preferred_angle_family`.

Canonical orientation fields:

```text
left_major_signed_deg
right_major_signed_deg
```

Here `0 deg` means the major axis is aligned with the fish's body-forward axis,
and positive values rotate toward anatomical left.

### Bianco/Engert Eye-Frame Angles

Fields:

```text
left_eye_angle_deg
right_eye_angle_deg
vergence_eye_angle_deg
```

These are derived from the canonical major-axis fields but use per-eye signs:

```text
left_eye_angle_deg     = -left_major_signed_deg
right_eye_angle_deg    =  right_major_signed_deg
vergence_eye_angle_deg = left_eye_angle_deg + right_eye_angle_deg
```

Interpretation:

| Value | Meaning |
| --- | --- |
| positive per-eye angle | that eye is rotated nasally/inward |
| negative per-eye angle | that eye is rotated temporally/outward |
| positive `vergence_eye_angle_deg` | the eyes are converged |
| negative `vergence_eye_angle_deg` | the eyes are diverged |
| same-sign body-frame rotation of both major axes | yoked rotation, not convergence |

These fields are the easiest match for Bianco/Engert-style larval zebrafish
eye-angle plots because both eyes become positive when they converge. They are
not a replacement for the body-frame major-axis fields; they are a biological
presentation of the same canonical orientation measurements.

### Per-Eye Gaze Signed Angles

Fields:

```text
left_gaze_signed_deg
right_gaze_signed_deg
```

Meaning:

| Value | Interpretation |
| --- | --- |
| `0 deg` | eye gaze axis aligned with fish-forward |
| positive | rotated toward anatomical left |
| negative | rotated toward anatomical right |

These values are useful for QA and geometric interpretation. They are body-frame
axis angles. They are not by themselves proof that an eye is fixating a target.

### Nasal Gaze

Fields:

```text
left_nasal_gaze_deg
right_nasal_gaze_deg
```

Current v5 definition:

```text
nasal_gaze = 90 - abs(gaze_signed)
```

Interpretation:

| Value | Meaning |
| --- | --- |
| larger | eye is more inward/nasal, more converged |
| smaller | eye is more outward/lateral |

This is usually easier to interpret biologically than raw signed gaze angles.

### Mean Eye Vergence

Field:

```text
mean_eye_vergence_gaze_deg
```

Definition:

```text
mean_eye_vergence_gaze_deg =
    0.5 * (left_nasal_gaze_deg + right_nasal_gaze_deg)
```

This is the preferred field for Johnson/BEAST-style "eye vergence over time"
plots. Higher values indicate stronger mean per-eye convergence.

### Total Gaze Axis Separation

Field:

```text
vergence_gaze_deg
```

This is the nonnegative separation between the two gaze axes. It is retained for
compatibility with earlier Palette outputs. It is often roughly twice the mean
per-eye vergence, depending on eye polarity and validity.

For new biological plots, prefer:

```text
mean_eye_vergence_gaze_deg_smoothed
left_nasal_gaze_deg_smoothed
right_nasal_gaze_deg_smoothed
```

Use `left_gaze_signed_deg` and `right_gaze_signed_deg` when you specifically
need signed orientation relative to the fish body axis.

## Heading, Turning, And Angular Metrics

Heading is a scalar representation of the fish forward axis:

```text
heading_deg = atan2(-forward_axis_y, forward_axis_x)
```

Angular differences must respect wraparound. For example, `179 deg` and
`-179 deg` are only `2 deg` apart, not `358 deg` apart.

Common derived quantities:

| Metric | Meaning |
| --- | --- |
| heading delta | frame-to-frame change in heading |
| angular velocity | heading change per second |
| net heading change per bout | heading at bout end minus heading before bout |
| within-bout heading range | largest heading excursion inside a bout |
| within-bout heading path | accumulated absolute heading movement inside a bout |

Small noisy heading deltas can become large angular velocity spikes, so always
check validity, smoothing, and frame gaps before interpreting angular metrics.

## Validity And Gaps

Palette analysis arrays should not require readers to infer all invalid state
from zeros or NaNs.

General rules:

| Signal | Preferred invalid representation |
| --- | --- |
| floating values | `NaN` plus explicit validity or reason fields |
| dense frame arrays | validity arrays distinguish missing from zero |
| bout summaries | valid fractions and `gap_censored` style flags |
| masks/geometry | component status, review status, and reason tags |

Zeros can be real values. For example, a fish can truly have zero speed. A zero
in a dense array should only mean "missing" if the paired validity field says it
is missing.

## What To Plot First

For exploratory review, start with these fields:

| Goal | First fields to plot |
| --- | --- |
| Basic movement | `speed_filtered_mm`, `speed_smoothed_mm` |
| Bout review | selected swim-bout intervals over the exact speed level used for segmentation |
| Movement amount | `frame_path_distance_*`, `cumulative_path_distance_*` |
| Bout distance | path length and net displacement side by side |
| Eye convergence | `mean_eye_vergence_gaze_deg_smoothed` |
| Per-eye convergence | `left_nasal_gaze_deg_smoothed`, `right_nasal_gaze_deg_smoothed` |
| Eye geometry QA | `left_gaze_signed_deg`, `right_gaze_signed_deg`, ellipse ratio/validity |
| Turning | heading, angular velocity, within-bout heading range/path |

When a plot looks surprising, inspect the raw and smoothed versions together.
The goal is to determine whether the behavior is in the source measurement or
was introduced by filtering, smoothing, segmentation, or a coordinate
interpretation.

## Common Pitfalls

| Pitfall | Safer interpretation |
| --- | --- |
| Treating image `y` as math `y` | Flip `y` when computing angles with `atan2` |
| Treating ROI coordinates as image coordinates | Check coordinate-space metadata and crop offsets |
| Treating missing frames as zero speed | Use validity arrays and gap-aware path-distance fields |
| Treating centered smoothing as causal | Use filtered or causal traces for onset timing |
| Treating `vergence_gaze_deg` as the Johnson mean | Use `mean_eye_vergence_gaze_deg` for mean per-eye vergence |
| Treating eye-axis angle as target fixation | Eye angle gives orientation; target tracking needs target coordinates too |
| Comparing signed angles naively near wraparound | Use circular/angular difference helpers |
| Reading stale derived runs as current truth | Check source lineage, latest attrs, and stale metadata |

## Where The Data Lives

| Analysis family | Zarr location | Concept |
| --- | --- | --- |
| Track kinematics | `analysis/track_kinematics_runs/<scope>/<run>` | per-track movement traces |
| Swim bouts | `analysis/swim_bout_runs/<run>` | bout segmentation candidates |
| Bout kinematics | `analysis/bout_kinematics_runs/<run>` | per-bout movement, heading, and optional eye summaries |
| Eye angles | `analysis/eye_angle_runs/<run>` | per-frame or per-detection eye angle traces |
| Subject shape | `analysis/subject_shape_runs/<run>` | mask-derived shape, body frame, eye geometry |
| Stimulus response | `analysis/stimulus_response_runs/<run>` | protocol-aware downstream summaries |

## Related Technical Docs

- [current_pipeline_contract.md](current_pipeline_contract.md)
- [body_frame_contract.md](body_frame_contract.md)
- [derived_analysis_run_contract.md](derived_analysis_run_contract.md)
- [track_kinematics_bout_status.md](track_kinematics_bout_status.md)
- [bout_kinematics_run_design.md](bout_kinematics_run_design.md)
- [swim_bout_peak_event_detector_design.md](swim_bout_peak_event_detector_design.md)
- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [tail_kinematics_tool_interop_design.md](tail_kinematics_tool_interop_design.md)
- [raw_vs_smoothed_metrics_behavioral_geometry.md](raw_vs_smoothed_metrics_behavioral_geometry.md)
- [eye_angle_variants.md](eye_angle_variants.md)
- [src/fisheye/docs/eye_angle_conventions.md](../src/fisheye/docs/eye_angle_conventions.md)
