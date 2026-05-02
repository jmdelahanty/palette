# Crimson Track Motion Read Contract

Date anchored: 2026-05-02

Purpose: define the Palette-side read contract Crimson should use for current
motion traces, swim-bout overlays, and per-bout metrics. This is the current
consumer-facing replacement for legacy Crimson `analysis/movement_runs` loading.

## Source Map

Use these run families for new Crimson readers:

| Need | Preferred Palette source |
| --- | --- |
| Per-track position, heading, speed, path distance, acceleration | `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/` |
| Swim-bout event windows and detector traces | `analysis/swim_bout_runs/<run>/<speed_level>/` |
| Per-bout physical movement, heading, eye-gaze summaries | `analysis/bout_kinematics_runs/<run>/` |
| Body frame, snout/tail landmarks, B-spline, subject-shape QC | `analysis/subject_shape_runs/<run>/` |

Legacy `analysis/movement_runs` may remain a compatibility path for old
archives, but it is not the current Palette motion source.

## Track Kinematics

`analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/` is the
analysis-grade motion source. It owns gap-aware speed, distance, heading, and
validity semantics.

Required reader fields:

```text
frame_indices
time_seconds
positions_px
positions_mm
heading_degrees
smoothed_heading_degrees
sample_valid
transition_valid
detection_source
```

Speed and distance fields may be exposed through the preferred grouped layout:

```text
movement/speed/<raw|filtered|smoothed|averaged>/
  px
  mm
  frame_path_distance_px
  frame_path_distance_mm
  acceleration_px
  acceleration_mm
  smoothed_acceleration_px
  smoothed_acceleration_mm
```

Reader fallback order:

1. Prefer `movement/speed/<level>/...`.
2. Fall back to flat `speed_<level>_px`, `speed_<level>_mm`,
   `frame_path_distance_<level>_px`, and `frame_path_distance_<level>_mm`.
3. For derivatives, fall back to `speed_derivatives/speed_<level>/...`.
4. Use historical flat `acceleration_*` arrays only for older archives that
   predate source-scoped acceleration.

`frame_indices` is sparse row-to-frame lineage. Crimson must not assume track
row index equals video frame index. Build a frame-to-row lookup for interactive
seeking.

Missing frames are gaps. Displaying them as zero speed is a UI choice, not the
stored analysis semantics.

## Swim-Bout Runs

`analysis/swim_bout_runs/<run>/<speed_level>/` is the canonical bout
segmentation candidate surface. It answers: what events did this detector or
speed level find?

Important fields:

```text
bouts
bout_points
inter_bout_intervals
global_metrics
detection_signal_mm_s        # present for transformed detector responses
speed_exponential_mm         # compatibility/plotting mirror when present
```

The `bouts` table includes frame and timing boundaries such as:

```text
start_frame
end_frame
core_start_frame
core_end_frame
start_time_s
end_time_s
duration_s
observed_duration_s
path_length_mm
path_length_px
net_displacement_mm
net_displacement_px
peak_detection_signal_mm_s
peak_physical_speed_mm_s
```

Frame boundaries are authoritative for overlay rectangles and slicing. Optional
interpolated threshold times are annotations, not replacements for frame
boundaries.

### Detector vs Physical Metrics

`speed_exponential` is a detector response, not an independent physical speed
measurement. It can be useful for bout segmentation and visualization, but
biological speed, path length, and active-duration metrics should be read from
declared physical movement sources such as `speed_filtered` or from linked
`analysis/bout_kinematics_runs`.

Crimson should label detector traces as detector responses when plotted.

## Matching Track Kinematics To Swim Bouts

Given a selected track-kinematics run, track ID, and speed level, Crimson should
discover compatible swim-bout candidates by lineage attrs:

```text
source_track_kinematics_run
track_id
detection_signal_source_level
detection_signal_source_path
movement_metric_source_level
```

Direct matches:

- selected `filtered` speed maps to subgroup `speed_filtered`
- selected `smoothed` speed maps to subgroup `speed_smoothed`

Transformed matches:

- `speed_exponential` is compatible only when its attrs point back to the
  selected source speed, for example
  `detection_signal_source_level = "filtered"`.

Do not auto-select a bout subgroup whose detector source points at a different
speed trace than the selected track-kinematics speed.

## Bout Kinematics

`analysis/bout_kinematics_runs/<run>/` is downstream of one exact
track-kinematics run and one exact swim-bout candidate. It owns physical
per-bout measurement policy.

Use it for:

- physical active duration
- physical active path length
- physical active mean and peak speed
- pre/post position and heading windows
- within-bout heading changes
- optional eye-gaze summaries

Primary group:

```text
movement/per_bout_metrics/
```

Key fields include:

```text
source_start_frame
source_end_frame
source_core_start_frame
source_core_end_frame
physical_active_start_frame
physical_active_end_frame
physical_active_duration_s
physical_active_duration_s_interpolated
physical_active_path_length_mm
physical_active_path_length_px
physical_active_mean_speed_mm_s
physical_active_peak_speed_mm_s
physical_active_valid
failure_reason_bytes
```

Crimson should not mutate swim-bout runs when displaying these metrics. Bout
kinematics is a linked measurement layer, not a replacement segmentation layer.

## Subject Shape Boundary

`analysis/subject_shape_runs/<run>` is the geometry/QC surface for body axes,
B-splines, snout/tail landmarks, and body-frame overlays. It is not the default
source for speed or path distance when a compatible track-kinematics run exists.

Subject-shape fallback motion labels are acceptable only as a preview/debug path
for archives missing `analysis/track_kinematics_runs`. If used, label them as
derived preview values.

## Recommended Crimson UI Slice

1. Discover `analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/`.
2. Let the user select track and speed level.
3. Build a sparse frame-to-track-row lookup from `frame_indices`.
4. Draw per-frame motion labels near the fish:
   - heading from `heading_degrees` or `smoothed_heading_degrees`
   - speed from selected `movement/speed/<level>/mm` or fallback flat arrays
   - px/s fallback only when mm/s is unavailable, with honest units
5. Discover matching `analysis/swim_bout_runs` candidates and overlay bout
   windows from `bouts/start_frame` and `bouts/end_frame`.
6. Load linked `analysis/bout_kinematics_runs` for per-bout measurement tables
   and histograms when present.
7. Keep subject-shape overlays independent from motion traces.

## Canary

Current feeding canary:

```text
/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
```

Useful current sources:

```text
analysis/track_kinematics_runs/offline/tk_hyst4_low2_s005/tracks/id_0
analysis/swim_bout_runs/<candidate>/<speed_level>
analysis/bout_kinematics_runs/<candidate>
analysis/subject_shape_runs/<candidate>
```

The archive has no useful `analysis/movement_runs` path for current Crimson
motion display.
