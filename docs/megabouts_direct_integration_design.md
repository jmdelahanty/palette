# Megabouts Direct Integration Design
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-05-01
-->

Purpose: define how Palette should call Megabouts directly from Palette Zarr
arrays without making Megabouts preprocessing, segmentation, or classifier
schemas canonical.

This document is based on a source-code audit of the local fork at:

```text
/home/delahantyj@hhmi.org/gitrepos/megabouts
```

## Summary

Direct Megabouts-on-Palette is a reasonable goal. Palette does not need an
external file export as the primary bridge because Megabouts constructors,
preprocessors, segmenters, and classifiers already accept NumPy arrays and
pandas DataFrames in memory.

However, Palette should not fork Megabouts or lock into Megabouts
preprocessing before we compare conventions and outputs. The first integration
should be an adapter that:

- reads exact Palette source runs
- converts Palette arrays into Megabouts in-memory objects
- runs Megabouts in a controlled audit or classification mode
- records all conversion/configuration details
- writes results back as derived imported/classifier outputs
- never mutates refined masks, subject-shape runs, tail-kinematics runs,
  swim-bout runs, or bout-kinematics runs

Parquet exports remain useful later for cross-recording grouped analytics.
They should not be required for direct runtime integration.

### Dependency Boundary And User Experience

Palette should be Megabouts-compatible, not Megabouts-dependent.

The default Palette installation should be able to produce Megabouts-compatible
tail posture views without requiring users to install or maintain Megabouts:

```text
Palette install
  -> subject-shape tail geometry
  -> megabouts-compatible tail posture view
  -> visualization/export/review
```

Actual Megabouts execution should be optional:

```text
optional Megabouts install
  -> Megabouts preprocessing/classifier execution
  -> Palette derived output runs with Megabouts provenance
```

Do not make Megabouts a required Palette dependency. Do not vendor Megabouts
source code or model weights into Palette. Commands that require Megabouts
should fail gracefully with an install/configuration message and should record
the Megabouts package version or checkout commit when they run.

If Palette implements a native preprocessing method using standard signal
processing operations, it should be named as a Palette method, for example
`palette_standard_tail_preprocessing`, not `megabouts`, unless it actually
calls Megabouts APIs.

### License And Attribution Boundary

The local Megabouts checkout is distributed under a non-commercial research and
academic-use license, not a permissive MIT/BSD/Apache-style license. The local
`pyproject.toml` classifies it as `License :: Other/Proprietary License`, and
`LICENSE.md` restricts use and derivative works to internal non-commercial
research and academic use.

Practical policy:

- Calling Megabouts as an optional installed dependency is acceptable for
  non-commercial academic workflows when the run records attribution and
  version/provenance.
- Copying, vendoring, or close-porting Megabouts source code into Palette should
  be avoided unless the project explicitly accepts the derivative-work/license
  implications.
- Independently implementing standard preprocessing operations in Palette is
  allowed as a Palette-native method, but it should not be described as
  Megabouts unless it calls Megabouts.
- Any publication or run report using Megabouts-derived outputs should cite
  Jouary et al., "Megabouts: a flexible pipeline for zebrafish locomotion
  analysis", bioRxiv, doi:10.1101/2024.09.14.613078.
- If a workflow involves fee-for-service, industry collaboration, or other
  commercial rights, the license question should be reviewed outside Palette
  development before using Megabouts.

### Current Direction

The first implementation target is classifier-only integration:

- Palette remains the source of truth for refined masks, subject shape, tail
  geometry, track kinematics, and swim-bout windows.
- The implemented `analysis/tail_posture_view_runs` writer resamples Palette
  tail geometry to Megabouts-style `K=11` ordered tail keypoints and writes
  `K=10` cumulative tail-angle channels using a Palette-owned compatibility
  implementation.
- Palette-selected `analysis/swim_bout_runs` define the bout windows that are
  sliced into classifier arrays.
- Megabouts is used as a bout classifier over those fixed arrays, not as the
  initial preprocessing or segmentation authority.
- Results land in a separate `analysis/bout_classification_runs` family with
  exact source refs and adapter provenance.

This direction does not block later Megabouts segmentation/preprocessing
comparisons. It only keeps those modes diagnostic until their conventions and
effects have been reviewed against Palette outputs.

## Megabouts Source-Code Contracts

### Tracking Inputs

Megabouts supports three tracking modes:

```text
full_tracking
head_tracking
tail_tracking
```

`TrackingConfig` requires an integer FPS in the range `[20, 700]`.

The most relevant constructors are:

```python
FullTrackingData.from_posture(
    head_x: (T,),
    head_y: (T,),
    head_yaw: (T,),
    tail_angle: (T, N_segments),
)

FullTrackingData.from_keypoints(
    head_x: (T,),
    head_y: (T,),
    tail_x: (T, N_keypoints),
    tail_y: (T, N_keypoints),
)

TailTrackingData.from_keypoints(
    tail_x: (T, N_keypoints),
    tail_y: (T, N_keypoints),
)
```

Megabouts normalizes tail posture to 10 segments. If keypoints are not already
11 points, it interpolates them to 11 points. If tail angles are not already 10
segments, it interpolates them to 10 segments.

Megabouts examples multiply positions by `mm_per_unit`; its trajectory speeds
and classifier assumptions should therefore be treated as metric-unit oriented.

### Tail Keypoints And Tail Angle Semantics

Megabouts tail angles are radians. When computed from keypoints, they are
cumulative signed angles between the body-to-tail-base vector and successive
tail segments.

For Megabouts keypoint input, the standard target is:

```text
tail_x: (T, 11)
tail_y: (T, 11)
```

These 11 ordered tail keypoints produce 10 Megabouts tail-angle segments:

```text
tail_angle: (T, 10)
```

The points should be ordered from the tail base / swim-bladder side toward the
tail tip. In Palette, these are not manual pose labels; they are evenly sampled
points along the mask-derived subject-shape tail curve or B-spline.

This differs from the first Palette tail-kinematics representation:

- Palette `tail_angle_rad` is a body-frame tangent angle at normalized tail
  samples.
- Megabouts `tail_angle` is a cumulative segment-angle representation.

They may be related, but they are not safe to treat as identical until a
recording-level convention audit confirms sign, offset, coordinate handedness,
and segment placement.

The current canary audit supports that separation: direct sign is better than
sign-flipped, but residuals remain large enough that the first classifier
adapter should derive Megabouts angles from K=11 keypoints instead of passing
Palette `tail_angle_rad` directly.

Palette's current behavior-facing `analysis/tail_kinematics_runs` default is
`K=10` angle samples. That default should remain valid. It is reasonable to
try a `K=11` behavior-facing tail-kinematics candidate for review, especially
because it makes the sampled markers look similar to Megabouts keypoint input.
If `K=11` is used this way, it should be recorded as an explicit candidate
parameter, not silently treated as a schema replacement.

For the first Megabouts classifier adapter, use
`analysis/tail_posture_view_runs/<run>/tail_angle_rad` rather than Palette
native `analysis/tail_kinematics_runs/<run>/tail_angle_rad`. The posture view
derives Megabouts-compatible `tail_angle` from `K=11` keypoints and records the
algorithm boundary in attrs. Do not pass Palette native `tail_angle_rad`
directly as Megabouts `tail_angle` until the convention audit proves the
mapping is safe.

These arrays are persisted as a tool-specific posture view, not as a
replacement `tail_kinematics_runs` output:

```text
analysis/tail_posture_view_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_posture_view_runs"
    schema_version                    1
    method                            "tail_posture_view_from_subject_shape"
    method_version                    1
    view_family                       "megabouts_compatible"
    compatible_tool                   "megabouts"
    dependency_policy                 "no_megabouts_dependency_required"
    source_subject_shape_run
    source_subject_shape_path
    source_refined_subject_masks_run
    source_tail_kinematics_run        optional comparison source
    source_tail_geometry_kind         "subject_shape_tail_curve_resample"
    head_source                       "head_endpoint_xy" | "snout_tip_xy"
    keypoint_count                    11
    angle_count                       10
    angle_convention                  "megabouts_cumulative_segment_angle"
    keypoint_order                    "tail_base_to_tail_tip"
    algorithm_provenance

  frame_index
  row_index/                          copied source lineage when available
  valid
  failure_reason_bytes
  head_xy                             (N, 2)
  head_yaw_rad                        (N,)
  tail_keypoints_xy                   (N, 11, 2)
  tail_angle_rad                      (N, 10)
  tail_angle_deg                      (N, 10)
```

The feeding canary run
`tail_posture_view_megabouts_compatible_canary_20260501` wrote 17,495 valid
rows and 1,740 invalid rows from 19,235 ROI rows in about 4.2 seconds.

### Preprocessing

Megabouts tail preprocessing expects DataFrame columns:

```text
angle_0 ... angle_9
```

It then:

1. interpolates short NaN gaps using nearest-neighbor interpolation
2. marks rows with remaining NaNs as `no_tracking`
3. replaces remaining NaNs with zero
4. denoises angles with PCA
5. smooths with a Savitzky-Golay filter
6. computes a baseline
7. subtracts the baseline
8. computes tail vigor from absolute angular derivatives plus boxcar smoothing

Megabouts trajectory preprocessing expects:

```text
x, y, yaw
```

It unwraps yaw, linearly interpolates short gaps, marks longer gaps as
`no_tracking`, fills remaining values for filtering, applies a one-euro filter,
computes axial/lateral/yaw speed, and computes a trajectory activity signal.

Important Palette implication: invalid Palette frames should enter Megabouts as
NaNs, not zeros. Megabouts will zero-fill only after it has recorded
`no_tracking`.

Palette should distinguish three preprocessing cases:

```text
megabouts_compatible_posture_view
  Clean geometric adapter. No Megabouts dependency required.

palette_standard_tail_preprocessing
  Palette-native standard signal-processing implementation. No Megabouts
  dependency required. Must not be labeled as Megabouts output.

megabouts_tail_preprocessing
  Optional third-party execution that directly calls Megabouts APIs. Requires
  Megabouts to be installed/configured and records license/citation/version
  provenance.
```

If Megabouts preprocessing outputs are persisted, use a separate run family so
different preprocessing configs can be compared without regenerating the
geometric posture view:

```text
analysis/tail_posture_preprocessing_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_posture_preprocessing_runs"
    schema_version                    1
    preprocessing_family              "megabouts" | "palette_standard_tail_preprocessing"
    source_tail_posture_view_run
    config_json
    api_entrypoint                    optional, for third-party calls
    package_version                   optional
    package_git_commit                optional
    license                           optional
    citation                          optional

  frame_index
  angle_raw_rad
  angle_processed_rad
  angle_baseline_rad
  tail_vigor
  no_tracking
  valid
  failure_reason_bytes
```

### Segmentation And Classification

Megabouts has two segmentation modes:

- tail segmentation: threshold tail vigor
- trajectory segmentation: find peaks in trajectory kinematic activity

Default bout duration is 200 ms. The classifier input is capped at 140 frames.
For full tracking, classifier input packs:

```text
channels 0:7   first seven tail-angle channels
channels 7:10  trajectory x, y, heading
```

The classifier writes:

```text
cat
subcat
sign
proba
first_half_beat
```

Megabouts category labels include:

```text
approach_swim
slow1
slow2
short_capture_swim
long_capture_swim
burst_swim
J_turn
high_angle_turn
routine_turn
spot_avoidance_turn
O_bend
long_latency_C_start
short_latency_C_start
```

## Palette Source Surfaces

Palette already has the required primitives, but they are split across several
run families.

### Tail Geometry Authority

```text
analysis/subject_shape_runs/<run>/
  row_index/frame_indices
  components/subject_body/snout_tip_xy
  components/subject_body/head_endpoint_xy
  components/subject_body/tail_base_xy
  components/subject_body/tail_tip_xy
  components/subject_body/tail_sample_s
  components/subject_body/tail_sample_xy
  components/subject_body/tail_tangent_xy
  components/subject_body/tail_curvature_px_inv
  components/subject_body/tail_sample_valid
  body_frame/forward_axis_xy
  body_frame/left_axis_xy
  body_frame/valid
```

These arrays are ROI-row aligned and are the preferred source for
mask-derived tail keypoints and body-frame polarity.

### Tail Kinematics

```text
analysis/tail_kinematics_runs/<run>/
  frame_index
  valid
  failure_reason_bytes
  tail_angle_sample_s
  tail_angle_sample_xy
  tail_angle_rad
  tail_angle_deg
  tail_tip_angle_rad
  tail_lateral_deflection_px
  tail_angle_rms_rad
  integrated_abs_tail_angle_rad
  tail_curvature_px_inv
```

These arrays are behavior-facing Palette traces. They are useful for plotting
and Palette-native summaries, but should not be passed to Megabouts as
Megabouts `tail_angle` until the convention audit is complete.

### Trajectory And Heading

```text
analysis/track_kinematics_runs/<scope>/<run>/tracks/id_<track>/
  frame_indices
  positions_px
  positions_mm
  heading_radians
  smoothed_heading_radians
  sample_valid
  transition_valid
```

These arrays are track-level and frame/time ordered. They are the preferred
source for Megabouts trajectory channels when positions are calibrated to mm.

### Bout Boundaries

```text
analysis/swim_bout_runs/<run>/<speed_level>/bouts
```

Palette swim-bout runs own Palette-selected bout windows. Megabouts may either
segment its own bouts or classify Palette-provided windows, but it must not
overwrite Palette swim-bout runs.

### Existing Bout Summaries

```text
analysis/bout_kinematics_runs/<run>/
```

This is the Palette-owned per-bout heading/movement/eye-gaze summary layer.
Megabouts classifier labels should be separate from it unless a future
Palette-native bout-classification comparison layer explicitly joins them.

## Direct Integration Modes

### Phase 0: Convention Audit

Goal: determine whether Palette tail geometry and Megabouts tail-angle
conventions agree.

Inputs:

- `subject_shape_runs/<run>/components/subject_body/head_endpoint_xy`
- `subject_shape_runs/<run>/components/subject_body/tail_sample_xy`
- `tail_kinematics_runs/<run>/tail_angle_rad`

Procedure:

1. Resample Palette tail geometry to 11 Megabouts-style keypoints from tail
   base to tail tip.
2. Use a head point in the same coordinate frame, preferably
   `head_endpoint_xy` or `snout_tip_xy`.
3. Compute Megabouts-compatible cumulative segment angles from those keypoints.
4. Compare cumulative segment angles to Palette `tail_angle_rad`.
5. Report sign, offset, correlation, residuals, and frames where the mapping
   fails.

Output: completed as a convention audit and implemented as
`analysis/tail_posture_view_runs` for the first subject-shape-derived view.

### Phase 1 Target: Palette Bout Windows With Megabouts Classifier

Goal: classify Palette-selected bout windows without accepting Megabouts
segmentation as the bout-definition authority.

Inputs:

- exact Palette `analysis/swim_bout_runs/<run>/<speed_level>/bouts`
- Megabouts-compatible tail angle time series from
  `analysis/tail_posture_view_runs/<run>/tail_angle_rad`
- `head_x/head_y`: preferably calibrated mm positions from
  `track_kinematics_runs/.../positions_mm`, or a documented subject-shape
  head point only if it is converted into the same global coordinate system
- `head_yaw`: Palette heading converted to Megabouts yaw convention
- optional Megabouts preprocessing outputs only when explicitly selected

Procedure:

1. Resolve source Palette runs and verify row/frame mapping.
2. Read the source posture view and reject rows where `valid == false`.
3. Slice Palette-selected bout windows from `analysis/swim_bout_runs`.
4. Build fixed-duration `tail_array` with shape
   `(n_bouts, 10, bout_duration_frames)`.
5. Build fixed-duration `traj_array` with shape
   `(n_bouts, 3, bout_duration_frames)`.
6. Use `BoutClassifier.run_classification(tail_array=..., traj_array=...)`.
7. Store labels in a Palette classifier/import run with a source reference to
   the exact Palette bout candidate.

This mode is useful for comparing classifier labels across Palette bout
segmentation candidates.

#### Implemented Readiness Check

Palette now includes a read-only classifier-input dry run:

```bash
scripts/py -m fisheye.analysis.megabouts_classifier_inputs <analysis.zarr> \
  --tail-posture-view-run latest \
  --track-kinematics-run latest \
  --track-scope offline \
  --track-id 0 \
  --swim-bout-run latest \
  --speed-level default \
  --json
```

This command resolves the exact Palette source runs, builds the in-memory
`tail_array` and `traj_array` tensors, reports per-bout validity coverage, and
prints a JSON summary. It does not import Megabouts, call Megabouts, write
classifier labels, or mutate the Zarr archive.

Use this dry run before implementing or executing the optional Megabouts
classifier call. A successful dry run means the Palette-side source surfaces are
resolvable and shaped correctly; it does not mean Megabouts labels have been
computed.

### Deferred Mode: Megabouts Preprocessing And Segmentation Comparison

Goal: compare Megabouts tail-vigor or trajectory-vigor segmentation against
Palette swim-bout candidates.

Megabouts-generated onsets/offsets should be stored as a candidate in a
Megabouts/import run or a dedicated comparison run. They should not replace
`analysis/swim_bout_runs` unless Palette explicitly imports them as a new
segmentation candidate with full provenance.

## Invalid Frame Policy

Palette invalidity should propagate into Megabouts as NaNs and explicit masks.

Invalid sources include:

- refined subject-mask component QC failures
- subject-shape invalid rows
- missing or failed B-spline/tail samples
- tail-kinematics `valid = false`
- track-kinematics `sample_valid = false`
- missing calibrated positions when metric units are required
- swim-bout windows with insufficient valid tail or trajectory coverage

Adapter policy:

- Invalid tail frames: set all Megabouts tail-angle channels to `NaN`.
- Invalid trajectory frames: set `head_x`, `head_y`, and/or `head_yaw` to
  `NaN`.
- Do not replace invalid Palette frames with zeros before calling Megabouts.
- Let Megabouts create `no_tracking`, but preserve the original Palette
  invalidity mask separately.
- Reject or mark invalid any bout whose valid coverage is below a configured
  threshold.

Recommended initial thresholds:

```text
min_tail_valid_fraction_per_bout = 0.90
min_traj_valid_fraction_per_bout = 0.90
max_consecutive_invalid_frames_per_bout = configurable, default 2 frames
```

These should be parameters, not hardcoded constants.

## Storage Specification

### Preferred Result Family

Megabouts classifier outputs should land in:

```text
analysis/bout_classification_runs/<run>/
```

Suggested attrs:

```text
schema_id                         "analysis.bout_classification_runs"
schema_version                    1
classifier_family                 "megabouts"
classifier_name                   "megabouts_transformer"
classifier_version
megabouts_git_commit              optional local source commit
megabouts_package_version         optional installed package version
adapter_method                    "palette_megabouts_direct"
adapter_method_version
source_subject_shape_run
source_tail_kinematics_run
source_track_kinematics_run
source_track_path
source_swim_bout_run              optional
source_swim_bout_speed_level      optional
source_mode                       "palette_bouts" | "megabouts_tail_segmentation" | "megabouts_traj_segmentation"
tail_angle_conversion             JSON/string payload
trajectory_conversion             JSON/string payload
invalid_frame_policy              JSON/string payload
megabouts_config_json             JSON/string payload
created_at_utc
```

Suggested arrays:

```text
per_bout/source_bout_id           optional, -1 when Megabouts segmented
per_bout/start_frame
per_bout/end_frame
per_bout/HB1_frame
per_bout/category_id
per_bout/category_label_bytes
per_bout/subcategory_id
per_bout/sign
per_bout/probability
per_bout/tail_valid_fraction
per_bout/traj_valid_fraction
per_bout/valid
per_bout/failure_reason_bytes
```

Optional debug arrays, if small enough and explicitly requested:

```text
debug/tail_no_tracking
debug/traj_no_tracking
debug/tail_vigor
debug/traj_vigor
debug/megabouts_onset
debug/megabouts_offset
```

Do not duplicate dense source masks, full video, or canonical Palette tail
geometry in the classifier run.

### Preprocessing Comparison Runs

If we want to persist Megabouts preprocessing outputs before classification,
prefer a separate derived run:

```text
analysis/tail_posture_preprocessing_runs/<run>/
```

with method names such as:

```text
palette_tail_posture_preprocessing
megabouts_tail_preprocessing_adapter
```

This keeps preprocessing comparison independent from classifier labels.

## K Values And Sample Counts

Palette and Megabouts use related but distinct sample-count language:

- Megabouts keypoint input: `K_keypoints = 11`
- Megabouts angle output: `K_segments = 10`
- Palette current tail-kinematics default: `tail_angle_sample_count = 10`

For direct classifier integration, the adapter should resample Palette
subject-shape tail geometry to 11 ordered points before asking Megabouts to
compute its 10 cumulative angle channels.

For Palette-native behavior-facing traces, it is acceptable to generate a
parallel `tail_kinematics_runs` candidate with `tail_angle_sample_count = 11`
for comparison and visualization. That candidate should be named and attributed
explicitly, for example:

```text
tail_kinematics_k11_<label>
```

The existence of a K=11 candidate does not imply the default K=10 run is wrong.
The default should change only after side-by-side review shows K=11 is more
useful for Palette-native analysis, not just for Megabouts adapter symmetry.

## Fork Policy

Do not make Palette depend on the local Megabouts fork by default.

Use the fork for inspection and experiments. Prefer a Palette adapter that uses
an installed Megabouts package or an explicitly configured source checkout.

Fork or patch Megabouts only if one of these blockers appears:

- no supported way to classify externally supplied Palette bout windows
- no way to preserve or inspect no-tracking masks
- no way to control device/precision/config reproducibly
- classifier API requires internal pipeline state that cannot be constructed
  from public objects
- convention conversion needs a small reusable API that upstream is willing to
  accept

If a fork is used for a run, record the fork path and git commit in the output
attrs.

## Recommended Implementation Plan

1. Done: implement the convention audit and `analysis/tail_posture_view_runs`
   using 11 ordered tail keypoints from Palette subject-shape geometry.
2. Write a read-only adapter module that resolves one Palette recording,
   tail-posture-view run, track-kinematics run, track id, and swim-bout
   candidate.
3. Implement dry-run array construction for Palette-selected bout windows:
   `tail_array`, `traj_array`, invalid masks, coverage summaries, and config
   provenance.
4. Run `BoutClassifier` against Palette-defined bouts only.
5. Define and implement `analysis/bout_classification_runs` for classifier
   labels, confidence, source refs, and adapter provenance.
6. Optionally add a `tail_kinematics_runs` K=11 candidate for Palette-native
   review. Treat this as a comparison run, not a prerequisite for the adapter.
7. Defer Megabouts preprocessing/segmentation comparison until classifier-only
   integration has a reviewed canary result.
8. Add Marimo/Crimson read support for classifier labels only after the run
   schema stabilizes.

## Open Questions

- Does Megabouts `tail_angle` match Palette `tail_angle_rad` after a simple
  sign/offset conversion, or should Palette always derive Megabouts angles from
  keypoints?
- Should a K=11 behavior-facing Palette tail-kinematics candidate remain
  review-only, or should it become the default after comparison against the K=10
  run?
- Should trajectory channels use track-kinematics position centroids or a
  subject-shape-derived head point transformed to global coordinates?
- Which Palette heading source best matches Megabouts `head_yaw`:
  `heading_radians`, `smoothed_heading_radians`, or subject-shape body-frame
  heading?
- What coverage threshold should invalidate a bout before classification?
- Should imported Megabouts preprocessing traces be persisted, or only
  summarized in reports until we decide whether to adopt any of those methods?
