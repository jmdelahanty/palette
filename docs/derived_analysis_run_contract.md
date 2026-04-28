# Derived Analysis Run Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-28
-->

Purpose: define the shared storage and provenance contract for deterministic
analysis products written under `analysis/`.

This contract is intentionally smaller than any one analyzer. It defines the
minimum semantics every derived analysis run must expose so downstream tools can
answer:

- what source artifacts were consumed
- what row or time axis the arrays use
- whether values are valid, missing, or failed
- whether the run can be regenerated when sources change

## Boundary

Palette should reserve root-level stage families for raw or refined authorities.

Use root-level stage families for:

- raw provenance snapshots, such as `subject_mask_runs/<run>`
- refined or editable authorities, such as `refined_subject_masks_runs/<run>`
- canonical curation surfaces, such as `refined_detect_runs/<run>`

Use `analysis/<analysis_type>_runs/<run>` for:

- deterministic measurements derived from one or more authorities
- biological geometry or behavior that depends on coordinate conventions
- temporal smoothing, event windows, track identity, or stimulus context
- summary/profile artifacts that can be regenerated from their sources

Derived analysis runs must not edit source authority arrays. If their source
changes, the derived run should be marked stale, superseded, or regenerated.

## Required Run Attributes

Every current analysis writer should record:

- `schema_id`: stable machine-readable schema name for this run family
- `schema_version`: integer schema version
- `method`: method or algorithm name
- `method_version`: implementation or contract version
- `created_at_utc`: ISO-8601 creation timestamp
- `row_axis`: declared alignment axis for primary arrays
- `source_refs`: dictionary of exact input runs and paths
- `parameters`: serialized user-visible parameters or config
- `provenance`: standard Palette provenance payload when available

Recommended `row_axis` values:

- `refined_subject_mask_rows`
- `refined_keypoint_rows`
- `keypoint_detection_rows`
- `detect_instance_rows`
- `frames`
- `tracks`
- `track_samples`
- `swim_bout_rows`
- `stimulus_steps`
- `profile_summary`

Writers may add narrower values, but they must be stable strings and documented
by the run family.

## Row-Aligned Analysis Runs

For row-aligned analysis runs, include a `row_index/` group when practical:

```text
analysis/<analysis_type>_runs/<run>/
  row_index/
    frame_indices
    detection_indices              optional
    source_refined_row_ids          optional
    source_row_indices              optional
```

The `row_index/` group should make it possible to map each derived row back to
the source authority without assuming physical row position is stable identity.

## Track-Aligned Analysis Runs

For track-aligned analysis runs, use the existing `tracks/id_<track_id>/`
pattern or an equivalent documented identity grouping. The run must state which
tracking or kinematics run resolved biological identity.

Required source refs when biological identity is used:

- `source_tracking_run` or `source_track_kinematics_run`
- any refined source run whose rows were sampled into the track
- coordinate-space and calibration inputs when measurements have physical units

## Validity And Failure State

Derived arrays should prefer explicit validity over implicit sentinel values.

Recommended pattern:

```text
<semantic_group>/
  value_array
  valid
  failure_reason_bytes              optional preferred string encoding
  failure_reason                    optional compatibility string array
```

Numeric outputs should use `NaN` for invalid floating-point values, but readers
must consult `valid` and reason arrays instead of inferring all failure state
from `NaN`.

Failure reasons should be short, stable tags such as:

- `missing_source_component`
- `empty_mask`
- `fit_failed`
- `ambiguous_polarity`
- `insufficient_valid_points`
- `source_row_stale`

## Body-Frame Support Data

Analyses that interpret values in fish-relative coordinates should declare the
body-frame source they used.

Preferred placement:

- shared mask/spline-derived body frames live in
  `analysis/subject_shape_runs/<run>/body_frame/`
- analysis-local support caches may live in
  `analysis/<analysis_type>_runs/<run>/support/body_frame/`
- `pose_schema.metadata.heading_computation` remains the keypoint-only fallback
  estimator for datasets without mask/spline body-frame products

Writers should record `body_frame_schema_id`, `body_frame_schema_version`,
`body_frame_estimator`, `body_frame_coordinate_space`,
`body_frame_angle_convention`, and exact `body_frame_source_refs` when a run
uses fish-relative coordinates. See `docs/body_frame_contract.md`.

## Relationship To Existing Analysis Runs

Existing analysis outputs already follow this direction:

- `analysis/stimulus_runs/<run>` imports stimulus and alignment metadata.
- `analysis/track_kinematics_runs/<online|offline>/<run>` stores
  identity-resolved movement outputs.
- `analysis/bout_kinematics_runs/<run>` stores per-bout heading and movement
  metrics derived from an exact swim-bout segmentation candidate without
  mutating that segmentation artifact. Schema v6 may also include an optional
  `eye_gaze/per_bout_metrics` subgroup with pre/post/within-bout eye-gaze and
  vergence summaries linked to an exact `analysis/eye_angle_runs/<run>` source.
- `analysis/eye_angle_runs/<run>` stores specialized eye-angle outputs
  interpreted relative to heading/keypoint context. Current v4 runs declare
  `schema_id = "analysis.eye_angle_runs"`, `schema_version = 4`,
  `method = "ellipse_and_centroid_eye_angles"`,
  `row_axis = "keypoint_detection_rows"`, and
  `eye_angle_output_schema` for output-group/units/suffix conventions.
  Readers should prefer explicit `*_gaze_*` arrays derived from the
  ellipse minor axis; legacy major/minor names remain compatibility outputs.
  Schema v4 retains the v3-compatible total axis-separation field
  `vergence_gaze_deg` and adds `left_nasal_gaze_deg`,
  `right_nasal_gaze_deg`, and BEAST/Johnson-comparable
  `mean_eye_vergence_gaze_deg`.
  Eye-angle writers should prefer `analysis/subject_shape_runs/<run>` eye
  geometry when a coherent body/eyes/swim shape run exists, and preserve
  refined-subject/refined-eye fallbacks as explicit lineage. Current v4 runs
  materialize keypoint-derived `support/body_frame/` arrays and future writers
  should prefer shared `analysis/subject_shape_runs/<run>/body_frame/` when
  available.
- `analysis/stimulus_response_runs/<run>` is the planned stimulus-aware
  downstream consumer.

New analysis families should follow the same `analysis/<analysis_type>_runs`
placement unless there is a clear reason they are an authority rather than a
derived product.

## Swim Bout Segmentation vs Per-Bout Metrics

`analysis/swim_bout_runs/<run>/<speed_level>/` is the bout segmentation
candidate surface. It should store the selected speed-level bout table,
inter-bout intervals, segmentation parameters, boundary semantics, and
segmentation-time summaries needed to review or compare candidate bout
definitions. Since bouts are frame-discrete, run schemas must persist both the
operator-facing duration parameters and their resolved frame counts. For example,
`min_gap_duration_s` should be paired with `resolved_min_gap_frames`,
`effective_min_gap_duration_s`, `min_gap_frame_source`, and the rounding policy
used to convert seconds to frames.

Frame-index bout boundaries remain the source of truth for row alignment.
Sub-frame timing should be stored as optional interpolated annotations beside
the frame fields. For swim-bout threshold detectors, those annotations describe
`core_*` threshold crossings, not `start/end` envelope boundaries, and consumers
must fall back to sampled frame times when interpolation is not valid. For
peak-event detectors, interpolated peak-width boundaries live in the aligned
`peak_events/` table and may be copied into downstream `source_peak_*` fields.

Any policy that changes whether candidate threshold regions are merged or split
must be recorded separately from boundary rendering. For example,
`gap_merge_policy="sampled_frame_gap"` and
`gap_merge_policy="interpolated_core_gap"` can produce different bout counts
from the same speed trace and threshold, so both the policy and its effective
minimum-gap threshold must live in run provenance and subgroup attrs.
Peak/event detection is a separate segmentation family, not a hidden extension
of threshold gap merging. See
[`swim_bout_peak_event_detector_design.md`](swim_bout_peak_event_detector_design.md).

`analysis/bout_kinematics_runs/<run>/<heading_level>/per_bout_metrics/` is the
linked measurement surface. It should store downstream per-bout biological
measurements computed from one exact swim-bout candidate and one exact
track-kinematics source. Examples include net heading change, pre/post position
means, interbout-epoch displacement, within-bout heading excursion, and explicit
validity/failure fields.

When the source swim-bout candidate has an aligned `peak_events/` table,
`bout_kinematics_runs` should preserve that source-boundary context as copied
`source_peak_*` columns and a `source_peak_events_path` lineage reference. These
fields do not replace integer frame boundaries. They let review tools and future
fractional-time analyses see the signal-derived peak-width boundaries while
current heading/position metrics continue to slice frame-indexed arrays by
`source_start_frame` and `source_end_frame`.

The rule is: changing bout detection creates or overwrites a `swim_bout_runs`
candidate; changing measurement logic creates or overwrites a
`bout_kinematics_runs` candidate. Neither surface should silently mutate the
other.

## Relationship To `derived_metrics_schema`

`derived_metrics_schema` describes the semantics of metric arrays inside a run.
This contract describes where derived analysis runs live and what lineage,
alignment, and validity metadata they must expose.

A run may use both:

- `analysis/<analysis_type>_runs/<run>.attrs` for source lineage and row-axis
  contract
- `<run>.attrs["derived_metrics_schema"]` for metric-level semantic metadata

## Subject Shape Placement

`analysis/subject_shape_runs/<run>` is the canonical location for interpreted
shape geometry derived from refined subject masks.

It should consume exact `refined_subject_masks_runs/<run>` sources and optional
keypoint, heading, tracking, or temporal context. It should not become a second
mask-pixel authority.

Subject-shape runs should be component-organized where possible:

- `body_frame/` stores shared row-aligned fish anatomical frame arrays when the
  run materializes a mask/spline/keypoint/hybrid body frame.
- `components/<component>` stores derived geometry whose primary subject is one
  semantic refined-mask component.
- expected first-class components are `subject_body`, `swim_bladder`,
  `eye_left`, and `eye_right` when those refined mask components are available.
- `relations/<relationship>` stores derived geometry whose meaning depends on
  multiple components or an external coordinate frame.
- expected first-class relations include `eye_pair`, `eyes_to_body`, and
  `swim_bladder_to_body` when the required components and coordinate frame are
  available.
- component groups in `analysis/subject_shape_runs` are not review or approval
  surfaces; source mask approval remains in `refined_subject_masks_runs`.

## Related Documents

- [subject_shape_runs_contract.md](subject_shape_runs_contract.md)
- [body_frame_contract.md](body_frame_contract.md)
- [derived_metrics_schema_contract.md](derived_metrics_schema_contract.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md)
- [bout_kinematics_run_design.md](bout_kinematics_run_design.md)
- [stimulus_response_run_design.md](stimulus_response_run_design.md)
