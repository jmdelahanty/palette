# Derived Analysis Run Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-04-26
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
- `detect_instance_rows`
- `frames`
- `tracks`
- `track_samples`
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

## Relationship To Existing Analysis Runs

Existing analysis outputs already follow this direction:

- `analysis/stimulus_runs/<run>` imports stimulus and alignment metadata.
- `analysis/track_kinematics_runs/<online|offline>/<run>` stores
  identity-resolved movement outputs.
- `analysis/eye_angle_runs/<run>` stores specialized eye-angle outputs
  interpreted relative to heading/keypoint context. New unified mask-derived
  eye geometry should be available from `analysis/subject_shape_runs/<run>`
  when a coherent body/eyes/swim shape run exists.
- `analysis/stimulus_response_runs/<run>` is the planned stimulus-aware
  downstream consumer.

New analysis families should follow the same `analysis/<analysis_type>_runs`
placement unless there is a clear reason they are an authority rather than a
derived product.

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
- [derived_metrics_schema_contract.md](derived_metrics_schema_contract.md)
- [current_pipeline_contract.md](current_pipeline_contract.md)
- [pose_kinematics_run_design.md](pose_kinematics_run_design.md)
- [stimulus_response_run_design.md](stimulus_response_run_design.md)
