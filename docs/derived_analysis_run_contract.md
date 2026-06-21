# Derived Analysis Run Contract
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-05-01
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

For future cross-recording Parquet/Arrow exports built from derived analysis
runs, see
[cross_recording_analytics_export_design.md](cross_recording_analytics_export_design.md).

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
- `source_fingerprint`, `source_lineage_hash`, and `lineage_hash`:
  run-level SHA-256 aliases over the meaningful source/method/schema/parameter
  state when available
- `fingerprint_status`: `complete`, `best_effort`, or `missing`; current v1
  derived writers use `best_effort` until all refined authoring revisions and
  source-content hashes are complete
- `lineage_payload_json`: strict canonical JSON payload used to compute the
  run-level fingerprint

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

Realtime consumers should not be forced to scan sparse row arrays for every
displayed frame. Row-aligned and track-aligned analysis runs may add
`frame_index/` and `track_index/` lookup groups as non-authoritative
accelerators. See
[`realtime_sparse_row_index_contract.md`](realtime_sparse_row_index_contract.md).

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
- `analysis/stimulus_epoch_runs/<run>` should hold reusable event-aligned
  window definitions derived from one exact stimulus run. Downstream
  modality-specific analyses should reference this run when computing
  pre/training/post, trial, loom, or other event-aligned summaries rather than
  each owning independent window semantics. See
  [`stimulus_epoch_run_contract.md`](stimulus_epoch_run_contract.md).
- `analysis/track_kinematics_runs/<online|offline>/<run>` stores
  identity-resolved movement outputs.
- `analysis/bout_kinematics_runs/<run>` stores per-bout heading and movement
  metrics derived from an exact swim-bout segmentation candidate without
  mutating that segmentation artifact. Schema v7 includes
  `movement/per_bout_metrics` for physical-active movement summaries measured
  from a declared physical speed source, while preserving source
  detector-window durations. It may also include an optional
  `eye_gaze/per_bout_metrics` subgroup with pre/post/within-bout eye-gaze and
  vergence summaries linked to an exact `analysis/eye_angle_runs/<run>` source.
- `analysis/eye_angle_runs/<run>` stores specialized eye-angle outputs
  interpreted relative to heading/keypoint context. Current v5 runs declare
  `schema_id = "analysis.eye_angle_runs"`, `schema_version = 5`,
  `method = "ellipse_and_centroid_eye_angles"`,
  `method_version = "eye_angle_analysis.v5"`,
  `row_axis = "keypoint_detection_rows"`, and
  `eye_angle_output_schema` for output-group/units/suffix conventions.
  Readers should treat explicit `*_major_*` arrays as the canonical
  orientation fields and prefer explicit `*_gaze_*` arrays for biological gaze;
  gaze/minor direction is derived from the resolved major axis. Legacy major
  and minor names remain compatibility outputs. Schema v5 retains the
  v3-compatible total axis-separation field
  `vergence_gaze_deg` and adds `left_nasal_gaze_deg`,
  `right_nasal_gaze_deg`, and BEAST/Johnson-comparable
  `mean_eye_vergence_gaze_deg`.
  Output schema v6 additionally exposes Bianco/Engert-style eye-frame fields
  `left_eye_angle_deg`, `right_eye_angle_deg`, and
  `vergence_eye_angle_deg`, with per-eye nasal-positive signs and signed
  vergence where positive means convergence and negative means divergence.
  Output schema v7 adds `eye_angle_variant_schema`, mirrored in run attrs, so
  marimo, Crimson, and other consumers can present selectable angle
  representations (`eye_frame`, `gaze`, `nasal_gaze`, `major`, `centroid`,
  `legacy`) from metadata rather than hardcoded field lists.
  Eye-angle writers should prefer `analysis/subject_shape_runs/<run>` eye
  geometry when a coherent body/eyes/swim shape run exists, and preserve
  refined-subject/refined-eye fallbacks as explicit lineage. Current v5 runs
  materialize keypoint-derived `support/body_frame/` arrays and future writers
  should prefer shared `analysis/subject_shape_runs/<run>/body_frame/` when
  available.
- `analysis/stimulus_response_runs/<run>` is the implemented stimulus-aware
  downstream consumer for protocol-step summaries and stimulus-specific
  adapters.
- `analysis/chaser_distance_runs/<run>` stores framewise offline fish-to-chaser
  distances and epoch summaries derived from exact refined detections,
  `analysis/stimulus_runs`, and `analysis/stimulus_epoch_runs` sources. It is a
  modality-specific stimulus analysis product, not a replacement for imported
  chaser state. Its `epoch_distributions/` subgroup stores reusable histogram
  bins/densities so distribution-shape plots do not have to rescan dense
  framewise arrays. See
  [`chaser_distance_run_contract.md`](chaser_distance_run_contract.md).
- Detection-derived per-recording analyses should avoid a generic
  `detection_summary_runs` family. Use the implemented
  `analysis/detection_profile_runs/<run>` surface for scalar/profile summaries,
  `analysis/detection_occupancy_runs/<run>` for epoch-aligned heatmaps and
  spatial occupancy summaries that consume `analysis/stimulus_epoch_runs`, and
  the current
  `analysis/detection_comparison_runs/<run>` compatibility surface for
  realtime-vs-offline detection/crop-sufficiency diagnostics. See
  [`detection_analysis_run_surfaces.md`](detection_analysis_run_surfaces.md)
  and
  [`spatial_occupancy_zone_summary_design.md`](spatial_occupancy_zone_summary_design.md).

New analysis families should follow the same `analysis/<analysis_type>_runs`
placement unless there is a clear reason they are an authority rather than a
derived product.

Run-level lineage fingerprint attrs are intentionally narrower than full
provenance. They exclude output path, run name, timestamps, hostname, scheduler,
wall time, and other operational details that do not change scientific outputs.
They include source refs/fingerprints, schema, method, method version,
parameters, and code revision when those are exposed by the writer. This makes
cross-recording manifests and Parquet exports able to compare selected runs
without parsing every full provenance payload. See
[`virtual_collection_manifest_schema.md`](virtual_collection_manifest_schema.md)
for collection/export use of these fingerprints.

## Read-Only Staleness Audit

Use `fisheye.utils.audit_analysis_staleness` to check whether existing derived
analysis runs still point at resolvable, fingerprint-compatible sources:

```bash
scripts/py -m fisheye.utils.audit_analysis_staleness <archive>.zarr
scripts/py -m fisheye.utils.audit_analysis_staleness /nvme1/recordings --recursive --json
```

The audit is read-only. It resolves same-archive paths from `source_refs` and
common `source_*_run` attrs, compares any recorded source fingerprints against
the current source group's `lineage_hash`/`source_lineage_hash`, reports explicit
`*_stale.state="stale"` payloads on sources, and warns when a source does not
match its parent `latest` selection. Pass `--require-latest-sources` when a
workflow treats non-latest upstream refs as stale rather than merely historical.

Statuses:

- `fresh`: all discovered sources resolve and fingerprint checks match.
- `stale`: a source is missing, explicitly stale, has a fingerprint mismatch,
  or is non-latest when `--require-latest-sources` is active.
- `warning`: lineage is incomplete or a source is not latest in non-strict mode.
- `no_sources`: no machine-readable source references were discovered.

This tool does not replace writer-side source revision capture. It is a gate and
diagnostic surface for finding derived runs that need regeneration or better
lineage metadata.

For a graph view of the same source relationships, use
`fisheye.utils.inspect_run_lineage_graph`. It builds a read-only run-lineage DAG
from the same `source_refs` / `source_*_run` metadata and renders canonical
node/edge JSON plus tree, Mermaid, or DOT projections. See
[run_lineage_dag_inspector.md](run_lineage_dag_inspector.md).

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

Detector signals and measurement sources must remain separate. A subgroup such
as `speed_exponential` may use a transformed detector response to define bout
boundaries, but physical movement summaries should declare and consume their
measurement source, for example `movement_metric_source_level="filtered"` with
`frame_path_distance_filtered_*` arrays. Existing fields follow this split by
separating `peak_detection_signal_mm_s` from `peak_physical_speed_mm_s`.

This separation also defines viewer behavior. Selecting a speed trace in a UI
should not imply that bout candidates are stored inside that speed group.
Instead, the UI should discover `analysis/swim_bout_runs` candidates linked to
the same track-kinematics run and `track_id`, then filter speed-level subgroups
by source metadata. Direct candidates match the selected speed level itself;
transformed candidates are compatible only when their detector-source metadata
points back to the selected speed source. This lets a viewer auto-show or
auto-use bout segmentations for the selected speed trace while preserving
`swim_bout_runs` as an independently versioned event-candidate family.

The boundary-duration fields in `swim_bout_runs` are still detector-boundary
measurements. `duration_s`, `observed_duration_s`, and `core_duration_*`
describe the stored bout window/core chosen by the detector and boundary
policy. They should not be silently reinterpreted as physical active duration
when the detector signal is transformed or broadened.

Schema v7 makes the detector-vs-estimator split first-class in
`analysis/bout_kinematics_runs/<run>/movement/per_bout_metrics/`. That table
copies detector-window duration fields into `detector_*` columns and writes
separate physical-estimator columns such as `physical_active_duration_s`,
`physical_active_duration_s_interpolated`,
`physical_active_path_length_mm`, `physical_active_mean_speed_mm_s`, and
`physical_active_peak_speed_mm_s`. The physical estimator records
`physical_active_boundary_policy="physical_active"` and keeps the boundary
search constraint as a separate enum, currently `clip_to_detector`,
`search_with_margin`, or `allow_extension`, with an explicit margin parameter
when applicable.

Future schema bumps should keep this split lean to avoid drift. Prefer grouped
metadata such as `detection_signal = {array_path, transform, transform_params}`
and `movement_estimator = {signal_array_path, path_distance_array_path,
position_array_path, validity_array_paths}` over many duplicated level/array
attrs. Metric metadata should identify only the metric source role
(`detector_boundary` or `physical_estimator`) and boundary policy
(`detector_start_end`, `detector_core`, or `physical_active`), plus any
metric-specific threshold/interpolation parameters.

An optional future `detector_response/per_bout_metrics` subgroup may summarize
the detector response itself, for example peak value, response area, response
width, rise time, or decay time for `speed_exponential`. That surface should be
diagnostic and should reference the source swim-bout detector metadata rather
than pretending the response trace is measured fish motion. Physical movement
estimates remain under `movement/per_bout_metrics`.

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
