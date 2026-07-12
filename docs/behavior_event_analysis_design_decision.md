# Behavioral Event And Trajectory Analysis Decision

<!-- decision-meta
status: accepted-design
created: 2026-07-10
owner: jeremy
scope: behavioral experimental units, stimulus-event alignment, trajectory
  analysis, dense Zarr traces, Parquet analytics exports, and visualization
  backends
depends_on: docs/dataset_reporting_contract.md,
  docs/cross_recording_analytics_export_design.md
related_decision: docs/bout_morphology_collection_design_decision.md
-->

## Decision Summary

Palette will treat behavioral measurements as hierarchical data. Frames, bouts,
and repeated stimulus events are measurements nested within a biological or
experimental unit; they are not independent replicates. The independently
treated fish, recording, dish, or other assignment unit controls inferential
sample size and uncertainty.

Stimulus-response analysis will preserve event-level observations while
separating three questions:

1. whether an animal responded;
2. when it responded;
3. what the conditional response trajectory or kinematics looked like.

Dense frame-aligned traces remain authoritative in recording Zarr analysis
runs. Cross-recording Parquet exports contain recording-, step-, fish-, bout-,
event-, endpoint-, and selected histogram-bin-level rows. Optional trace-level
Parquet exports may be added for recurring cohort analyses, but exporting every
dense trace is not the default.

Zarr-backed and Parquet-backed visualizers are complementary source adapters,
not automatically separate plot implementations. They share a renderer only
when they produce the same normalized render-data contract and answer the same
scientific question.

The portable, cross-recording delivery of contours and synchronized
swim-bout measurements is specified separately in
[Bout Morphology Collection Design Decision](bout_morphology_collection_design_decision.md).

## Why This Decision Exists

Palette recordings can contain many frames, bouts, and stimulus presentations
per fish. Treating all of those rows as independent animals would inflate
effective sample size and understate uncertainty. At the other extreme,
retaining only one final mean or histogram per cohort would discard response
latency, trial variability, trajectory geometry, habituation, and alternative
future endpoints.

The storage and analysis architecture must therefore preserve fine-grained
measurements without confusing measurement count with biological replication.
It must also support both recording-local diagnostics and reproducible cohort
analysis without making Parquet the authority for dense scientific arrays.

## Hierarchy And Experimental Unit

A common hierarchy is:

```text
batch or acquisition day
  dish or recording
    biological fish or run-local track
      stimulus event or trial
        bout
          frame
```

The experimental unit is the smallest unit independently assigned to the
condition being tested. Depending on experimental design, this may be a fish,
dish, recording, or batch. Track identity must not be silently treated as
cross-recording biological identity.

Analyses and plots report biological units and measurement units separately,
for example `N=12 fish, n=438 stimulus events`. Additional events improve the
within-fish estimate but do not create additional fish.

When event counts differ across fish, a pooled event mean answers an
event-weighted question and gives prolific animals more influence. A fish- or
recording-weighted estimand first summarizes within the experimental unit or
uses a hierarchical model. The chosen estimand must be explicit.

## Stimulus-Response Decomposition

Every stimulus-event family should represent these outcomes independently:

### Response occurrence

A binary or categorical outcome records whether a contract-defined response
occurred. Nonresponses remain present. Appropriate inference may use a
hierarchical logistic or multinomial model.

### Response latency

Latency is measured from a declared stimulus anchor to response onset.
Nonresponses are censored observations rather than missing successful trials.
Time-to-event methods are preferred when censoring is material.

### Conditional response kinematics

Trajectory shape, speed, displacement, heading change, eye response, and bout
structure may be analyzed conditional on a response. Conclusions from this
surface apply to response form, not response probability.

This separation avoids conditioning on successful responses and then
incorrectly generalizing the result to overall stimulus effectiveness.

## Alignment Contracts

Palette will retain both stimulus-locked and response-locked representations
when both questions matter.

- **Stimulus locked:** `relative_time_s = 0` is the exact stimulus event. This
  supports response probability, latency, and baseline-to-response dynamics.
- **Response locked:** `relative_time_s = 0` is the detected bout, turn, or
  escape onset. This supports conditional motor-program and trajectory
  comparisons.

Each alignment run records:

- event definition and event ID;
- source stimulus run and step;
- anchor camera frame and timestamp;
- pre/post window and baseline interval;
- sampling grid and interpolation policy;
- overlap/censoring policy for neighboring events;
- response detector and threshold contract;
- validity and missingness masks;
- exact source analysis runs and lineage hashes.

Windows and thresholds should be selected independently of the tested effect
when possible. Long tracking gaps are not silently bridged. Interpolated and
observed samples remain distinguishable.

## Trajectory Coordinate Contracts

Absolute camera or arena trajectories remain available for diagnosis. Derived
event-centered coordinate frames may translate the position at the anchor to
the origin and rotate by one declared direction:

- fish heading at the anchor;
- stimulus direction;
- chaser/predator bearing;
- arena axis.

Optional mirroring of left/right trials is allowed only when symmetry is a
scientific assumption declared in the contract. The transform, handedness,
angle convention, origin, units, and source frame are persisted. Useful
derived components include radial/tangential displacement and egocentric
bearing.

Different coordinate normalizations are different data contracts. They must
not be inferred from figure appearance.

## Tail Spline And Posture Analysis

Tail movement is represented as a time-varying curve rather than as independent
sampled `x,y` points. For normalized arc length `s` from tail base to tail tip
and time `t`, the primary behavioral fields are:

```text
theta(s, t)   signed local tangent angle in the body frame
kappa(s, t)   signed curvature, approximately d theta / d s
```

Directly averaging camera-space spline points across fish is not an accepted
comparison because translation, rotation, scale, and body polarity would
dominate the result.

### Spatial normalization

For every valid row:

1. resolve tail base, tail tip, forward axis, and anatomical-left axis;
2. translate the tail base to the origin;
3. rotate the sampled curve into the declared fish body frame;
4. parameterize the curve by monotonic normalized arc length `s in [0, 1]`;
5. resample every tail onto the same spatial grid;
6. normalize lateral displacement by body or tail length for cross-fish use;
7. store dimensionless curvature such as `kappa * reference_length` in
   addition to source curvature units.

Absolute camera/arena spline points remain available for overlay diagnosis.
The body-frame transform, reference length, handedness, polarity, angle
convention, units, and source geometry are persisted.

### Palette source contracts

Palette currently has two valid but distinct tail-angle surfaces:

- `analysis/tail_kinematics_runs/<run>/tail_angle_rad` contains Palette-native
  body-frame tangent angles sampled along normalized tail position, together
  with curvature, lateral deflection, tail-tip angle, RMS angle, and integrated
  metrics;
- `analysis/tail_posture_view_runs/<run>/tail_angle_rad` contains cumulative
  segment angles from 11 ordered tail keypoints, yielding 10
  Megabouts-compatible channels.

These conventions must retain distinct schema and provenance identities. They
are not concatenated, averaged together, or passed interchangeably to a
classifier. The original subject-shape B-spline and sampled points remain the
geometry authority from which either view can be regenerated.

### Chronological track binding

Subject-shape, tail-kinematics, and tail-posture arrays are currently aligned to
ROI rows. Before time-series analysis, each valid row must be joined through
its exact source frame and observation identity to a concrete tracking run and
run-local `track_id`:

```text
tail ROI row
  -> source frame and observation identity
  -> tracking run and track_id
  -> chronological sample for one fish/track
```

Row order must not be treated as time order, and single-fish recordings must
not cause the general contract to hard-code track 0. Duplicate tail rows for
one track/frame, missing associations, identity changes, and track gaps receive
explicit resolution or invalid status.

The resulting logical continuous arrays have shape:

```text
tail_angle[track, time, tail_position]
tail_curvature[track, time, tail_position]
valid[track, time, tail_position]
```

### Spatial versus temporal sampling

The number of spline points controls spatial resolution along the tail. Video
frame rate and timestamps control temporal resolution. A dense spatial spline
does not recover tail beats that were undersampled in time. Tail-beat
frequency, phase lag, and curvature-wave speed require a documented effective
sampling rate and anti-aliasing assessment.

Analyses use timestamps when available and record whether the time grid is
regular. Derivative-based measurements require a frozen smoothing policy and
sensitivity checks because curvature and acceleration amplify geometry noise.

### Event-aligned tail tensor

For stimulus or bout analysis, fixed windows produce a dense tensor:

```text
tail_angle[event, relative_time, tail_position]
tail_curvature[event, relative_time, tail_position]
valid[event, relative_time, tail_position]
observed[event, relative_time, tail_position]
```

Stimulus-locked tensors preserve response occurrence and latency. Bout-locked
tensors compare conditional motor execution. Absolute-time windows remain
primary. Optional phase-normalized bouts may compare shape after stretching to
a common 0--100% duration, but they cannot replace absolute duration,
frequency, or latency measures.

### Tail visualizations

The standard recording/event visualization includes:

- signed tail-angle and curvature kymographs with relative time on the x-axis
  and normalized tail position on the y-axis;
- body-frame tail outlines at declared relative times;
- tail-tip angle, RMS angle, integrated curvature, and validity over time;
- head or centroid trajectory in the matching event-centered frame;
- coverage and failure-reason summaries.

Cohort views include individual-fish mean kymographs, condition mean and
difference kymographs, experimental-unit bootstrap uncertainty, endpoint
distributions, and examples chosen by a reproducible rule. Every plot reports
fish/recording, event, bout, and valid-frame counts separately.

### Low-dimensional posture and bout representations

Two reductions are supported:

- **spatial posture PCA:** fit a common basis across frame-level tail-angle
  vectors and retain component scores over time;
- **spatiotemporal bout PCA or functional PCA:** flatten or functionally model
  the `relative_time x tail_position` tensor for each aligned bout.

For descriptive cohort work, a basis may be fit once on the declared reference
cohort. Predictive analysis fits the basis on training fish only and projects
held-out fish. Separate per-condition PCA bases must not be compared as though
their component numbers had common meaning.

Continuous features remain available even when a bout classifier is used.
Megabouts-compatible categories summarize bout selection; they do not replace
tail amplitude, curvature, beat timing, wave propagation, or trajectory
measurements.

### Interpretable tail endpoints

Candidate per-event endpoints include:

- maximum tail-tip angle and lateral deflection;
- maximum and integrated absolute angle or curvature;
- tail-beat amplitude, frequency, and count;
- left/right asymmetry and first-bend direction;
- normalized location of maximum curvature;
- proximal-to-distal phase lag and curvature-wave speed;
- bout duration, displacement, heading change, and posture-PC scores.

The analysis keeps response selection separate from motor execution: a
condition may alter response probability, bout-category selection, execution
within a category, or resulting head trajectory. These are different
estimands.

### Canonical tail-event products

The dense Zarr product is:

```text
analysis/tail_event_runs/<run_id>/
  events/
    event_id
    track_id
    stimulus_step_index
    stimulus_anchor_frame
    bout_anchor_frame
    response_status
  samples/
    relative_time_s
    tail_position_s
    tail_angle_rad
    tail_curvature_length_normalized
    lateral_deflection_body_length
    valid
    observed
  features/
    tail_tip_peak_angle
    max_curvature
    integrated_curvature
    tail_beat_frequency
    curvature_wave_speed
    posture_pc_scores
```

Compact cohort exports use `tail_event_endpoints`, with one row per event, and
`tail_event_pc_scores`, with one row per event and component. A long-form
`tail_event_aligned_samples` Parquet table is optional and introduced only for
a demonstrated cross-recording trace query. The dense tail tensor remains
authoritative in Zarr.

## Inference Policy

The initial analysis family should match outcome support and hierarchy:

| Outcome | Default candidate |
| --- | --- |
| Response success | Hierarchical logistic model |
| Censored latency | Survival/time-to-event model with unit effects |
| Count outcomes | Poisson or negative-binomial mixed model |
| Scalar continuous endpoints | Linear or generalized mixed model |
| Heading and turn angles | Circular model or sine/cosine vector model |
| Nonlinear time course | Generalized additive mixed model or functional mixed model |
| Exploratory contiguous time effects | Experimental-unit-level cluster permutation |

Models should consider fish/recording effects, batch/day effects, event order,
time since prior stimulus, baseline state, starting geometry, and tracking
coverage where scientifically relevant. Residual temporal autocorrelation must
not be ignored merely because traces were aligned.

Separate uncorrected hypothesis tests at every frame are not an accepted
default. Cluster procedures or simultaneous functional inference must exchange
or resample complete experimental units, not individual frames.

## Canonical Data Products

### Dense event-aligned Zarr run

```text
analysis/event_aligned_runs/<run_id>/
  events/
    event_id
    track_id
    stimulus_step_index
    stimulus_anchor_frame
    response_anchor_frame
    response_status
  samples/
    relative_time_s
    valid
    observed
    position_xy
    speed
    heading
    eye_angles
    convergence
```

The exact arrays are capability-dependent; absent measurements are not filled
with fabricated values. Z position may be added under the same coordinate and
validity rules.

### Event-level Parquet

`behavior_events` contains one row per track/fish and stimulus event. It stores
event identity, stimulus parameters, response status, latency/censoring,
baseline state, starting geometry, tracking coverage, event order, and source
lineage.

### Endpoint-level Parquet

`behavior_event_endpoints` contains one row per event and endpoint family, or a
contracted wide event row where the endpoint set is stable. Examples include
peak speed, displacement, maximum turn, bout count, duration, escape success,
and post-event occupancy.

### Optional aligned-sample Parquet

`behavior_event_aligned_samples` is introduced only for a demonstrated
cross-recording trace query. Its long-form grain is one event and relative-time
sample. It includes recording, track, event, alignment, coordinate, validity,
and source-run identity. Partitioning and decimation policies must be explicit
because this table can be orders of magnitude larger than endpoint tables.

## Visualization Policy

Recommended event-aligned review surfaces include:

- event-by-time heatmaps;
- individual experimental-unit mean traces;
- cohort traces with experimental-unit or hierarchical-bootstrap uncertainty;
- event-centered trajectory overlays and density maps;
- response probability and censored-latency plots;
- endpoint distributions showing experimental-unit summaries;
- explicit counts of fish, recordings, events, bouts, and valid frames.

Zarr is preferred for frame traces, trajectories, geometry, and recording
diagnostics. Parquet is preferred for cross-recording distributions,
experimental-unit comparisons, and group statistics. A semantic visualization
declares `zarr`, `parquet`, or `both` backend capability.

If both backends support a visualization, adapters construct the same
normalized render payload and use the same renderer. Backend-equivalence tests
must compare normalized payloads on a shared fixture. If aggregation or
scientific meaning differs, the outputs receive different semantic IDs and
visualization contracts even when their figures look similar.

Every artifact records its backend, concrete source runs or export tables,
collection/export manifest hashes, query and filters, aggregation unit,
coordinate and alignment contract, visualization contract, and renderer
version.

## Export And Report Placement

Reports derived from an indexed analytics export are co-located under an
immutable sibling of the Parquet table partitions:

```text
palette_analytics/v1/
  manifests/export_run_id=<export_run_id>.json
  <table>/export_run_id=<export_run_id>/part-*.parquet
  reports/
    export_run_id=<export_run_id>/
      report_id=<report_id>/
        report_manifest.json
        montages/
        artifacts/
```

The report manifest binds the export run and manifest hash. It distinguishes a
copied contracted Zarr visualization from one computed from exported Parquet;
physical placement does not imply derivation.

The future report registry is a one-to-many child of `analytics_exports`. It
indexes report identity, manifest path/hash, materialization policy, status,
and semantic visualization IDs. Detailed tile lineage remains in the report
manifest rather than generating one registry row per tile.

## Quality And Sensitivity Requirements

An analysis should expose or test:

- tracking and source-run coverage;
- missingness versus true inactivity;
- alternative reasonable smoothing and event-window choices;
- event overlap and censoring;
- habituation or event-order effects;
- batch, arena, and acquisition-day effects;
- influence of individual fish/recordings;
- conditional-response selection;
- coordinate-transform and calibration validity.

Plots alone are not statistical evidence. Numeric tables, model specification,
diagnostics, and exact cohort/export manifests remain part of the report.

## Consequences

### Benefits

- Event detail is preserved without frame-level pseudoreplication.
- Nonresponses and censored latencies remain analyzable.
- Dense trajectory work remains efficient in Zarr.
- Parquet remains queryable and appropriately sized for common cohort work.
- Recording and cohort plots can share renderers without conflating semantics.
- Reports remain tied to exact source and export versions.

### Costs

- Event detection, alignment, coordinate normalization, and endpoint extraction
  require explicit versioned contracts.
- Hierarchical models and unit-aware resampling are more involved than pooled
  frame or bout tests.
- Some cohort trace questions require a selective aligned-sample export.
- Biological identity uncertainty limits cross-recording subject-level models.

## Rejected Alternatives

### Treat every frame, bout, or event as an independent replicate

Rejected because repeated measurements from one unit are correlated and would
inflate apparent sample size.

### Export only final summaries and figure histograms

Rejected because it prevents alternative endpoints, model checks, habituation
analysis, and event-level sensitivity analysis.

### Export every dense trace to Parquet by default

Rejected because of size, duplicated authority, and the risk of encouraging
frame-level pooled inference. Trace exports remain purpose-built.

### Maintain unrelated Zarr and Parquet implementations of the same plot

Rejected because identical scientific semantics should share normalized render
data and a renderer. Different estimands receive different visualization
contracts rather than backend-specific lookalikes.

## Methodological References

- Lazic, 2010, [The problem of pseudoreplication in neuroscientific
  studies](https://pmc.ncbi.nlm.nih.gov/articles/PMC2817684/).
- Aarts et al., 2014, [Multilevel analysis quantifies variation in the
  experimental effect](https://pmc.ncbi.nlm.nih.gov/articles/PMC4684932/).
- Saravanan et al., 2022, [Analyzing nested experimental
  designs](https://pmc.ncbi.nlm.nih.gov/articles/PMC9098003/).
- Maris and Oostenveld, 2007, [Nonparametric statistical testing of EEG- and
  MEG-data](https://pubmed.ncbi.nlm.nih.gov/17517438/).
- Dunn et al., 2016, [Neural circuits underlying visually evoked escapes in
  larval zebrafish](https://pmc.ncbi.nlm.nih.gov/articles/PMC4742414/).
- Sridhar et al., 2024, [Uncovering multiscale structure in the variability of
  larval zebrafish navigation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11588111/).
- Di Santo et al., 2021, [Convergence of undulatory swimming kinematics across
  a diversity of fishes](https://pmc.ncbi.nlm.nih.gov/articles/PMC8670443/).
- Patterson et al., 2013, [Visually guided gradation of prey capture movements
  in larval zebrafish](https://pmc.ncbi.nlm.nih.gov/articles/PMC4074221/).
- Jouary et al., 2024, [Megabouts: a flexible pipeline for zebrafish
  locomotion analysis](https://elifesciences.org/reviewed-preprints/107859).
