# Composable Stimulus Analysis and Plot Recipe Design
<!-- contract-meta
status: accepted-design
last_updated: 2026-08-17
-->

Purpose: define how Palette should expose generic analyses for every compatible
stimulus step, add protocol-specific analyses without forking the generic
system, and compose exact stimulus states into comparisons and figures.

Generic position, heading, speed, and angular-motion metrics must also bind an
explicit observation provider. The provider architecture and migration plan are
defined in
[`position_body_frame_and_motion_provider_design.md`](position_body_frame_and_motion_provider_design.md).
Its exact logical array and dtype requirements are defined in
[`subject_position_storage_contract_v1.md`](subject_position_storage_contract_v1.md).
The accepted first implementation slice for generic trajectory, occupancy,
contrast, cohort, and plot-recipe work is tracked in
[`provider_aware_spatial_analytics_implementation_checklist_2026-08-18.md`](provider_aware_spatial_analytics_implementation_checklist_2026-08-18.md).

This document records the design direction accepted during the 2026-08-14
GoodBatBadBat positional-occupancy campaign. It is an implementation plan, not
an implemented persistence contract. Existing immutable runs remain unchanged.

## Decisions

1. Every canonical stimulus step is a first-class state. `SOLID_BLACK`, for
   example, is not inherently a pre-period or baseline.
2. Analysis roles such as `baseline`, `treatment`, and `control` belong to a
   saved composition. They must not be inferred from stimulus mode or order.
3. A compatible step receives generic analysis offers such as occupancy,
   speed, and distance to the arena boundary.
4. Protocol capabilities add specialized analyses and optional annotation
   providers. They do not replace the generic offers.
5. Exact step and interval selections compile into immutable frame sets before
   a metric or contrast is computed.
6. Scientific metrics and scientifically interpreted contrasts are
   materialized products. Panel layout, labels, display scaling, annotations,
   and exports are versioned render recipes over those products.
7. Canonical analysis Zarrs are the source of truth. SQLite is a rebuildable
   discovery projection and external PNG directories are immutable exports.
8. The existing `analysis/stimulus_epoch_runs` v2 contract remains the
   authority for its flat, contiguous, non-overlapping windows. Composable
   selections require a new versioned surface rather than changing v2 in
   place.
9. Position and body frame are separately selected scientific surfaces.
   Occupancy, trajectory, and linear motion bind one exact position provider;
   heading and angular motion bind one exact body-frame/heading provider.
10. Detection-centroid, keypoint-derived, and mask-derived position methods
    are distinct offers. Analyses record the selected provider and cannot
    silently fall back between them.

## Conceptual Model

```text
canonical recording timeline
└── atomic stimulus steps and interval annotations
    └── saved selection expressions
        └── exact resolved frame sets
            ├── generic metric products
            ├── protocol-specific metric products
            └── scientific contrast products
                └── render recipes
                    ├── generic panels
                    ├── protocol annotation providers
                    └── durable and external artifacts
```

The layers have separate responsibilities.

### Atomic stimulus state

`analysis/stimulus_runs/<run>/steps/step_<index>` remains the authority for an
authored or observed state. An atomic step reference must resolve to:

- recording and exact selected stimulus run;
- step index, path, mode, name, and metadata digest;
- half-open source interval `[start_frame, end_frame)`;
- source-camera timeline identity;
- any exact coordinate or presentation authority needed by a consumer.

Transition intervals and protocol context, such as `chaser_present`, may
overlap atomic steps. They are interval annotations, not competing definitions
of the stimulus timeline.

### Saved selection and resolved frame set

A selection is an immutable expression over atomic steps and compatible
interval annotations. The first version should support:

- exact step references;
- predicates that are persisted together with their concrete resolution;
- directional leading and trailing trims with units and rounding policy;
- ordered `union`, `intersection`, and `difference` operations;
- `pool_intervals` or `keep_occurrences` aggregation;
- named membership roles such as `baseline` or `treatment`.

The materialized result is an ordered, de-duplicated set of half-open frame
intervals. It preserves each source interval and role even when the metric
pools their samples.

A selector such as `mode == SOLID_BLACK` is not enough for durable science.
The saved artifact must also list the exact matching step references, effective
frame bounds, exclusions, and source digests used in that run.

### Metric product

A metric request binds one resolved frame set to a versioned computation. A
spatial occupancy product must declare at least:

- sample unit, such as every detection row or one selected detection per frame;
- exact position-provider ID, run, estimator digest, and validity policy;
- exposure denominator and missing-detection policy;
- coordinate frame and spatial grid;
- smoothing and scientific normalization;
- exact detection, geometry, stimulus, and frame-set lineage;
- per-occurrence and pooled exposure counts when intervals are pooled.

Scientific normalization is distinct from display scaling. Per-panel maximum
normalization may be useful for displaying spatial shape, but it is not an
effect-size measure and must not enter a scientific contrast.

### Contrast product

Materialize a contrast when it produces a reusable or scientifically
interpreted result: pooled steps, baseline subtraction, ratio or log-ratio,
standardized difference, uncertainty, or a statistical endpoint.

A contrast preserves named arms and every contributing step:

```yaml
contrast_id: chaser_minus_black_baseline
operation: difference
metric_id: detection_centroid_occupancy_per_exposure_frame.v1
arms:
  baseline:
    selection: black_before
  treatment:
    selection: chaser_presentation
pairing_policy: recording_within_subject
```

It must fail closed if the arms disagree on metric identity, sample policy,
denominator, grid, coordinate frame, source recording, or other required
scientific semantics.

### Plot recipe and annotation provider

A plot recipe is presentation metadata over exact metric or contrast products.
It owns:

- panel and facet bindings and order;
- labels and captions;
- shared or per-panel display scale;
- colormap, origin, renderer, style, and output parameters;
- requested annotation providers and whether each is required;
- exact source product paths and digests;
- analysis and render signatures.

An annotation provider receives a resolved frame set and returns display
entities plus provenance. It must not change metric values or membership.

For the current chaser campaign,
`chaser_positions_by_behavior.v1` would resolve exact chaser rows, role
vocabulary, protocol profile, post-transition settling, and the persisted
arena-to-source-camera transform. Marker fill uses the experimental protocol
color while marker shape and direct text encode behavior. This provider can
decorate a generic occupancy heatmap without creating a second definition of
occupancy.

## Generic and Protocol-Specific Plot Offers

Plot availability should be capability-driven.

```text
step facts
├── exact interval
├── refined subject trajectory
└── registered arena geometry
    => occupancy, speed, path, and boundary-distance offers

protocol facts
├── chaser states
├── resolved behavior roles
└── valid presentation-to-camera transform
    => chaser-distance analyses and chaser annotation offers
```

Each step can therefore advertise:

- generic plots whose declared requirements are satisfied;
- specialized plots contributed by protocol providers;
- recording-level compositions in which that step is a named member.

A composed plot is discoverable from every source step. A UI showing a black
step used as a baseline should also show the corresponding baseline-versus-
chaser contrast and label that step's `baseline` membership.

Missing optional annotations do not invalidate the generic plot. A recipe that
declares an annotation required must block rather than silently render without
it.

## GoodBatBadBat Example

The recording name and campaign use `goodbatbadbat`; some existing analysis
profiles and modules retain the historical `GoodCopBadCop` name. The generic
model must not depend on either label.

Suppose canonical metadata resolves these steps:

```yaml
steps:
  - {step_index: 0, mode: SOLID_BLACK}
  - {step_index: 1, mode: CHASER_PRESENTATION}
  - {step_index: 2, mode: SOLID_BLACK}
```

Saved selections could be:

```yaml
selection_sets:
  black_before:
    members: [{step_index: 0}]
    trim: {leading_s: 10}
    analysis_role: baseline
  chaser:
    members: [{step_index: 1}]
    trim: {leading_s: 2}
    analysis_role: treatment
  black_after:
    members: [{step_index: 2}]
    trim: {leading_s: 10}
  all_black:
    operation: union
    operands: [black_before, black_after]
    aggregation: keep_occurrences
```

Available outputs then include:

- generic occupancy for every individual step;
- a generic three-panel occupancy recipe;
- a materialized `chaser - black_before` occupancy contrast;
- a protocol-specific chaser-distance metric;
- the generic occupancy panels decorated with chaser behavior annotations.

The existing pre/post figure becomes one compatibility recipe assembled from
generic occupancy products plus the chaser annotation provider. It is not the
underlying scientific schema.

## Proposed Persistence Surfaces

Exact names remain an implementation decision, but the responsibilities should
be distinct:

```text
analysis/stimulus_segment_runs/<segment_run>/
  atomic_steps/
  interval_annotations/
  selection_sets/<selection_id>/
    requested_spec_json
    resolved_intervals/
    source_members/

analysis/detection_occupancy_runs/<occupancy_run>/
  ... scientific occupancy arrays and provenance ...
  plot_artifact_runs/<artifact_attempt>/
    recipe_json
    visualizations/
      snapshot_png/
      interactive_spec/

analysis/detection_occupancy_contrast_runs/<contrast_run>/
  ... authoritative derived arrays and exact source references ...
  plot_artifact_runs/<artifact_attempt>/
```

The segment surface compiles from canonical stimulus steps; it does not replace
their authority. Existing `stimulus_epoch_runs` remain readable and can be
adapted into simple selections.

A single occupancy run may support many recipes when they only select already
materialized windows and perform display-only operations. A derived contrast
run stores only new result arrays and provenance; it references source
occupancy arrays by exact path and digest rather than copying their full stacks.

Plot artifact attempts are immutable and cannot change the parent scientific
selector. Failed attempts remain selector-ineligible tombstones and retries use
new names.

## Publication and Export Lifecycle

1. Resolve complete, eligible scientific sources to exact paths and digests.
2. Persist the requested selection and its concrete resolved frame set.
3. Materialize and validate metric or contrast arrays in staging.
4. Create a new immutable run or plot-artifact attempt as selector-ineligible.
5. Write arrays, recipe/spec, PNG, manifests, hashes, and provenance.
6. Validate source binding, array inventory, media hashes, and completion
   contracts.
7. Activate a scientific selector only through the guarded final commit for
   that scientific run family.
8. Consolidate root metadata as the final published visibility step and verify
   the intended generation through consolidated reads.

An `/nvme1` figure campaign is an immutable export generation. Its manifest is
written last and records the source scientific signatures, recipe digest,
Palette commit, complete file inventory, and hashes. It is convenient and
auditable, but never the scientific authority.

## Discovery and Registry Projection

Completed scientific and plot-artifact runs should expose versioned
`analysis_offer` and `plot_offer` descriptors, or participate in a recording-
local catalog synthesized only from those immutable descriptors.

An analysis offer records:

- analysis and plot class IDs;
- exact run and artifact references;
- every source step/window and its composition role;
- source, selection, analysis, and render digests;
- independent readiness dimensions.

Recommended readiness dimensions are:

| Dimension | Example states |
| --- | --- |
| scientific | `ready`, `blocked_missing_source`, `stale_lineage`, `invalid_contract` |
| render | `ready`, `needs_render`, `render_contract_mismatch` |
| review | `not_required`, `pending`, `approved`, `rejected` |
| annotation | `not_requested`, `available`, `unavailable_optional`, `invalid_required` |
| registry projection | `current`, `stale`, `absent` |

The registry should project, not author, these offers. A minimal projection
needs rows for analysis offers, their source-step memberships, and plot offers.
Every row links back to a Zarr descriptor and digest.

Parallel workers write Zarr products and immutable receipts only. One dependent
finalizer validates successful receipts and replaces one recording's projected
rows inside a short SQLite transaction. A failed refresh leaves the previous
projection intact; retry is idempotent. A UI may use SQLite for cohort listing,
then verify the selected recording's Zarr catalog before presenting it.

## Current Implementation Gaps

- `stimulus_epoch_runs` records reusable windows, but its strict v2 shape
  cannot represent non-contiguous sets or overlapping annotations.
- Current profile resolution is centered on `pre_event`, `training_event`, and
  `post_event` aliases instead of arbitrary canonical step selections.
- Current occupancy, trajectory, and speed consumers generally assume
  detection/crop centroids instead of resolving one explicit position-provider
  contract.
- Current track kinematics can combine detection/crop positions with keypoint
  heading, but those source families are tightly coded rather than selected as
  independent typed position and body-frame providers.
- `detection_occupancy_runs` imports detection parsing and heatmap computation
  from `plot_detection_epoch_heatmaps`; scientific computation should move to
  a metric module consumed by both writer and renderer.
- Current occupancy maps count detection rows while some zone summaries select
  one detection per frame. A new metric version must use and record one
  consistent sample-unit policy.
- Current per-panel maximum normalization is display-oriented and unsuitable
  for cross-epoch effect comparisons.
- The current chaser overlay is restricted to pre/post labels and should become
  a generic frame-set annotation provider.
- Occupancy v1 does not bind all consumed arrays, geometry, transforms, and
  annotation rows by value digest.
- The direct occupancy writer and generic plot helpers do not yet implement the
  immutable attempt and final consolidated-publication lifecycle above.
- Reporting has a static plot catalog but no dynamic occupancy/composition
  offers or source-step reverse membership.
- The existing registry step-status key cannot represent many analysis runs,
  plots, or per-composition memberships.

### Phase 4A foundation status (2026-08-17)

The first shared contract layer is implemented without publishing a metric or
activating a selector:

- `provider_analysis_offers` defines immutable provider identities,
  independent position/body-frame/motion requirements, exact temporal
  selection identities, and selector-ineligible analysis offers.
- `provider_track_motion_source_handle` strictly reads the flat
  `analysis/track_kinematics_runs/provider/<run>` layout while preserving
  independent source, sample, transition, and reason-code arrays.
- `resolved_epoch_selection` adapts one explicit maintained stimulus-epoch v2
  run into digest-bound half-open intervals. Chronological non-overlap is
  required; legitimate gaps are preserved rather than filled.
- `provider_analysis_bindings` derives readiness from verified inputs. Missing
  recording identity blocks before missing timing, known cross-recording
  composition fails, and only exact recording/timing authority can become
  `ready`.
- `provider_recording_timing_authority` revalidates the canonical recording,
  source-video metadata, selected acquisition frame clock, complete frame
  domain, and direct/consolidated metadata before minting one shared digest.
  Existing immutable position, body-frame, and provider-motion runs can bind
  that digest at read time when their source indices and declared FPS agree;
  no source run is rewritten.

The numerical policy remains
`nominal_fps_bound_to_acquisition_frame_domain.v1`. A 2026-08-17 read-only
audit of all 84 canonical GoodBatBadBat archives (14,202,392 frames) found
100% valid, strictly increasing camera timestamps, no duplicate/decreasing
deltas, a maximum recording p99 interval error of 25 ns from the 10,000,000 ns
nominal interval, and full-recording span drift within +/-26 ns. That evidence
does not justify copying timestamp/delta arrays or introducing variable-delta
motion. Missing acquisition-clock authority remains an explicit legacy block,
and any future variable-delta policy must be a new versioned computation. The
implemented no-write authority loader subsequently bound all 84 archives with
zero failures.

## Composition Safety Rules

1. Combine intervals only from the same recording timeline and exact stimulus
   source unless a separate cross-recording analysis contract applies.
2. Persist both requested and resolved selections.
3. Make trims directional, unit-bearing, and explicit about frame rounding.
4. De-duplicate overlapping frames and preserve the requested overlap policy.
5. Preserve occurrence identity and both pooled and per-occurrence exposure.
6. Never infer baseline/treatment roles from stimulus mode, step index, or a
   `pre`/`post` label.
7. Require scientific-semantic compatibility before contrasts.
8. Keep display normalization out of scientific contrast calculations.
9. Materialize scientifically interpreted aggregation or contrast results.
10. Keep annotations independent of metric values and selection membership.
11. Fail closed on stale source digests or incomplete/ineligible inputs.
12. Do not make SQLite or an external image directory the analysis authority.

## Implementation checkpoint (2026-08-18)

The provider-aware foundation now supports exact recording/timing bindings,
provider-motion source handles, resolved epoch selections, immutable
provider-epoch behavior summaries, cohort exports/plots, and a
provider-position chaser-distance canary. The recording Marimo explorer can
inspect the selector-ineligible chaser candidate, semantic chaser roles,
egocentric bearing, and bout-response rows without changing scientific or
registry state.

This is a narrow canary implementation, not the complete composable analytics
system described here. Generic occupancy/contrast schemas, arbitrary
selection algebra, plot recipes, recording-local discovery, and provider
promotion remain open. No production provider selector was changed.

## Implementation Checklist

### Phase 1: contracts and in-memory composition

- [ ] Freeze current GoodBatBadBat/GoodCopBadCop figures as compatibility
      fixtures.
- [ ] Define versioned atomic-step reference, interval annotation, selection
      expression, resolved frame-set, analysis-offer, and plot-offer schemas.
- [ ] Implement a pure selection compiler for exact step refs, trimming,
      union, intersection, difference, occurrence handling, and canonical
      digests.
- [x] Add a compatibility adapter from existing stimulus epoch windows.
- [x] Define the provider-analysis capability/offer foundation used by the
      implemented motion, epoch-behavior, and chaser-candidate canaries.
- [ ] Generalize that capability registry to all generic metrics, protocol
      providers, plot classes, and composed selections.
- [x] Define typed position-provider and body-frame-provider requirements so
      generic metrics do not hardcode detection centroids or keypoint headings.
- [x] Bind provider and temporal-selection identities to one exact validated
      recording/source-video/acquisition-clock authority digest.
- [x] Bind implemented provider-motion, epoch-behavior, cohort, and
      chaser-candidate products to exact provider identities and digests.
- [ ] Extend exact provider binding to every generic position, speed, heading,
      and angular-motion offer.

### Phase 2: scientific products

- [ ] Move occupancy computation out of the visualization module.
- [ ] Define occupancy v2 with one explicit sample-unit policy and scientific
      exposure-normalized arrays.
- [ ] Record exact source-step membership and all detection, geometry,
      coordinate, transform, and configuration digests.
- [ ] Add a versioned occupancy contrast run with strict compatibility checks.
- [ ] Preserve existing occupancy v1 readers and immutable runs.

### Phase 3: recipes and durable visualizations

- [ ] Define canonical plot-recipe JSON with separate analysis and render
      signatures.
- [ ] Generalize chaser overlays into a resolved-frame-set annotation provider.
- [ ] Recreate the current pre/post figure as a compatibility recipe.
- [ ] Publish immutable plot-artifact attempts beneath exact metric or contrast
      runs using existing visualization artifact helpers.
- [ ] Validate PNG/spec/media hashes and consolidated metadata before
      publication is complete.
- [ ] Define manifest-last immutable external export generations.

### Phase 4: discovery, registry, and UI

- [ ] Build a recording-local Zarr analysis catalog from completed eligible
      offer descriptors.
- [ ] Return generic, protocol-specific, and composed offers for every source
      step they reference.
- [ ] Expose independent scientific, render, review, annotation, and registry
      status dimensions.
- [ ] Add rebuildable SQLite projection tables for offers, source memberships,
      and plot artifacts.
- [ ] Refresh projections through one receipt-driven serial finalizer, never
      per-worker SQLite writes.
- [x] Expose the selector-ineligible provider chaser-distance candidate,
      semantic role labels, bearing, and bout-response rows in the
      recording-level Marimo canary.
- [ ] Update the review/reporting UI to browse available analyses and plot
      classes by recording, step, composition, and readiness.

### Phase 5: migration and validation

- [ ] Synthesize explicit legacy/unbound offers for compatible existing runs.
- [ ] Add mixed `SOLID_BLACK -> treatment -> SOLID_BLACK` fixtures.
- [ ] Test exact interval boundaries, trims, set algebra, empty sets, overlap,
      and duplicate-frame prevention.
- [ ] Test stable canonical digests and stale-lineage rejection.
- [ ] Test contrast rejection for mismatched source, grid, coordinate, sample,
      denominator, and normalization contracts.
- [ ] Test optional versus required annotation behavior.
- [ ] Test one scientific run serving multiple recipes without mutation.
- [ ] Test immutable retries, media tampering, consolidation, and external
      export manifest validation.
- [ ] Test direct-Zarr discovery against the SQLite projection and atomic,
      idempotent registry refresh.

## Open Decisions Before Implementation

1. Final namespace: `stimulus_segment_runs`, `stimulus_selection_runs`, or a
   smaller selection component under another run family. The required contract
   behavior is settled; the name is not.
2. The exact first-version contrast algebra, weighting, overlap, and pooling
   policies. Start narrow rather than accepting arbitrary expressions.
3. Whether behavior roles represent protocol intent, independently reviewed
   observed behavior, or two separately named vocabularies.
4. Whether plot artifact attempts need a local `latest_complete` convenience
   selector. They must never alter the parent scientific selector.
5. Which generic plot classes join occupancy in the first implementation.
