# Validated recording-behavior composition design

<!-- decision-meta
status: implementation-active
created: 2026-08-31
last_updated: 2026-08-31
scope: recording-local behavioral source authority, exact cross-analysis
  composition, visualization adapters, and reusable core-plus-stimulus views
related: docs/behavior_event_analysis_design_decision.md,
  docs/marimo_explorer_architecture.md,
  docs/diagnostics/chaser_exact_full_gap_closure_implementation_checklist_2026-08-30.md
-->

## Question this design answers

Palette should let any number of visualizations consume validated behavioral
datasets without inventing a new scientific authority for every plot. The
question is therefore not merely how to add a full chaser dashboard. It is:

> What is the smallest exact, reusable interface that identifies compatible
> validated behavioral data, and when does a new view require only a renderer
> versus a new scientific successor?

The desired boundary is one recording-level composition of exact source
authorities and independently published scientific products. Marimo, static
publication, reporting, and export adapters may then request bounded
projections from that composition. They do not choose sources independently,
reinterpret missingness, or become upstream scientific authorities.

## Proposed decision

1. Define a **validated recording-behavior bundle** as a small immutable,
   digest-bound selection and compatibility envelope over existing analysis
   runs. It is an interface and join contract, not a copied all-in-one dataset.
2. Keep dense scientific arrays in their existing recording Zarr run families.
   Keep independently derived products in their own lineages and receipts.
3. Bind one exact source for each selected capability and record unavailable,
   inapplicable, invalid, and review-required capabilities explicitly.
4. Let visualizations declare required capabilities, row grains, fields, and
   display parameters against the bundle. A pure view does not require a new
   scientific successor.
5. Publish a new successor only when a transformation creates reusable
   scientific meaning: membership, classification, coordinate normalization,
   missing-data or censoring policy, null model, scientific bins, denominator,
   or an inferentially reused summary.
6. Treat a composition receipt as proof of one exact compatible source choice.
   It is not a selector, cache authority, production authority, or substitute
   for child validation.
7. Treat a plot receipt as publication provenance for a rendered artifact. It
   is not the authority for the numeric values shown in the plot.

This design generalizes the useful property already implemented by the exact
chaser projection receipt: independently validated children remain independent,
while one composition record prevents a consumer from mixing whichever runs
happen to be discoverable at read time.

## Scoped vocabulary

"Authority" must always be qualified. The following terms are distinct:

| Term | Meaning | What it does not mean |
|---|---|---|
| Canonical run | A run conforms to the accepted schema and scientific definition for its data family | It is not necessarily the run selected for this workflow |
| Workflow source authority | The exact canonical or explicitly admitted run chosen to supply one data surface in this bundle | It is not automatically a global production selector |
| Scientific successor | An immutable derived dataset that introduces contracted scientific meaning | It is not merely a differently styled plot |
| Validation receipt | Evidence that a named run, manifest, arrays, and metadata passed a declared validation contract | It does not choose among multiple scientifically eligible runs unless selection is part of its contract |
| Composition receipt | Exact paths and digests for independently validated children plus their compatibility policy | It does not replace child receipts or contain their numerical payloads |
| Visualization projection | A bounded, read-only selection of fields and rows for one renderer | It cannot define canonical events, validity, sources, or scientific denominators |
| Plot receipt | Source composition, consumed fields, display recipe, and output-byte hashes for an exported artifact | It is not an input authority for later analysis |
| Production authority | A separately promoted selector-visible state for operational reuse | The current exact chaser candidates and composition receipts explicitly do not claim this |

A source can be authoritative for one quantity and inappropriate for another.
For example, a detection centroid may be a first-class fish-position provider
while being incapable of supplying anatomical body orientation. A subject mask
may be authoritative for component pixels while not being the chosen fish
position provider.

## Existing architectural evidence

The accepted behavioral-event design already states that dense frame-aligned
Zarr arrays remain authoritative and that Zarr and Parquet are source adapters,
not automatically separate plot implementations. It also specifies canonical
stimulus-independent bout facts joined to stimulus membership instead of copied
protocol-specific bouts.

The current exact chaser projection receipt has the correct composition
property. It supports closed versioned profiles; its fullest schema-v8 profile
binds these exact children:

- protocol-semantic selection;
- keypoint and detection radial/near-field products;
- controller trials;
- generalized bout response;
- escape/freeze;
- spatial occupancy;
- gaze (required by the v8 profile and absent from no-gaze profiles such as
  epoch-plus-alignment v7);
- semantic-v2 epoch behavior;
- body alignment by distance; and
- keypoint and detection chaser-relative frame children.

It creates no new scientific arrays. It freezes exact child receipt paths,
receipt digests, run paths, manifests, and payload digests for a closed
interactive source choice.

The current Core Behavior explorer is weaker as a cross-capability composition:

- it starts from one concrete track-kinematics run and track;
- it resolves swim bouts that are lineage-compatible with that track;
- it discovers eye-angle and tail-kinematics capabilities independently; and
- an omitted explicit eye or tail run may still be resolved through a latest
  default inside the capability-specific catalog.

That is useful metadata-first browsing, but it is not yet one sealed
Core-Behavior-plus-Chaser dataset. Merely showing both provider menus for the
same recording does not prove that they use the same track, frame rowset,
timing, coordinate system, body source, or optional eye/tail lineage.

The existing full-chaser-profile successor is also useful but different. It
binds profile applicability, module state, execution waves, and exact module
product digests. It does not currently define the generic Core Behavior source
interface, and it is not yet invoked by the maintained cohort runner or exposed
as a Marimo readiness route.

## Current GoodBatBadBat source map

The current exact cohort runner freezes the following source classes. These are
workflow choices for this workload, not universal precedence rules for every
future protocol.

| Data domain | Current exact source | Data supplied | Scope and limitation |
|---|---|---|---|
| Recording and row identity | acquisition-frame and track-sample row-axis evidence | recording ID, acquisition frame ID, track sample ID, optional acquisition timestamp, row-axis digest | Required for all exact joins; same-length arrays are insufficient |
| Fish position: keypoint | exact keypoint-triad provider and its immutable proxy/binding | fish XY, validity, source-row identity | Current motion/body-capable provider; one explicit position choice, not universal truth |
| Fish position: detection | exact detection/bounding-box-centroid provider and its immutable proxy/binding | fish XY, validity, source-row identity | First-class alternative for distance and spatial products; does not supply anatomical heading |
| Anatomical body frame | body-frame run bound by the selected keypoint-provider motion manifest | body origin, forward axis, anatomical-left axis, heading, validity, source row | No motion-heading, mask-axis, or detection fallback |
| Motion | exact provider track-kinematics run selected from a supported frozen pair | calibrated position, speed levels, path, heading/turning, validity and transition state | Current workload uses keypoint-triad provider motion; published summaries reject raw speed |
| Canonical bouts | exact swim-bout run bound to the selected provider motion and track | bout identity, onset/offset/peak, duration, path, validity, inter-bout intervals | Bouts remain stimulus-independent facts |
| Stimulus semantics | canonical stimulus evidence plus protocol-semantic selection successor | exact pre/training/post roles, step identity/hash, interval bounds and membership | No protocol-name or trial-count inference |
| Controller trials | producer-logged active-trial successor | trial identity, trigger alignment, exact active-row membership, retained nonmember gaps | Gaps are evidence but are not trial members |
| Chaser observations | exact stimulus/chaser source projected onto acquisition rows | chaser identity, position, occurrence, active state, trial, behavior role | Role is independent of display color and array order |
| Geometry and scale | reviewed arena selection and physical calibration/coordinate-frame record | arena circle, in-arena meaning, pixel-to-mm scale, coordinate convention | Required for physical distance, occupancy, wall, and geometric-null products |
| Subject pixels and shape | refined subject-mask bundle and subject-shape successor | dense component masks, body/eye geometry, tail spline samples | Optional for most chaser distance products; quality is scoped to descendants |
| Eye orientation | exact eye-angle run plus explicit accepted biological convention receipt | eye-angle variants, convergence, validity, directed gaze meaning | Unavailable when review is absent or rejected; no selector or convention inference |
| Tail behavior | subject-shape and tail-kinematics successors | body-frame tail angles, deflection, curvature, validity | Not yet bound into the exact chaser composition |

For historical recordings, immutable acquisition/input-provenance proxy runs
may mediate a source binding. New recordings should use direct frame-bound
acquisition identity. The logical interface remains the same: the bundle records
the exact admitted source and proof rather than making visualizers know which
historical repair path produced it.

## Data-grain model

A composable interface cannot be one untyped wide table. Behavioral facts have
different row grains:

| Grain | Stable keys | Representative data |
|---|---|---|
| recording | recording/session identity | protocol, arena, calibration, recording-level applicability |
| frame | recording + acquisition frame + track sample | fish position, speed, heading, body frame, eyes, tail scalars, validity |
| frame x chaser | frame keys + chaser identity | chaser position/role/occurrence, distance, bearing, relative velocity |
| bout | recording + track + canonical bout ID | bout bounds, duration, path, kinematics, validity |
| bout x chaser/context | bout ID + chaser/epoch/trial identity | onset distance/bearing, separation response, response membership |
| trial or event | contracted event identity | trigger, response occurrence, latency, censoring, aligned samples |
| semantic epoch | semantic step/interval identity | persisted motion, bout, occupancy, and support summaries |
| scientific bin | product identity + epoch/chaser/bin index | fixed histogram or distance-bin summaries and denominators |

Composition joins these grains through exact association surfaces. It does not
duplicate canonical bout facts for each protocol or broadcast tail tensors into
every event row. Every association retains the source identities needed to
return to the authoritative dense arrays.

## Authority and derivation graph

```text
recording identity + acquisition-frame/timing authority
                         |
          +--------------+------------------+
          |              |                  |
   fish providers   stimulus/chaser    arena/scale
  keypoint+detection   semantics         authority
          |              |                  |
          +------ exact frame joins --------+
                         |
               chaser-relative rows
                         |
       +-----------------+------------------+
       |                 |                  |
 provider motion     semantic epochs   controller trials
       |                                    |
 canonical bouts ---------------------------+
       |                 |                  |
       +---- bout response/escape ---------+

refined subject masks -> subject shape -> eye angles -> reviewed convention
                                  |              |
                              tail data       gaze successor

independent scientific children + exact compatibility evidence
                         |
          validated recording-behavior bundle
                         |
       +-----------------+-------------------+
       |                 |                   |
     Marimo          static figures     Parquet/report adapters
```

The bundle sits below all renderers. It does not sit above them as another
rendered artifact, and it does not replace the independently governed children.

## Same source data versus a new scientific product

| Requested view | Relationship to existing data | New scientific successor? |
|---|---|---|
| Whole-session speed trace with another color or time window | Direct projection of selected motion arrays | No |
| Speed summarized by exact semantic epoch | Same motion plus semantic membership and a contracted aggregation/denominator | Use the existing epoch-behavior successor |
| General bout distribution | Direct projection or descriptive summary of canonical bout rows | No, when the source contract permits that summary |
| Bout response by chaser distance | Canonical bouts plus exact onset-distance, chaser, epoch, and trial association | Use the generalized bout-response successor |
| Distance trace and body-bearing polar view | Different projections of the same exact relative rows | No |
| Bearing x distance point cloud | Same relative distance/body-bearing rows with bounded display sampling | No |
| Fish trajectory overlay | Direct position and chaser projection with exact epoch filtering | No |
| Occupancy heatmap | Position rows plus scientific grid, arena mask, validity, normalization, and denominator | Use the spatial-occupancy successor for the contracted result |
| Near-field dwell or geometric-null comparison | Relative positions plus ring thresholds, arena correction, denominators, and null policy | Use the radial/near-field successor |
| Raw eye-angle trace | Direct projection of one exact eye-angle run | No |
| Eye-to-chaser gaze error and sustained lock | Eye angles plus body bearing, biological convention, threshold, duration, and validity policy | Use the gaze successor |
| Tail trace in ordinary time | Direct projection of exact tail-kinematics arrays | No |
| Tail response around ring entry or escape | Tail arrays plus a contracted event alignment, censoring, and sampling policy | A reusable aligned-event successor is required |
| Static PNG and interactive Plotly rendering of the same normalized data | Two renderers over one data projection | No new scientific successor; each exported artifact may have a plot receipt |

Existing child contracts remain controlling. A viewer must not rebin a product
whose contract explicitly prohibits viewer rebinning. More generally, a
transient descriptive transformation may be permitted only when its source
contract allows it, its parameters are visible and exportable, and it is not
presented as the canonical persisted statistic.

## Decision test for future visualizations

Before adding a successor, answer these questions in order:

1. **Are all plotted numeric fields already present in one compatible validated
   bundle?** If no, identify the missing scientific surface.
2. **Does the transformation only change layout, style, bounded sampling,
   filtering over persisted membership, or hover/detail presentation?** If yes,
   it belongs in a renderer or projection.
3. **Does it decide membership, identity, validity, interpolation, censoring,
   coordinate frame, threshold, null model, scientific binning, weighting, or
   denominator?** If yes, it is scientific logic and normally belongs in a
   versioned successor.
4. **Will the derived values be reused in cohort export, statistical inference,
   training, or another analysis?** If yes, persist and seal them even when a
   notebook could recompute them cheaply.
5. **Is it an exploratory display-only summary from retained row evidence?** If
   permitted by the source contract, it may remain transient but must record
   its parameters and carry an exploratory/non-authoritative label.

This test keeps ordinary visualization development inexpensive without moving
scientific decisions into notebooks.

## Proposed logical bundle interface

The physical carrier remains an implementation decision, but its logical
record should contain at least:

```text
validated_recording_behavior_bundle
  schema and method identity
  recording/session/raw-acquisition identity
  row-axis and timing identity
  source bindings
    fish_position[keypoint]
    fish_position[detection]
    anatomical_body_frame
    provider_motion
    canonical_swim_bouts
    semantic_epochs
    controller_trials
    chaser_observations
    reviewed_arena_and_scale
    optional subject_shape
    optional eye_angles_and_convention
    optional tail_kinematics
  scientific child bindings
    chaser_relative[keypoint]
    chaser_relative[detection]
    epoch_behavior
    bout_response
    escape_freeze
    radial_near_field[keypoint,detection]
    spatial_occupancy
    optional gaze
    optional body_alignment
  compatibility proofs
  capability states and typed reasons
  child receipt bindings
  software and validation policy
  bundle digest
```

Each binding must include an exact immutable path, schema/version, manifest or
payload digest, recording identity, and the timing/row/coordinate/provider
identities applicable to that surface. A selector name such as `latest`,
`authoritative`, or `default` is not an acceptable frozen binding.

The bundle should not copy the bound arrays. Compact association products may
remain independently published children and be referenced by digest. Whether
the bundle record is stored in a small recording-local immutable run, an
external durable receipt, or both must not change its logical validation.

## Visualization requirement declarations

Every visualization should declare a closed requirement set rather than know
how to discover sources. For example:

```text
body_bearing_polar
  grain: frame_x_chaser
  requires:
    chaser_relative[keypoint].body_bearing_deg
    chaser_relative[keypoint].body_bearing_valid
    semantic_epochs.membership
    chaser_observations.behavior_role

bout_distance_response
  grain: bout_x_chaser_context
  requires:
    canonical_swim_bouts.bout_id
    generalized_bout_response.distance_at_onset_mm
    generalized_bout_response.validity
    semantic_epochs.identity

core_tail_trace
  grain: frame
  requires:
    tail_kinematics.time_or_frame_identity
    tail_kinematics.scalar_channels
    tail_kinematics.validity
```

The bundle loader validates source compatibility once. The analysis adapter
loads only the declared fields and bounded rows. Marimo, a static renderer, and
a Zarr/Parquet adapter should share the same normalized render-data contract
when they answer the same scientific question.

## Current batch disposition

The 2026-08-31 historical gaze-prerequisite cohort demonstrated that the gaze
workflow behaves correctly when the subject masks are anatomically adequate.
Pigmentation outside the original segmentation model's training distribution
causes poor masks in part of this batch. Until retraining and a new review:

- do not issue an accepted gaze-convention bundle for this cohort;
- do not attach its gaze candidate as an available bundle capability;
- record gaze as `review_required`, or as `unavailable` with the typed reason
  `upstream_segmentation_quality`, rather than invalidating the complete
  recording behavior bundle; and
- retain the candidate and review panels as validation/retraining evidence,
  not as scientific cohort data.

Distance, position, motion, bout, trial, escape/freeze, occupancy, and
keypoint-body-frame children do not depend on those masks and remain separately
valid when their own receipts pass.

## Safety and invalidation rules

- [x] Compatibility is proved by exact recording, row, timing, coordinate,
      track, provider, and semantic identities as applicable; never by shape or
      row count alone.
- [x] Every selected capability names one exact immutable run and digest. No
      read-time selector resolution occurs after the bundle is frozen.
- [x] Keypoint and detection position providers retain separate identities and
      are never silently averaged or substituted.
- [x] Detection remains a first-class position provider but cannot satisfy an
      anatomical-body-frame requirement.
- [x] Optional capability failure is scoped. Mask or eye review invalidates
      mask/shape/eye/gaze descendants, not unrelated distance or motion data.
- [ ] Association tables add context to canonical facts; they do not replace or
      duplicate the canonical source rows.
- [x] Missing rows remain missing. Composition does not interpolate, use a
      nearest row, or attach events across inactive trial gaps.
- [x] Independent child receipts remain independently reviewable and reusable.
- [x] The composition record is selector-ineligible until a separate explicit
      production promotion contract exists.
- [ ] A renderer cannot change scientific validity, membership, classification,
      denominators, or contracted bins.
- [ ] Static and interactive artifacts expose the exact bundle digest, consumed
      source fields, display parameters, and renderer/version.
- [ ] Cohort inference retains the experimental-unit hierarchy and never treats
      frame, bout, visit, or trial rows as independent animals.

## Implementation checklist

### Phase 0 — Freeze the boundary

- [x] Confirm the logical name and schema identity for the validated
      recording-behavior bundle.
- [x] Decide its physical carrier for schema v1: an external immutable JSON
      receipt. Recording-local discovery remains a later, separately governed
      carrier decision.
- [x] Confirm that the current chaser-bound keypoint provider motion and its
      exact same-track bouts are the default Core Behavior source for this
      GoodBatBadBat composition.
- [x] Preserve detection centroid as an explicit parallel position capability,
      not a legacy fallback.
- [ ] Decide whether loose metadata-first Core Behavior browsing remains an
      explicitly diagnostic mode after exact bundle routing is available.
- [x] Freeze the capability-state vocabulary, including complete, unavailable,
      inapplicable, invalid, stale, and review-required.

### Phase 1 — Contract and normalized source handles

- [x] Define closed records for source binding, scientific-child binding,
      compatibility proof, capability state, and bundle identity.
- [x] Require exact schema/version, run path, manifest/payload digest,
      recording identity, and relevant row/timing/coordinate/provider digests.
- [x] Define explicit transitive-binding behavior so a child that already seals
      motion or bouts can expose that proof without duplicating it ambiguously.
- [ ] Define a closed row-grain registry and stable key requirements.
- [ ] Define a visualization requirement declaration schema for capabilities,
      arrays/tables, grain, and permitted display transformations.
- [x] Keep the interface independent of Marimo and directly unit-testable.

### Phase 2 — Exact bundle resolver and validator

- [x] Resolve an initial bundle only from explicit task-pinned inputs and
      existing child receipts; do not discover `latest` children.
- [ ] Validate exact recording/raw-acquisition, frame-axis, timing, coordinate,
      scale, and track compatibility across selected sources.
- [x] Verify provider motion and canonical bouts share the exact source track.
- [x] Verify body frame is the exact source bound by the selected motion run.
- [ ] Verify optional eye and tail sources against the bundle row/timing/body
      identities before marking their capabilities available.
- [ ] Admit gaze only with the exact accepted convention receipt and exact eye
      source binding.
- [x] Reuse existing validation receipts and targeted hashes so opening the
      bundle does not deep-rescan unrelated arrays.
- [x] Emit an immutable bundle digest and human-readable, digest-bound
      compatibility proofs.

### Phase 3 — Core Behavior adapter migration

- [x] Add an exact bundle-backed Core Behavior source adapter.
- [ ] Make speed, heading, position, bouts, baseline, eyes, and tail request
      exact bundle capabilities rather than independently choosing runs.
- [x] Remove `latest` eye/tail resolution from the exact bundle route. The
      first slice exposes only exact direct provider-motion views; unimplemented
      bundle-backed bout, baseline, eye, and tail routes fail closed.
- [x] Keep missing optional capabilities visible with typed reasons in the
      exact Core Behavior source-identity panel.
- [x] Retain bounded direct projections; do not pre-load sibling capabilities.
- [x] Verify in focused synthetic tests that existing speed and position
      projections return the identical selected provider-motion values.

### Phase 4 — Chaser composition reuse

- [x] Reference the current exact-chaser projection receipt or its validated
      children instead of reproducing their bindings under new names.
- [x] Make semantic selection, controller, relative-frame, radial, spatial,
      bout-response, escape/freeze, epoch, gaze, and alignment capabilities
      independently addressable.
- [x] Express the current no-gaze batch as a truthful partial capability set.
- [x] Add tail as an optional core capability without claiming a chaser-tail
      response product until an exact event/alignment successor exists.
- [ ] Integrate the existing full-profile applicability envelope as a readiness
      view over the bundle, not as another numerical source.

### Phase 5 — Visualization adapters

- [ ] Move each Core Behavior and exact-chaser route to a closed requirement
      declaration.
- [x] Build the first normalized projection object independent of Marimo: the
      exact provider-motion track projection. Other grains remain pending.
- [ ] Record exact bundle digest, child digests, consumed arrays/tables,
      bounded row selection, and display parameters in figure metadata.
- [ ] Share pure render-data preparation between interactive and static outputs
      where the scientific question and aggregation scope match.
- [ ] Ensure adding a pure visualization requires no Zarr writer, selector, or
      new scientific successor.
- [ ] Keep exploratory display summaries visibly distinct from contracted
      persisted summaries.

### Phase 6 — Associations and portable exports

- [ ] Preserve canonical `swim_bout_events` and `inter_bout_interval_events`
      independently of stimulus family.
- [ ] Define or reuse exact frame-to-epoch, frame-to-trial, bout-to-epoch, and
      bout-to-chaser association surfaces.
- [ ] Export dense frame samples only through bounded purpose-specific adapters;
      dense Zarr remains authoritative.
- [ ] Keep tail samples in a separate normalized-tail-position surface rather
      than duplicating them into every event row.
- [ ] Bind each Parquet export to the exact recording bundle and source digests.
- [ ] Preserve recording/fish/session/acquisition-batch identities needed for
      clustered or hierarchical cohort inference.

### Phase 7 — Fail-closed validation

- [x] Reject same-length but different row identities.
- [ ] Reject cross-recording, cross-camera, cross-track, and mixed-provider
      composition.
- [x] Reject a bout run bound to another motion run or track.
- [x] Reject independently discovered eye/tail runs under the exact route.
- [ ] Reject missing, stale, rejected, or source-mismatched review evidence.
- [x] Reject source substitution after a bundle is frozen, including a
      re-digested bundle envelope; exact routes perform no selector lookup.
- [x] Reject direct/consolidated metadata disagreement for published immutable
      sources.
- [x] Prove scoped invalidation: an eye/mask failure removes gaze without
      removing unrelated distance, bout, or occupancy capabilities.
- [ ] Prove renderer isolation: display changes cannot alter bundle identity or
      scientific child payloads.
- [ ] Prove Zarr and Parquet adapters yield equivalent normalized render data
      for explicitly dual-backend views.

### Phase 8 — Canary and cohort adoption

- [x] Build one read-only bundle plan for the existing GoodBatBadBat smoke
      recording and compare every selected source against the current exact
      projection receipt and cohort task.
- [ ] Materialize only the small composition record after local tests and all
      required CI are green; do not recompute validated children.
- [x] Load direct Core Behavior and exact epoch-behavior routes from the same
      read-only canary bundle; wider route migration remains pending.
- [ ] Compare existing figure data numerically, not only by visual appearance.
- [x] Confirm gaze is absent with the recorded segmentation/review reason for
      the present historical batch.
- [ ] Freeze a cohort task with one explicit bundle identity per recording.
- [x] Keep production selector activation and registry authority as separate,
      later decisions.

## Implementation progress — 2026-08-31

The first read-only slice is implemented on the isolated branch
`agent/palette/validated-behavior-composition-design-20260831`:

- `validated_recording_behavior_bundle.py` defines schema
  `palette.analysis.validated_recording_behavior_bundle` version 1, builds and
  validates the closed composition, validates existing child receipts, and
  emits an external immutable JSON record with a self digest;
- `plan_validated_recording_behavior_bundle.py` is the explicit-input CLI. It
  accepts no selector, registry, or discovery option and requires a typed
  disposition for every absent capability; and
- focused unit tests exercise the complete no-gaze path and reject mismatched
  shared-axis content, mixed motion/bout tracks, wrong body-frame lineage,
  mixed semantic dependencies, selector-named sources, stale bundle digests,
  and re-digested source substitution.

One read-only smoke composition was validated against recording
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` and the existing schema-v7 exact
chaser projection receipt. It resolved 10 source bindings, 11 independently
addressable scientific-child bindings, 17 complete capabilities, and four
explicitly unavailable capabilities. Gaze, eye angles, and subject shape are
unavailable because of `upstream_segmentation_quality`; tail kinematics is
unavailable because it is `not_persisted`. The smoke record is approximately
71 KiB, well below the schema's 256-KiB metadata bound. The final smoke also
proved direct/consolidated equivalence for all 59 provider-motion metadata
nodes (57 arrays and two groups), with declaration digest
`e454bebf92d195b71b54d709a72627116e1f8623d5daeb9fe86cc55ba1f53c0f`.

The final smoke was built and reopened in memory only. An earlier temporary
specimen at
`/tmp/validated_recording_behavior_bundle_canary_uncommitted_20260831.json`
predates the consolidated-metadata binding and is obsolete diagnostic evidence.
Neither is a durable publication, selector eligible, or production authority.
Regenerate the bundle from the eventual exact implementation commit after
required CI is green; do not copy or promote the temporary specimen.

The next read-only adapter slice now routes one bundle across both viewer
families:

- `validated_recording_behavior_source.py` opens and validates one exact
  bundle, exposes typed capability states, and refuses any source or selector
  discovery;
- provider motion is opened through published consolidated metadata and its
  exact manifest. The consumer hashes only the requested payload arrays plus
  `track_ids` and `track_row_offsets`, which prove that the returned rows are
  exactly the bundle-bound track segment;
- `ValidatedCoreBehaviorSource` exposes exact speed, heading, and position
  projections, including their matching validity/reason fields. It does not
  fall through to independent bout, eye, tail, or baseline selectors;
- each direct projection records the bundle digest, provider-motion manifest
  and verification digests, exact track interval, consumed paths and content
  hashes, and the structural track-partition hashes; and
- the exact-chaser `epoch_behavior` route can resolve the exact projection
  receipt from the same bundle and includes the bundle path/digest in its
  reactive selection identity and rendered provenance.

The general explorer accepts the explicit external carrier with:

```bash
scripts/run_palette_explorer.sh \
  --zarr-path /absolute/recording_analysis.zarr \
  --validated-behavior-bundle /absolute/validated_behavior_bundle.json
```

If `--exact-chaser-receipt` is supplied at the same time, it must resolve to
the exact receipt already bound by the bundle. A mismatch fails at launch.
Only `epoch_behavior` currently claims bundle-backed exact-chaser routing;
other exact-chaser routes retain their existing receipt/deep-audit behavior
until their closed requirement declarations are migrated.

The first real adapter canary used the GoodBatBadBat smoke recording and an
explicitly non-durable diagnostic bundle at
`/tmp/validated_recording_behavior_bundle_adapter_canary_uncommitted_20260831.json`.
It proved the exact provider-motion track is track 0 with 151,478 rows. Bundle
opening and current-source validation took approximately 4.6 seconds, the
provider-motion catalog and track-partition proof took 0.44 seconds, and seven
requested numeric arrays were read and rehashed in 0.08 seconds on the tested
workstation. A combined 0–10 second Core Behavior speed projection plus the
receipt-backed exact epoch view completed in approximately 14.7 seconds. The
same bundle digest appeared in Core Behavior metadata, the exact-chaser
selection identity, and exact-chaser rendered provenance.

That canary also exposed and closed one consumer defect without changing any
scientific payload. Epoch manifests describe fixed-width logical byte-string
columns such as `|S32`, while Palette's authorized columnar carrier stores them
as bounded two-dimensional `uint8` matrices. The exact-child receipt auditor
already reconstructed the logical string values before hashing, but the
targeted epoch source handle compared the physical matrix directly to the
logical declaration. Direct and consolidated physical values were identical;
there was no data mutation. The shared columnar module now owns the strict
logical reconstruction and storage-authority check, and both receipt auditing
and targeted consumption use that common grammar. Object-reference arrays
remain prohibited.

The temporary canary bundle records base commit
`53d8d26f1694397c3a2753d2f65c731dcaf0fc6b`, which does not include these
uncommitted changes. It is diagnostic evidence only and must not be promoted,
copied into durable operations, or treated as implementation authority. A
fresh bundle must be generated from the eventual green committed revision.

The real artifact corrected one initial assumption. Keypoint and detection
relative-frame runs may legitimately bind different provider-proxy timing and
row-authority identifiers. They can still share one logical acquisition axis.
Schema v1 preserves both provider-specific authorities and proves that common
axis through exact equality of the sealed acquisition-frame, timestamp,
validity, temporal-selection, chaser-identity, chaser-role, occurrence, and
chaser-position array declarations and content hashes. Equal shapes or row
counts are never sufficient.

One historical semantic limitation remains visible rather than normalized
away: this source records
`trial_index_integrity_status = palette_computed_not_producer_asserted` and
`step_end_interval_semantics = producer_contract_pending`. The bundle proves
that every selected child uses that exact historical semantic source; it does
not upgrade the source to producer-authored authority. New acquisition-backed
recordings should use the producer-bound successor when it is available.

## Definition of done

This design is implemented when:

- one exact bundle identifies the compatible validated Core Behavior and
  stimulus/chaser data for a recording;
- every available capability resolves to exact immutable child paths and
  digests with typed compatibility evidence;
- optional capability loss is scoped rather than invalidating unrelated data;
- Core Behavior, exact chaser, static publication, and export adapters can
  consume that bundle without independently resolving sources;
- adding a style- or projection-only visualization does not require a new
  scientific run;
- transformations that create reusable scientific meaning remain versioned,
  immutable successors with their own contracts and receipts;
- interactive and static views expose their exact consumed fields and display
  recipes; and
- focused, adversarial, real-artifact, Marimo, and required CI validation all
  pass before integration or deployment.

## Open decisions

1. **Recording-local discovery.** Schema v1 uses an external immutable JSON
   receipt. A future recording-local metadata mirror could improve discovery,
   but it must carry byte-identical logical content and must not become a
   selector or independently mutable authority.
2. **Default provider policy.** The current cohort uses keypoint-triad motion
   and bouts while retaining detection as a position comparator. Future bundles
   should bind a task choice, not encode that cohort-specific preference as
   universal precedence.
3. **Association generality.** Frame/bout stimulus membership should be generic
   enough for other protocols without erasing protocol-specific event
   semantics.
4. **Exploratory transforms.** Each source contract must state which transient
   aggregations are permitted and how they are labeled; existing prohibitions,
   such as alignment viewer rebinning, remain controlling.
5. **Promotion.** A validated selector-ineligible bundle is useful for exact
   analysis and review. Production selection remains a separate governed
   operation and is not implied by this design.

## Recommended first implementation slice

Implement a read-only bundle planner and validator over the existing exact
GoodBatBadBat sources and receipts. Emit external JSON only, with no Zarr,
registry, selector, or production mutation. Use it to prove that the selected
provider motion, bouts, body frame, semantic selection, relative children, and
optional capabilities form one exact compatibility set. Then migrate one Core
Behavior route and one exact-chaser route to consume the same normalized bundle
handle before widening the interface.
