# Composable Chaser Analytics Implementation Checklist

<!-- contract-meta
status: accepted-design
last_updated: 2026-08-20
implementation: partial
-->

Purpose: migrate Palette's existing chaser analyses into the composable,
provider-aware analytics system without changing their scientific meaning,
silently selecting observation sources, or retaining protocol-branded parallel
implementations.

This checklist extends:

- [`chaser_analysis_profile_design.md`](chaser_analysis_profile_design.md);
- [`chaser_stimulus_camera_temporal_projection_audit_2026-08-20.md`](chaser_stimulus_camera_temporal_projection_audit_2026-08-20.md);
- [`composable_stimulus_analysis_and_plot_recipes_design.md`](composable_stimulus_analysis_and_plot_recipes_design.md);
- [`position_body_frame_and_motion_provider_design.md`](position_body_frame_and_motion_provider_design.md);
- [`provider_position_chaser_distance_implementation_checklist.md`](provider_position_chaser_distance_implementation_checklist.md); and
- [`provider_aware_spatial_analytics_implementation_checklist_2026-08-18.md`](provider_aware_spatial_analytics_implementation_checklist_2026-08-18.md).

Existing immutable chaser-distance runs, derived components, cohort exports,
figures, and compatibility readers remain unchanged. New work publishes
successors or explicit adapters.

## Implementation checkpoint: full-profile planning and relative-frame storage

The current branch implements the provider-neutral planning and storage
foundation without changing a production selector or provider default:

- `chaser_behavior_full_v3` declares the complete mature module catalog,
  independent provider policies, requirement classes, and controlled
  capabilities. It is not yet the maintained runner default.
- `chaser_profile_applicability` produces a strict digest-bound plan that
  distinguishes applicable, inapplicable, missing, invalid, review-required,
  stale, and complete states.
- `chaser_relative_frame` computes complete acquisition-frame x chaser
  geometry over already-projected immutable inputs. It keeps the selected
  position provider independent from the optional anatomical body-frame
  provider, requires every provider to bind the exact row-axis authority, and
  never derives orientation from motion.
- `chaser_relative_frame_schema` defines numeric `ArrayContract` storage for
  the base relation and optional body extension, including exact row lineage,
  pixel and calibrated coordinates, transition censoring, reason codes, and
  source-camera/anatomical-axis conventions.
- `chaser_relative_frame_storage` converts strings to bounded registries,
  keeps row evidence in arrays, requires readable digest-bound subject,
  selection, occurrence, projection, and profile records, and optionally
  binds arena geometry plus its exact arena-to-source-camera transform as one
  capability.
- `materializers.chaser_relative_frame` writes a node-local candidate, validates
  every logical array and manifest, atomically publishes one immutable child
  below `analysis/chaser_relative_frame_runs/`, preserves all parent selectors,
  and verifies the final direct and consolidated metadata generation.
- `chaser_relative_frame_source_handle` is the strict consolidated reader for
  that immutable publication. It requires an exact run name, revalidates the
  manifest, payload and array hashes, proves direct/consolidated equivalence,
  and exposes copied read-only arrays rather than a mutable Zarr group.
- `provider_chaser_stimulus_source_handle` applies the same fail-closed reading
  rules to the existing selector-ineligible native stimulus-sample candidate.
  It preserves every native sample, including multiple stimulus samples that
  map to one acquisition frame, and proves their exact chaser and fish-side
  row lineage.
- `chaser_relative_distance_view` provides a provider-neutral, read-only view
  over a validated relative-frame handle. It preserves the complete
  frame-by-chaser relation and exposes nearest-chaser membership only as a
  derived convenience field.

The focused profile, applicability, computation, schema, storage, and real-Zarr
publication suite passed 128 tests outside the Codex sandbox before the
proxy-adapter increment. This remains selector-ineligible implementation
evidence, not a production rollout. The explicit current-recording proxy and
typed coordinate adapter now exist, but the standard v3 runner, scientific
module successors, registry projection, and production selectors do not use
them. This branch does not silently retain the historical duplicate-sample
collapse behavior.

## Accepted default-execution decision

For a configured production chaser protocol, the protocol adapter selects a
named, versioned **full chaser-analysis profile by default**. The profile plans
every mature analysis and plot recipe that is scientifically applicable to the
protocol and available capabilities. Operators should not have to invoke the
near-field, bout, trial, escape, gaze, or visualization modules one by one.

"By default" has these precise limits:

1. **Default planned, never default guessed.** The plan names exact position,
   body-frame/heading, motion, bout, stimulus, timing, geometry, and selection
   authorities. No module may choose a newest run or silently fall back between
   detections, keypoints, masks, or trajectory direction.
2. **Capability-driven applicability.** Static positioned objects, moving
   chasers, trialized pursuit, body-frame observations, eye orientation, and
   swim bouts enable different modules. A module that is scientifically
   inapplicable is recorded as `inapplicable` with a reason; it is not reported
   as successful or failed.
3. **Independent fail-closed modules.** A missing or invalid provider blocks
   every dependent module and the profile's corresponding readiness dimension,
   but does not prevent independent modules from completing.
4. **No false full completion.** A full-profile run is not `complete` while a
   required or conditionally required applicable module is blocked, stale,
   pending review, or unrun.
5. **Reduced profiles are explicit.** A distance-only, core-only, diagnostic,
   or visualization-free workload uses a separately named profile. It cannot be
   presented as the full protocol analysis.
6. **Reuse is exact.** The planner may reuse an existing immutable output only
   after validating its exact inputs, provider policies, parameters, manifests,
   and consolidated publication state.

Behavior labels such as `aggressive`, `random_non_chasing`, and `inert` remain
semantic role data, not module switches by themselves. The protocol adapter
derives explicit capabilities and role intervals; the analysis profile uses
those declarations to determine applicability.

## Scientific axes that must remain separate

### Position provider

The position provider answers where the fish is. Initial explicit estimators
include `detection_bbox_centroid.v1` and
`keypoint_anatomical_triad_mean.v1`. Future mask providers remain distinct
estimators.

Position supplies spatial trajectory, fish-to-chaser distance, occupancy,
entry/dwell, path, speed, acceleration, and separation metrics. A consumer must
bind one exact position run and must not substitute another provider on invalid
rows.

### Body-frame or heading provider

The body-frame provider answers which way the fish's anatomy points. The
canonical zebrafish anterior axis is:

```text
swim_bladder -> midpoint(eye_left, eye_right)
```

Reusable authority should be the complete body frame (`origin_xy`,
`forward_axis_xy`, `left_axis_xy`, and `axis_valid`), with scalar heading
derived under its declared coordinate convention. The body-frame source may
differ from the position modality only after exact row, time, coordinate, and
transform compatibility is proven.

A position centroid, bounding-box orientation, velocity direction, and
anatomical heading are different estimators. None may silently stand in for
another. A future trajectory-direction estimator requires its own controlled
ID and cannot satisfy a requirement for anatomical heading.

### Motion policy

Linear motion derives from one exact position series and timing authority.
Angular motion derives from one exact body-frame/heading series and timing
authority. Algorithm identity remains separate from provider identity, so a
provider comparison does not masquerade as an algorithm change.

### Temporal and event selections

Every product binds exact composable selections. The system must distinguish:

- a stimulus step or composed pre/chaser/post selection;
- a near-field visit, defined by explicit entry/exit hysteresis and censoring;
- a controller chase trial, identified by exact logged trial authority;
- a swim bout, identified by one exact bout run and selected signal level; and
- an escape event, identified by a versioned classifier over an exact bout and
  motion source.

These units are not interchangeable. A near-field visit is not automatically a
chase trial, and one chase trial may contain zero, one, or several visits,
bouts, or escape events.

## Capability and default-module matrix

| Declared capability | Default applicable analyses |
| --- | --- |
| positioned chaser/object + valid position | chaser distance, quadrant/radial/near-field occupancy, distance CDF, entries, dwell, visit extraction |
| ordered position + timing | path, speed, acceleration, separation velocity, response regimes |
| swim-bout authority | distance-conditioned bout rate, onset distance, duration, path length, peak speed, visit membership |
| moving chaser track | fish/chaser separation decomposition, pursuit and recapture metrics |
| logged pursuit trials | trial segmentation, trigger-aligned trajectories, escape/freeze, habituation |
| body-frame/heading authority | egocentric bearing, heading change, turn-toward/away, directed escape |
| body frame + bout authority | turn bias and anatomically directed bout response |
| body frame + eye-orientation authority | gaze and eye-relative chaser tracking |
| arena geometry + virtual-reference policy | wall-matched controls, annulus normalization, wall-distance diagnostics |
| completed scientific products | compatible per-window, per-visit, per-trial, per-event, and cohort plot recipes |

Unsigned tangential velocity or circling magnitude can in principle derive from
position and motion alone. Existing combined bout-response components also
contain heading-dependent metrics; migration must either preserve that full
dependency or split a clearly named distance/motion-only component from the
body-frame extension. It must not change an existing schema's meaning in place.

## Shared provider-aware chaser-relative foundation

The migrated suite should consume one common, immutable relative-geometry
foundation rather than independently joining fish, chaser, timing, and event
rows in every component.

The foundation must preserve two different authorities:

1. Native stimulus-state samples and their exact `stimulus_state_key`, source
   timestamps, chaser identities, role intervals, and source-row lineage.
2. A versioned projection onto acquisition/track frames for fish behavior and
   motion, preserving the exact temporal sampling/interpolation policy and all
   contributing stimulus-state rows.

The acquisition-frame projection should materialize, per frame and chaser:

- acquisition frame and exact track-sample identity;
- fish position, validity, provider ID, and source-row identity;
- chaser position, validity, identity, behavior role, and source-row identity;
- fish-to-chaser vector and distance in declared pixel and physical frames;
- nearest-chaser identity without discarding the complete per-chaser axis;
- position-transition validity and explicit censoring state;
- composable selection and occurrence membership;
- controller trial identity and active-pursuit state when available;
- exact timing, geometry, transform, and scale authorities; and
- readable policy records plus compact array-backed row evidence.

An optional body-frame extension over the exact same row domain adds:

- body-frame origin, forward and left axes, and validity;
- scalar heading under the declared coordinate convention;
- chaser forward/left coordinates and egocentric bearing;
- heading-transition validity; and
- exact body-frame provider and row-join lineage.

Missing body-frame rows remain invalid and never synthesize orientation from
motion. Distance-only consumers use the base product without requiring the
extension.

## Existing analysis migration inventory

| Existing surface | Scientific unit | Provider requirements | Target migration state |
| --- | --- | --- | --- |
| `chaser_distance` | frame x chaser | position | replace detection-specific authority with sealed provider-aware base; retain v1 reader |
| quadrant/radial/near-field occupancy | selection/phase x chaser | position, geometry | consume exact composable selections and common relative geometry |
| epoch behavior summary | selection x fish/chaser | position, motion, bouts as declared | replace fixed alias assumptions with explicit selections |
| response regimes | distance bin x selection x chaser | position, motion | preserve dropout, validity, and near/far support guards |
| egocentric bearing | frame x chaser | position, body frame | consume the body-frame extension |
| gaze tracking | frame/selection x chaser | position, body frame, eye orientation | remain blocked unless every orientation authority is valid |
| bout response | bout x reference | position, motion, bouts; body frame for directed metrics | split only through new versioned schemas if dependencies differ |
| escape events | escape event and chase trial | position, motion, bouts, trials; body frame for directed/high-turn metrics | preserve trigger, pursuit, recapture, valid-time rates, and threshold evidence |
| escape/freeze summary | chase trial | position, motion, trials | consume exact trial intervals and preserve candidate-classification status |
| visit trajectories | near-field visit | position, geometry, visit policy | become immutable per-visit recipes over persisted visit membership |
| ring traversal | near-field visit | position, bouts, events; moving-chaser transform for chase | preserve separate static-object and moving-chaser frames |
| habituation figures | chase trial | trial and escape products | become per-recording and cohort recipes with dropout/wall confounds required |

## Implementation checklist

### Phase 0: freeze contracts, applicability, and compatibility evidence

- [ ] Freeze representative existing outputs for near-field, response-regime,
      bout-response, escape-event, escape/freeze, visit, traversal, and
      habituation surfaces as compatibility fixtures.
- [ ] Inventory every existing module's row axis, denominator, validity,
      temporal window, provider, geometry, and heading dependency.
- [ ] Freeze controlled capability IDs for positioned object, moving chaser,
      logged trial, swim bouts, body frame, eye orientation, arena geometry,
      and virtual-reference support.
- [ ] Freeze module applicability states and reason codes: `applicable`,
      `inapplicable`, `blocked_missing_capability`, `blocked_invalid_source`,
      `blocked_review_required`, `stale`, and `complete`.
- [ ] Decide whether the first provider-aware bout product remains one full
      heading-dependent schema or becomes separately versioned distance/motion
      and body-frame extensions.
- [x] Freeze the current-recording input-provenance proxy projection policy:
      select one complete all-chaser sample by logged CPU timestamp, stimulus
      frame, and source-row lineage; retain every candidate; never interpolate
      or carry across an unmapped input acquisition. Physical presentation
      remains a separate unavailable authority.
- [x] Permit current-recording chaser analyses under an explicit
      `input_provenance_proxy_allowed` requirement while keeping
      `physical_presentation_required` distinct and unavailable; never silently
      fall back from physical alignment to proxy alignment.
- [ ] Confirm that every existing protocol-branded schema has either a generic
      successor or an explicit immutable compatibility boundary.

### Phase 1: full-profile planning and default execution

- [x] Define the provider-neutral `chaser_behavior_full_v3` declaration with a
      full scope and complete mature module catalog. Production adapter
      selection remains gated on the runtime items below.
- [ ] Keep protocol identity, chaser identity, behavior role, and module
      applicability as separate records.
- [x] Add exact profile fields for position policy, body-frame policy, motion
      policy, bout policy, geometry policy, trial policy, module requirement
      class, plot-recipe selection, and explicit temporal-alignment selection.
- [x] Support `required`, `conditional_required`, and `optional` module
      requirement classes without treating blocked work as inapplicable.
- [x] Preserve schema-v1 normalized payloads and digests while introducing the
      versioned schema-v2 fields.
- [x] Implement a digest-bound, strict applicability-plan contract that keeps
      `not_applicable`, missing, invalid, review-required, stale, applicable,
      and complete states distinct and prevents reduced profiles from claiming
      full readiness.
- [x] Add `chaser_temporal_alignment` at the `chaser_distance` dependency root.
      An explicit verified input-provenance projection makes the chaser chain
      applicable; a physical-presentation request remains blocked with
      `presentation_time_unavailable`; independent detection occupancy is not
      blocked.
- [x] Implement the pure
      `latest_logged_cpu_state_per_input_acquisition_proxy_v1` selector with
      complete candidate lineage, no-carry behavior, exact unique-input-frame
      denominator, source manifest/verification binding, and read-only arrays.
- [x] Require a semantically exact proxy projection record in the prepared
      common relative-frame context and preserve its readable record and
      digest in publication provenance.
- [x] Publish the proxy candidate/selection arrays under an immutable typed,
      selector-ineligible Zarr schema and add a strict consolidated-metadata
      source handle that verifies the exact run, manifest, arrays, source
      authority, and direct/consolidated metadata agreement.
- [x] Add a dry-run/apply CLI for publishing one exact named proxy run from one
      verified native provider-chaser candidate. Dry-run performs no target
      write; apply retains parent selector state and publishes no selector.
- [x] Add the keyed adapter that joins a published proxy selection to the
      common relative-frame input without changing coordinate authority or
      losing the candidate multiplicity arrays. The adapter reopens the exact
      native source, verifies the proxy binding, uses the published
      arena-to-selected-canvas-to-source-camera transform chain, and makes no
      camera-timestamp or physical-presentation claim.
- [x] Add an explicit selector-ineligible candidate DAG for native proxy
      publication, coordinate-safe relative-frame publication, and one final
      digest-bound applicability/readiness receipt.
- [ ] Resolve the complete dependency graph before submission and persist the
      normalized profile, explicit overrides, applicability decisions,
      execution order, and digests.
- [ ] Run independent branches concurrently after their immutable authorities
      are available; fan in only where dependencies require it.
- [ ] Reuse exact eligible products rather than recomputing them.
- [ ] Add separately named reduced profiles; prevent them from publishing full
      chaser-profile readiness.

### Phase 2: common chaser-relative frame products

- [x] Reuse the existing selector-ineligible provider-chaser candidate as the
      native stimulus-state authority through a strict loader-minted handle;
      do not add a parallel writer or schema family.
- [x] Define schemas and `ArrayContract` declarations for acquisition-frame
      relative geometry and its optional body-frame extension.
- [x] Implement the pure acquisition-frame x chaser computation over already
      projected inputs, preserving stable identity separately from time-varying
      role, exact row keys/source rows, pixel and physical geometry, explicit
      transition censoring, and an optional anatomical body-frame extension.
- [x] Enforce Palette's source-camera top-left/+Y-down convention and
      determinant-negative-one anatomical-left body-frame convention without a
      presentation flip or velocity-heading fallback.
- [ ] Build typed loader-minted handles for exact chaser source, role intervals,
      composable selection, position track, timing, geometry, transform, and
      scale authorities.
- [x] Add strict loader-minted handles for the existing native stimulus-sample
      candidate and the immutable acquisition-frame relative publication,
      including exact manifests, payloads, arrays, source authorities, and
      consolidated metadata.
- [x] Materialize complete frame x chaser arrays without collapsing to only the
      nearest chaser.
- [x] Preserve all native stimulus samples and their exact stimulus-run,
      source-row, chaser, acquisition-frame, and timestamp lineage, including
      duplicate acquisition-frame mappings.
- [x] Preserve all contributing native sample IDs and the exact deterministic
      selection receipt for each represented input-acquisition projection. The
      current proxy performs no interpolation or carry.
- [x] Persist sample, transition, tracking-gap, selection, occurrence, trial,
      and censoring validity separately.
- [x] Publish an optional body-frame extension through an exact keyed join.
- [x] Reject same-length-only provider joins, cross-recording composition,
      coordinate flips, implicit transforms, provider fallback, stale
      manifests, stale arrays, and stale consolidated metadata in the common
      computation and published-source handles.
- [x] Apply those same fail-closed rules to immutable proxy publication and its
      exact named source handle, including source-manifest and source-handle
      verification digests, candidate multiplicity, same-sample all-chaser
      lineage, and selector-ineligible state.
- [x] Keep row-cardinality evidence in arrays and metadata compact, readable,
      and bounded.

### Phase 3: position- and motion-only analyses

- [x] Add a provider-neutral read-only distance view over the validated common
      relative-frame publication without collapsing the complete chaser axis.
- [ ] Publish the sealed provider-aware chaser-distance successor with generic
      `source_position_*` lineage.
- [ ] Migrate quadrant and radial occupancy to exact composable selections.
- [ ] Migrate near-field occupancy v2 with valid-time entry denominators,
      hysteresis, complete-visit dwell, and censoring unchanged.
- [ ] Migrate epoch behavior summary v2 to explicit selections and provider
      motion.
- [ ] Migrate response regimes with distinct fish and chaser separation
      contributions, dropout QC, and support thresholds.
- [ ] Add a sealed distance/motion bout-response surface if Phase 0 approves the
      split, including onset distance, bout rate, duration, path length, peak
      speed, displacement, and distance-band summaries.
- [ ] Validate static-object pre/post and moving-chaser calculations separately.
- [ ] Validate detection and keypoint position providers against identical
      selections without averaging or selecting a default.

### Phase 4: body-frame and heading integration

- [ ] Consume one explicit body-frame provider independently from position.
- [ ] Validate exact row/time/coordinate compatibility or a sealed projection
      before composing position and body frame.
- [ ] Migrate egocentric chaser bearing to the common body-frame extension.
- [ ] Migrate turn-toward/away, turn bias, predicted miss, directed escape, and
      angular-motion metrics only after their body-frame dependencies are
      explicit.
- [ ] Migrate gaze only when eye-orientation and body-frame authorities are
      independently valid.
- [ ] Keep speed-only escape classification separate from the heading-based
      high-turn tier in schema and provenance.
- [ ] Add a genuinely mask-aware body-frame manifest and reader before
      publishing `mask_component_axis` output.
- [ ] Keep full-body-mask orientation deferred until its shape estimator,
      anterior/posterior polarity, validity, and canary policy are frozen.
- [ ] Add provider-specific coverage and disagreement evidence; do not silently
      promote keypoint or mask heading.

### Phase 5: visits, trials, bouts, and escape events

- [ ] Materialize near-field visit membership with exact hysteresis, invalid-gap
      reset, boundary censoring, and complete-dwell policies.
- [ ] Materialize controller chase trials from exact logged trial IDs while
      preserving gaps, fallback use, trigger selection, and trial ordinal.
- [ ] Bind one exact swim-bout run and selected signal level; prevent the known
      multi-level bout duplication failure.
- [ ] Publish escape events with separate speed threshold, optional high-turn
      tier, trigger distance, valid-time rates, gain, recapture, latency, and
      threshold sweep.
- [ ] Preserve per-event, per-trial, and reduced per-recording arrays rather
      than embedding row evidence in metadata.
- [ ] Prove that bouts and escapes attach to exactly one compatible visit/trial
      occurrence and are never double counted.
- [ ] Keep event counts in rates even when an event-aligned trace is unusable;
      record the trace exclusion reason separately.

### Phase 6: composable plot recipes and Marimo

- [ ] Define immutable recipes for pre/chaser/post distance, near-field,
      response-regime, bearing, gaze, and bout-response panels.
- [ ] Define per-visit trajectory overlay, per-visit reference sheet, and ring
      traversal animation recipes.
- [ ] Define per-trial chaser-centric trajectory, trigger-aligned distance,
      escape raster, escape/freeze, wall-distance, and habituation recipes.
- [ ] Define per-escape onset-aligned gain and recapture trace recipes.
- [ ] Require wall geometry and virtual controls in recipes whose scientific
      interpretation depends on them.
- [ ] Preserve separate coordinate frames for static-object visits and
      moving-chaser pursuit; never mix them in one unlabeled panel.
- [ ] Bind exact scientific products and render parameters without recomputing
      scientific arrays inside the viewer.
- [ ] Expose applicability, provider IDs, coverage, censoring, support,
      promotion state, and source manifests beside every view.
- [ ] Publish immutable plot attempts and validate PNG/spec/media digests and
      final consolidated visibility.

### Phase 7: cohort products, registry, and automation readiness

- [ ] Freeze a cohort input manifest with exact recording, biological subject,
      protocol, chaser role, provider, and scientific-run identities.
- [ ] Publish recording-balanced cohort products separately from pooled-event,
      pooled-bout, or pooled-frame descriptive products.
- [ ] Preserve visit, trial, event, recording, subject, camera, and acquisition
      batch as distinct statistical levels.
- [ ] Add recording-local analysis and plot offers for every completed module
      and every source selection/event occurrence it references.
- [ ] Project offers into SQLite through immutable worker receipts and one
      dependent serial finalizer.
- [x] Publish a candidate-chain applicability/readiness receipt that binds the
      exact native, proxy, relative-frame, profile, and capability evidence
      while explicitly declining registry update and selector activation.
- [ ] Expose per-module complete, blocked, stale, inapplicable, review, render,
      and registry states; do not collapse them to one chaser status.
- [ ] Mark the full chaser profile ready only when all required applicable
      modules and required recipes satisfy their contracts.
- [ ] Batch notifications for blocked/review-required protocol workloads without
      sending one message per failed worker.

### Phase 8: validation, canary, promotion, and production

- [ ] Add closed-form synthetic fixtures for distance, bearing, separation,
      visit censoring, trial segmentation, event attachment, pursuit/recapture,
      and heading-coordinate conventions.
- [x] Add an asymmetric coordinate fixture for the proxy adapter proving no
      presentation reflection or heuristic Y flip reaches the source-camera
      chaser coordinates.
- [ ] Extend asymmetric coordinate fixtures across the later production
      module publications so no presentation reflection or
      heuristic Y flip reaches camera-native fish/body-frame calculations.
- [ ] Test missing, stale, partial, duplicated, reordered, cross-provider, and
      cross-recording inputs for every provider join.
- [ ] Test that position-only modules complete while heading-dependent modules
      remain explicitly blocked when body-frame authority is absent.
- [ ] Test that inapplicable static-object or trial modules are distinguished
      from failed applicable modules.
- [ ] Test exact reuse, immutable retry, selector ineligibility, source
      preservation, compact metadata, and direct/consolidated equivalence.
- [ ] Run the first full canary with detection and keypoint position providers
      kept separate and one reviewed keypoint body-frame provider.
- [ ] Visually inspect representative static pre/post visits, moving-chaser
      trials, escape events, bearing, and provider disagreement.
- [ ] Compare provider coverage and bias by camera, stimulus state, distance,
      visit, trial ordinal, and response class.
- [ ] Record a timestamped position-provider promotion decision separately from
      any body-frame-provider promotion decision.
- [ ] Pass every required CI check before integration, production selector
      activation, shared-checkout update, or full cohort campaign.
- [ ] Launch production from a frozen cohort/profile plan and preserve task,
      receipt, finalization, registry, and export accounting.

## Current implementation checkpoint

- [x] A protocol-neutral full chaser profile declares the existing analysis
      graph.
- [x] Generic successors exist for the historical protocol-branded quadrant,
      near-field, epoch-summary, and escape/freeze families.
- [x] Composable selection, provider trajectory, occupancy, and occupancy
      contrast candidates exist for detection and keypoint position.
- [x] A selector-ineligible provider-position chaser-distance candidate exists.
- [x] Strict source handles now validate both that native stimulus-sample
      candidate and the new selector-ineligible relative-frame publication.
- [x] A coordinate-safe adapter binds an exact published proxy to the common
      relative frame through the authoritative arena-to-camera transform.
- [x] A three-stage candidate LSF workflow and final applicability receipt
      preserve exact dependencies without updating SQLite or production
      selectors.
- [x] A provider-neutral distance view can consume the validated relative-frame
      handle while retaining every chaser and exact frame-level lineage.
- [x] The recording explorer exposes read-only provider-aware bout-response rows
      and exact heading-bound egocentric bearing.
- [x] Typed position, body-frame, tracking, timing, and provider-motion
      foundations exist, including explicit mixed-modality compatibility.
- [ ] The provider-aware chaser-distance candidate is not a sealed production
      base and no provider has been promoted for GoodBatBadBat.
- [ ] Near-field, response-regime, trial, escape, gaze, and visualization
      surfaces have not yet migrated to the new composable provider contracts.
- [ ] Plot recipes, recording-local discovery, registry projection, full cohort
      products, and production profile readiness remain open.
- [ ] Required CI and integration remain separate gates for the implementation
      work described here.

### Implementation checkpoint: 2026-08-20

The pure proxy selector, strict temporal-alignment capability, v3 profile
dependency, immutable selector-ineligible proxy publication, strict proxy
reader, explicit CLI, coordinate-safe relative-frame adapter, candidate DAG,
and digest-bound applicability receipt are implemented. Sharded focused
validation now passes 189 focused/adjacent tests covering profile parsing,
applicability, candidate selection, same-sample all-chaser behavior,
missing-frame no-carry, malformed lineage, source verification, denominator
declarations, schema/storage tampering, dry-run/apply publication, strict
direct/consolidated readback, the common relative-frame
computation/storage/materializer, exact proxy-bound relative-frame source
handles, and the provider-neutral distance view. Ruff, `py_compile`, and
`git diff --check` also pass for the changed slice.

The adapter/DAG coverage includes the exact transform, source-binding
rejection, candidate workflow ordering, non-production state, and receipt
tamper rejection. Tests wrote only disposable temporary Zarr fixtures. No
production Zarr, selector, registry, or archive was changed.
Standard profile-runner integration, downstream module publication, registry
projection, selector promotion, and required CI remain open before this branch
is merge-ready or production-authoritative.

The producer audit establishes that a physical temporal projection from native
stimulus states to camera exposures is unsupported for current recordings.
`source_acquisition_frame_index` identifies the Orange acquisition used as
input provenance; the Citrus state was produced after that frame arrived and
has no persisted state-bound presentation timestamp. Current native evidence
uses `all_native_states_by_input_acquisition_provenance_v1`. A request requiring
physical presentation remains unsupported with
`presentation_time_unavailable`. Current analyses may instead explicitly opt
into `input_provenance_proxy_allowed`, with every product and visualization
marked `controller_input_provenance_proxy`; this is never an implicit fallback
or evidence of physical presentation. Native-sample relations must not be
counted as independent camera observations: repeated fish-side timepoints
require explicit sample/frame multiplicity counts and a declared denominator.
The current path-by-path behavior, producer verdict, proxy boundary, and
statistical risks are recorded in
[`chaser_stimulus_camera_temporal_projection_audit_2026-08-20.md`](chaser_stimulus_camera_temporal_projection_audit_2026-08-20.md).

## Explicit non-goals for the first implementation

- Do not retrain or promote the current mask models.
- Do not infer anatomical heading from a detection box, velocity vector, or
  unpolarized full-body mask.
- Do not add arbitrary user formulas or numeric anchor weights.
- Do not rewrite historical chaser components or figures.
- Do not make the Marimo viewer or SQLite registry a scientific authority.
- Do not fit the proposed escape-hazard, semi-Markov state-transition, response-
  latency, phenotype, or body-shape roadmap models until the common relative
  frame and event contracts are stable.
