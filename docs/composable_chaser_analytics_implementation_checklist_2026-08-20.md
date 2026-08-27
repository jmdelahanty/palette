# Composable Chaser Analytics Implementation Checklist

<!-- contract-meta
status: accepted-design
last_updated: 2026-08-25
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

## Protocol semantic identity checkpoint (2026-08-24)

- [x] Validate and materialize the exact Citrus semantic hash, semantic JSON,
      and trial-index JSON as one fail-closed stimulus-run contract.
- [x] Bind every materialized step to its exact producer recipe index, mode,
      family, duration, display context, and full semantic hash.
- [x] Keep completed and selector-visible stimulus runs immutable; require a
      new successor when semantic metadata is missing.
- [x] Add nullable registry migration 72 fields and exact array-backed
      extraction so the `SOLID_BLACK -> CHASER` and `CHASER` cohorts are
      queryable without reopening every Zarr.
- [x] Preserve the distinction between the existing authored-protocol
      `protocol_hash` and producer `protocol_semantic_hash`.
- [x] Add a producer-authored trial-index checksum in Citrus snapshot v2.
      Historical snapshot-v1 recordings retain the explicitly local Palette
      digest and are never upgraded by inference.
- [x] Replace raw camera-correspondence `STEP_END` authority in snapshot v2
      with exact half-open `stimulus_frame_num` execution intervals. Historical
      snapshot-v1 recordings retain the conservative unresolved-end policy.
- [x] Materialize a sealed per-row stimulus/camera correspondence proxy for
      v2 imports, including exact content digests, step and chaser-phase
      membership, and an explicit visualization/exploratory-only use class.
- [x] Complete the code-level frame-bound acquisition chain for future
      recordings. Shaman-v2 ABI revision 3 carries a per-slot Orange recording
      token; the v6 companion seals that token, the acquisition camera
      ID/serial/numeric-ID tuple, and the exact closed raw H5 plus finalized
      observation receipt. Palette validates those producer records, verifies
      the raw H5 size/SHA-256, joins raw `timestamp_ns_session` by exact
      `(chaser_index, stimulus_frame_num)`, and feeds the existing immutable
      selector-ineligible `controller_input_provenance_proxy` materializer.
      Controlled four-camera hardware validation remains a deployment and
      promotion gate; this contract still does not claim physical display
      presentation.
- [x] Implement a pure, selector-ineligible v2 semantic selection candidate
      with optional `standalone_solid_black` plus `chaser_pre`,
      `chaser_training`, and `chaser_post` nested inside the exact `CHASER`
      step. It uses the conservative boundary common to both unresolved
      `STEP_END` conventions and records any excluded terminal frame.
- [x] Implement immutable, selector-ineligible semantic selection publication,
      a strict source handle, and the first exact binding into the provider
      position suite. The operator path requires an exact epoch manifest digest
      and never updates selectors or the registry.
- [x] Add a recording-local provider motion/swim-bout epoch-summary v2 that
      consumes the exact semantic selection handle, limits output to
      `chaser_pre`/`chaser_training`/`chaser_post`, and repeats the producer
      semantic hash and step identity on every output row. Its legacy v1 path
      remains unchanged and both paths remain selector-ineligible.
- [x] Add a version-dispatched provider epoch cohort input/export/Arrow/plot
      successor. Schema v2 freezes each exact semantic selection run, manifest,
      producer hash, and summary run; carries role/hash/step identity on every
      fish and bout row; keeps source window ID separate from semantic order;
      and repeats the proxy-not-physical-presentation boundary in export and
      plot receipts. Schema v1 remains unchanged.
- [ ] Activate a maintained profile only after exact acquisition-row mapping,
      acquisition-produced immutable stimulus successors where source evidence
      permits them, focused scientific review, and all required CI are
      complete.
- [x] Mark standalone-baseline contrasts not applicable for `CHASER`-only
      recipes; never substitute legacy `black_before`/`black_after` roles.

The durable contract is
[`protocol_semantic_step_identity_contract.md`](protocol_semantic_step_identity_contract.md).

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
- [x] Publish the sealed provider-aware chaser-distance successor with generic
      `source_position_*` lineage.
- [x] Implement selector-ineligible quadrant and moving-chaser radial occupancy
      over exact caller-role-bound half-open epoch selections, one sealed
      position provider, and one reviewed circular geometry/scale authority;
      publish the resulting position-only suite as typed arrays in one exact
      immutable analysis-Zarr run; and bind it to an explicitly reduced
      profile and exact publication/readiness DAG.
- [x] Implement the selector-ineligible near-field position suite with the
      maintained valid-time entry denominator, invalid-gap reset, hysteresis,
      complete-visit dwell, and boundary/gap censoring semantics unchanged.
      Persisted visit-membership arrays remain Phase 5 work.
- [x] Migrate epoch behavior summary v2 to the exact protocol-semantic
      selection, provider motion, and one exact selector-ineligible swim-bout
      authority. The semantic source and producer step identity are repeated
      on every output row; this is a sealed proxy-aligned recording-local
      candidate, not an exact physical-presentation claim.
- [ ] Migrate response regimes with distinct fish and chaser separation
      contributions, dropout QC, and support thresholds.
- [x] Add a sealed distance/motion bout-response surface under the versioned
      generalized successor split, including exact onset/end chaser distance,
      separation gain, bout rate, duration, path length, peak/mean speed,
      displacement, and valid-time distance-band summaries. One explicit
      selector-ineligible swim-bout run and default signal are bound; the
      optional body-frame directed extension does not gate the base product.
- [ ] Validate static-object pre/post and moving-chaser calculations separately.
- [x] Validate detection and keypoint position providers against identical
      selections without averaging them. The bounded eight-recording,
      four-camera comparison remained evidence-only; the separate timestamped
      provider decision retained detection as the GoodBatBadBat default.

### Phase 4: body-frame and heading integration

- [ ] Consume one explicit body-frame provider independently from position.
- [ ] Project body-frame observations onto the common relative-frame axis only
      by exact `acquisition_frame_id` identity. Distinguish a missing exact
      body-frame source row from a present source row whose anatomical axis is
      invalid; retain both as explicit invalid evidence, without interpolation
      or a motion-heading fallback. Position-only measurements remain usable
      wherever their own validity contract passes.
- [ ] Validate exact row/time/coordinate compatibility or a sealed projection
      before composing position and body frame. Persist the source-row identity,
      coverage state, anatomical-axis validity, and reason code on every
      frame-by-chaser row.
- [ ] Bind an existing applicable anatomical-orientation review receipt or
      produce a chaser-specific one before treating body-relative output as
      reviewed scientific evidence. Structural body-frame completeness alone
      is not a review claim.
- [ ] Migrate egocentric chaser bearing to the common body-frame extension.
- [ ] Migrate turn-toward/away, turn bias, predicted miss, directed escape, and
      angular-motion metrics only after their body-frame dependencies are
      explicit.
- [x] Add a gaze successor that consumes an exact body-frame relative run and
      exact compact-v7 eye run only after numerical convention validation and
      a human-reviewed ellipse-direction receipt bind the exact 41-array eye
      payload and review artifact. Raw versus smoothed gaze is an explicit,
      digest-bound source-channel policy; smoothed is the loader default.
- [x] Keep speed-only escape classification separate from the heading-based
      high-turn tier in schema and provenance.
- [ ] Add a genuinely mask-aware body-frame manifest and reader before
      publishing `mask_component_axis` output.
- [ ] Keep full-body-mask orientation deferred until its shape estimator,
      anterior/posterior polarity, validity, and canary policy are frozen.
- [ ] Add provider-specific coverage and disagreement evidence; do not silently
      promote keypoint or mask heading.

The 2026-08-27 audit of the first deeply audited recording found that its
body-frame source is present and already has an exact ordered-instance-key
compatibility proof with the keypoint-position provider. The current chaser
adapter nevertheless supplies `body_frame=None`, and the cohort runner
explicitly requests `--no-body-extension`. The precise projection terminology,
recording-specific coverage counts, and bout/trial impact are recorded in
`docs/diagnostics/chaser_body_frame_projection_gap_2026-08-27.md`.

### Phase 5: visits, trials, bouts, and escape events

- [ ] Materialize near-field visit membership with exact hysteresis, invalid-gap
      reset, boundary censoring, and complete-dwell policies.
- [x] Materialize controller chase trials from exact logged trial IDs while
      preserving gaps, trigger selection, and per-chaser trial ordinal. The
      fallback-used field is preserved and required to remain false; inferred
      fallback segmentation is prohibited in the successor. Exact active-row
      membership and first-to-last visualization/censoring envelope identity
      are separate arrays; every envelope gap has a reason code, and an active
      row with unavailable trial identity remains unresolved nonmember evidence.
- [x] Bind one exact swim-bout run and selected signal level; prevent the known
      multi-level bout duplication failure.
- [x] Publish escape events with separate speed threshold, optional high-turn
      tier, trigger distance, valid-time rates, gain, recapture, latency, and
      threshold sweep.
- [x] Preserve per-event, per-trial, and reduced per-recording arrays rather
      than embedding row evidence in metadata.
- [x] Attach bouts and escape events only through exact active trial membership;
      retain onset envelope identity and gap reason for visualization and
      censoring without counting a gap as a trial event.
- [ ] Prove that bouts and escapes attach to exactly one compatible visit/trial
      occurrence and are never double counted.
- [x] Keep event counts in rates even when an event-aligned trace is unusable;
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
      The schema-v2 provider epoch cohort contract now supports all of these
      identities, but no real cohort manifest has yet been operator-authored or
      published from immutable semantic recording successors.
- [x] Freeze the bounded position-suite cohort task with exact recording,
      provider manifest, epoch-role, geometry-selection, physical-scale, arena,
      and camera identities. Biological-subject and generalized protocol
      bindings remain open for the full production cohort manifest.
- [x] Publish recording-balanced position-suite cohort products separately from
      pooled-event, pooled-bout, or pooled-frame descriptive products. The
      position canary explicitly declines inferential statistics and keeps one
      recording as the aggregation unit; generalized event/bout cohort products
      remain open.
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
- [x] Record a timestamped position-provider default recommendation separately
      from any body-frame-provider promotion decision. The bounded
      GoodBatBadBat recommendation is in
      `docs/goodbatbadbat_position_provider_default_recommendation_2026-08-23.md`;
      selector activation remains pending required CI and explicit approval.
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
- [x] Selector-ineligible, sealed, receipt-backed detection-bbox-centroid and
      keypoint-triad chaser-distance canaries now exercise the generic
      `source_position_*` publication and bounded reader.
- [x] The recording explorer exposes read-only provider-aware bout-response rows
      and exact heading-bound egocentric bearing.
- [x] Typed position, body-frame, tracking, timing, and provider-motion
      foundations exist, including explicit mixed-modality compatibility.
- [x] A timestamped GoodBatBadBat decision retains
      `detection_bbox_centroid.v1` as the current position-provider default,
      based on an eight-recording, four-camera comparison. This is not a
      body-frame/heading promotion and no production selector has been
      activated.
- [x] A single-recording selector-ineligible position suite now exercises
      exact pre/training/post distance CDF, quadrant, moving-chaser radial,
      near-field, and aggressive-minus-inert products from the detection
      provider.
- [x] A bounded eight-recording/four-camera successor now exercises the same
      position suite with exact frozen per-recording provider, epoch, geometry,
      and physical-scale authorities. Cohort summaries use one value per
      recording, and radial plots require complete eight-recording support.
- [x] An immutable selector-ineligible analysis-Zarr publication stores the
      seven position-suite tables as typed arrays with compact readable
      manifests, bounded receipts, and strict direct/consolidated readback.
- [x] The reduced `chaser_position_suite_v1` profile and exact two-job LSF DAG
      bind publication followed by a non-mutating readiness receipt. The
      receipt separates scientific-candidate completeness from CI, selector,
      production-authority, and registry-projection readiness.
- [ ] Response-regime, persisted visit membership, and generalized immutable
      visualization recipes have not yet migrated to the new composable
      provider contracts. Controller-trial, generalized bout-response,
      escape/freeze, gaze, their shared selector-ineligible publication, and
      the digest-bound v4 full-profile envelope are complete locally. The
      gaze source still requires a recording-specific reviewed convention
      receipt, and no real recording has exercised the complete graph yet.
- [ ] Immutable generalized plot recipes, recording-local discovery, explicit
      selector promotion, serialized registry projection, non-position cohort
      products, and full production-profile readiness remain open.
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

### Implementation checkpoint: 2026-08-21

Commit `bda296db0180cc6a1cffe4d4be89c7fa5f039075` was deployed through a
commit-pinned cluster worktree and used to publish two selector-ineligible
GoodBatBadBat canaries:

- `provider_chaser_distance_detection_bbox_centroid_canary_20260821_v2`
- `provider_chaser_distance_keypoint_triad_canary_20260821_v2`

LSF jobs `153531902` and `153531903` succeeded. Each publication contains
149,946 frames, two chasers, 299,892 rows, and 38 arrays. Neither publication
updated a selector, registry, or production authority. The detection and
keypoint manifest digests are respectively
`df5ed6cc3b43f7672898a4cbee006266be0dca7ee25774209d5b91d316127710`
and `5f4f6131e6cd48c495b61f739276c5a873e6b5c0cf50f73433717863bd265b7e`.

The compact-validation correction is effective: the run metadata files are
63,795 and 63,559 bytes, every atomic validation checkpoint declares row
evidence in Zarr arrays, and none embeds an `arrays` object. Consolidating both
runs increased the canonical root metadata by only 220,141 bytes. Both ordinary
readers succeeded with dense hashing deliberately disabled, while explicit
deep audits independently recomputed and passed all 38 declared output hashes.

The canaries preserve identical frame, fish, track, chaser, and chaser-position
lineage. Detection provides 148,963 valid source-position frames; keypoint
triad provides 146,412, all of which overlap detection. Across those common
frames, the median source-position difference is 6.56 px, the 95th percentile
is 22.37 px, and nearest-chaser identity agrees for 99.943% of frames. This is
bounded single-recording canary evidence only. It neither averages providers
nor promotes a default; broader camera-, state-, and recording-stratified
validation remains open.

The failed oversized v1 publication attempts were archived with an inventory
and whole-file digest before their canonical children were removed. They are
not selector-visible evidence and must not be used as source authorities.
Required CI has not run for `bda296db`; the branch and canaries therefore remain
incomplete and non-production-authoritative under the integration contract.

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

### Implementation checkpoint: 2026-08-23 position-only suite

Commits `ac3103910bacc11a5bd12ac11fc26e87781f66f3` and
`8ba71230f8a9acbe2652f3e8b8a7d1b7094457ea` add a pure provider-aware
position suite and an atomic operational canary publisher. The suite requires:

- one exact selector-ineligible sealed provider-chaser-distance run;
- the exact epoch-v2 authority already bound by that provider source;
- explicit caller bindings from source window IDs to analysis roles;
- one exact reviewed arena-geometry selection and native-camera frame;
- one exact physical camera scale; and
- exactly one explicitly named treatment and baseline chaser role for paired
  role contrasts.

It computes exact half-open pre/training/post summaries, distance CDFs,
native-camera/+Y-down quadrant occupancy, area-corrected moving-chaser radial
occupancy, arena radius/wall distance, valid-time near-zone occupancy and
hysteresis entries, and treatment-minus-baseline scalar/radial contrasts. It
contains no motion, heading, body-frame, bout, gaze, trial, or escape inference.

The reviewed GoodBatBadBat canary is:

`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_position_suite_canary_20260823_v2`

- artifact-manifest SHA-256:
  `02de7583e6210c00a269a19505ac1c11d4efe5706ec22bc44308507fb3221910`;
- software commit:
  `8ba71230f8a9acbe2652f3e8b8a7d1b7094457ea` (clean worktree);
- source provider: `detection_bbox_centroid.v1`;
- source provider manifest SHA-256:
  `7caac391aed297ba215a763eb08fa315f7be23f7fb95508a4c42cab63de038ce`;
- selected geometry record SHA-256:
  `4eac1c3560463db623f8ad1e7182e0b30cbbbe0679d4fe23cecb3501832e2884`;
- disposition: selector-ineligible operational evidence, with no analysis-Zarr,
  registry, or production-authority mutation.

All eleven v2 artifacts independently revalidated against their manifest. The
first immutable v1 canary remains as audit evidence; visual review found that
its legacy 2--20 mm CDF ladder truncated the displayed distribution. V2 fixes
that presentation policy by deriving the full CDF threshold axis from the
selected arena radius plus the maximum bound chaser-center radius; it does not
alter the underlying per-epoch metrics.

The canary also records a strict equivalence policy for the stimulus-scoped and
recording-scoped physical-frame records. Only their path-scoped `frame_id` and
selected-evidence `record_ref` may differ. Camera identity, source-camera pixel
authority, selected-evidence digest, coordinate semantics, scale, and physical
extent must remain identical.

Focused validation passed `27` tests covering the new suite, provider distance
comparison, and fixed arena-grid authority. Required CI had not run. At this
single-recording checkpoint, the full-cohort position canary, sealed
analysis-Zarr publication, profile/registry projection, and production
integration were still open; the cohort item is superseded by the checkpoint
below.

### Implementation checkpoint: 2026-08-23 bounded position cohort

Commits `eb996682f1403f6001b3706b725b89dda5e4322d`,
`27eacdc62b52cbfc8b83be3c1d924ac7aded164b`, and
`909f28b1a159bcca6f6600b4f12edeeaac9327ff` add the bounded cohort planner,
strict frozen-task runner, recording-balanced aggregation, atomic operational
publication, and reviewed cohort plots.

Planning is deliberately separate from execution. The no-write `plan` command
resolved the same earliest/latest recording in each of arenas 1--4 used by the
provider-comparison campaign, then froze every exact sealed provider manifest,
reviewed arena-geometry selection, source-camera physical authority, camera,
and pre/training/post epoch-role binding. The resulting task SHA-256 is:

`21dd9f7079de39cac987442bf03a233fc57338b714e3c96cc080a74ca2d8da39`

The reviewed immutable cohort successor is:

`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_position_suite_cohort_canary_20260823_v3`

- artifact-manifest SHA-256:
  `1bb72ffda8f2dbf932005eed9cfe491f3ee43fc82265f219245ba7aa6123148d`;
- software commit:
  `909f28b1a159bcca6f6600b4f12edeeaac9327ff` (clean worktree);
- selection SHA-256:
  `07d350ad096c67e077ecb7aeae9197f7fa27300293a1407490317f113e5f8891`;
- scope: eight recordings, two temporal extremes per arena, and cameras
  `2010093`--`2010096`;
- output: 107 independently hash-verified artifacts, including eight complete
  per-recording evidence sets, recording/epoch/role tables, recording-balanced
  summaries, and four cohort figures; and
- disposition: selector-ineligible operational evidence, with no analysis-Zarr,
  registry, selector, or production-authority mutation.

Every epoch/chaser stratum contributes one value per recording to cohort
summaries; frames are never pooled across recordings. All 42 scalar
epoch/role/metric summaries contain eight recording-level observations. The
minimum valid-distance fraction across the 48 recording/epoch/role rows is
`0.9721035058430718`. Radial CSVs preserve all per-recording bins and explicit
support counts, while the reviewed plot uses
`complete_recording_support_only_v1` and therefore displays only bins with all
eight recordings.

Immutable v1 and v2 cohort attempts remain as audit evidence. Visual review of
v1 found that tail bins with partial cohort support were still displayed; v2
fixed that scientific display policy but had a clipped title. V3 retains the
support correction and fixes the title without altering scientific rows.

Focused outside-sandbox validation passed all 15 cohort and adjacent
position-suite tests. Required CI has not run. Sealed analysis-Zarr
publication and reduced profile/DAG integration are superseded by the
checkpoint below. Generalized plot recipes, selector/registry integration, and
at that checkpoint the motion/bout/body-frame/gaze/trial/escape phases remained
open. The later 2026-08-24 semantic checkpoint completes the recording-local
motion/bout epoch summary and its versioned cohort export/plot path only.

### Implementation checkpoint: 2026-08-23 sealed position-suite publication

Commits `0f533c6bc3a7146c0b6f90605b84d674c1937713`,
`e0de92e9eb007d18abd61c8a17e7d6d7b7330954`, and
`5be4d47bb33bbe8d6a5aa36a452f24bad30faeaf` add the immutable analysis-Zarr
publication, compact receipt correction, reduced profile, exact LSF workflow,
and bounded readiness receipt for the provider-aware position-only suite.

The selector-ineligible GoodBatBadBat publication canary is:

`analysis/provider_chaser_position_suite_runs/provider_chaser_position_suite_detection_bbox_centroid_canary_20260823_v1`

inside the canonical analysis Zarr for
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`. Its immutable manifest SHA-256
is `7c20874695a18e8d356aaca4347dcc8972cd42793553e85c38f00815c86ec65d`.
It contains 729 rows across seven typed tables and 148 arrays. The durable
publication result is:

`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_position_suite_publication_canary_20260823_v1/result.json`

The ordinary reader validates exact run identity, compact manifest and
declarations, array schemas and lengths, source bindings, and direct versus
consolidated metadata without rehashing upstream dense inputs. Deep array
rehashing remains an explicit audit mode. The canary changed no raw data,
upstream provider publication, selector, production authority, or registry.

`chaser_position_suite_v1` is intentionally reduced. It claims only stimulus
epochs and the sealed provider-position suite; it cannot imply motion,
body-frame, bout, gaze, trial, or escape readiness. Its exact two-job workflow
publishes one immutable candidate and then writes a bounded readiness receipt
outside the Zarr. That receipt truthfully reports the scientific candidate as
complete while production remains blocked by required CI and explicit
selector promotion. Registry projection remains ineligible and is not a job in
this DAG.

The exact canary was subsequently read through that strict path and produced
the external readiness receipt:

`/groups/johnson/johnsonlab/jeremy/operations/provider_chaser_position_suite_publication_canary_20260823_v1/readiness_receipt.json`

Its payload digest is
`ab2cf7d401cc5d99324eae79c7c7cb62c192fe45c40241de234d4e0b469367da`;
its status is `candidate_complete_production_blocked`; and it records exact
direct/consolidated equivalence for 149 declarations. This check did not write
inside the analysis Zarr or update a selector, registry, or production status.

Focused outside-sandbox validation passed 39 profile, computation,
publication, workflow, and readiness tests. Ruff, `py_compile`, and
`git diff --check` passed. Required CI has not run for this branch, so neither
the commits nor the canary are merge-ready or production-authoritative.

### Implementation checkpoint: 2026-08-25 composable successor operator

The remaining recording-local successors now have one no-write-by-default
operator:

`scripts/py -m fisheye.utils.materialize_composable_chaser_successors`

It loads exact immutable relative-frame, semantic-selection, provider-motion,
swim-bout, and reviewed eye/gaze inputs; expands dependency closure; prepares
controller-trial, generalized bout-response, escape/freeze, and gaze products;
and preflights or publishes only selector-ineligible immutable runs. It never
changes a selector, production authority, or registry row.

The first real eligibility trial used the 2026-08-12 arena-1 GoodBatBadBat
archive. Relative-frame and provider-motion authorities loaded exactly, but no
protocol-semantic selection publication or reviewed gaze source exists. The
receipt at `/tmp/composable_chaser_trial_20260825_v1.eligibility.json` therefore
reports `blocked_no_products` with four explicit module blockers and no archive
writes. This is useful operational evidence: source absence is now reported as
a bounded module/dependency result rather than a partial analytic publication.

The companion integration required for future Citrus snapshot-v2 recordings
is also implemented locally. It reloads the exact raw-H5/companion pair, checks
materialized execution bytes against that sealed raw H5, projects producer
half-open stimulus intervals onto acquisition rows, and requires exact mapped
CHASER phase envelopes. Reused current-input rows use the same deterministic
latest-stimulus policy as the chaser input-provenance proxy; all candidates,
reuse counts, and acquisition gaps remain evidence. The observed historical
ratio (215,987 stimulus frames to 179,885 unique camera frames) confirms this
many-to-one rule is required.

Focused outside-sandbox validation passes 105 successor/protocol integration
tests. `py_compile` and Ruff pass. Required CI, a real complete v6 companion,
the controlled four-camera hardware trial, reviewed gaze evidence, and a real
successful successor publication remain outstanding; this checkpoint is not
merge-ready or production-authoritative.

### Implementation checkpoint: 2026-08-25 exact-trial timed publication and plots

The 2026-08-12 arena-1 GoodBatBadBat archive now has a successful
selector-ineligible recording-local publication chain. Historical controller
trials are accepted only from exact strictly positive producer-logged
`chase_trial_id` values on active rows. Trial gaps remain evidence but are not
members; no contiguous-active or other fallback segmentation is allowed.

The relative-frame successor is:

`analysis/chaser_relative_frame_runs/chaser_relative_frame_keypoint_triad_cohort_20260825_exact_trials_session_time_v2`

Its immutable manifest SHA-256 is
`2bd71f856d49dfca3f83492d7ce6410017565eef9b15d5a6bbaa3c5e003f929d`.
It contains 149,887 frames, 299,774 frame-by-chaser rows, and 53 arrays. Session
time comes from the exact logged `timestamp_ns_session` shared by both chaser
records and sealed by the input-provenance proxy. The adapter requires exact
cross-chaser equality, strict increase, and exact equality to the proxy-sealed
timestamps. It neither derives time from camera start/stop markers nor
interpolates it. Exact provider-motion projection matched 148,996 relative
frames; the 891 unmatched relative frames are invalid/NaN evidence, and 32,549
provider-only frames are not injected into the relative timeline.

Three successors share the run name
`goodbatbadbat_chaser_successors_20260825_exact_trials_session_time_v2`:

- controller trials: four exact trials, manifest
  `6b35d78d74b3f3922025bab3a28af674fc02ea6d95a79b9fbf549f74dba7a110`,
  scientific payload
  `2d34642953dae4301960532d438b9c2dc247ad9d796d8e989255b8a84f9f208e`;
- generalized bout response: 1,673 bouts, 3,346 bout-by-chaser rows, and 30
  summary rows, manifest
  `1dc77b756a423131aaaec4ce4619c0117738b4edb6562e9d4a68f200ff83468e`,
  scientific payload
  `1bc77098b2f9e9d47b4bc525b00a8e568e9fd78002eae1412628b3ef9d8d22ac`;
  and
- escape/freeze: nine speed-threshold escape events, four trial rows, and 20
  threshold-sweep rows, manifest
  `e2281f7a807968bdbcc4e9a2c9324dba676b1decfe97a390054e7da97302a056`,
  scientific payload
  `8e69b7c47ed462a9b72a62b3c403411a9b952688d70e91389aa45838b0e229f8`.

All four exact trials belong to chaser identity 1 and have zero membership
gaps. Their response classes are freeze, escape, freeze, and other. The escape
trial contains nine events at the 20 mm/s primary threshold, has 19.16 seconds
of valid trial time, a first-event latency of 10.50 seconds, and all nine event
traces are usable. Trial 4 contains two bouts, but neither reaches 20 mm/s; its
threshold sweep retains two events at 10 mm/s, one at 15 mm/s, and zero at
20--30 mm/s. Freeze-window speed coverage is 100% for all four trials. These
are descriptive candidate results, not production conclusions.

The deeply audited dashboard and external receipt are under:

`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_chaser_successors_20260825_exact_trials_session_time_v2/plots`

The PNG SHA-256 is
`d94fbf63b1777c1377368402447753208694648e275c565240543ffa515c0feb`,
the PDF SHA-256 is
`28ea98e590d25397bf2e0a7dc3906864096aac76e6b333793702922e2963abba`,
and the receipt payload SHA-256 is
`5f409a29da44cc20b7b6bb20f8bc7bfe3349fb35a78607f6e28086d3bf3e2db8`.
The plotter resolves no selectors, deep-rehashes all plotted arrays, and
requires exact dependency payload digests. The earlier
`...exact_trials_v1` dashboard is timing-incomplete preliminary evidence and
is superseded for scientific review by this session-time-v2 dashboard.

This result remains deliberately reduced: no reviewed gaze/eye source or
body-frame extension was supplied, and the historical semantic selection uses
the conservative pending `STEP_END` policy until producer-authored v2 endpoint
evidence is available. No selector, registry row, or production authority was
changed. Required CI is still unrun, so neither the branch nor these candidate
artifacts are merge-ready or production-authoritative.

### Correction checkpoint: 2026-08-25 controller activity is not position validity

The exact-session-time v2 relative-frame run above has been scientifically
superseded. Its computation incorrectly required `chase_sequence_active` for
otherwise valid fish-to-chaser geometry. That erased all pre and post geometry
and all inert-chaser geometry during training: relative-position coverage was
0/0 rows per chaser in pre, 7,747/0 in training, and 0/0 in post despite finite
position coverage.

Current computation no longer contains an activity-gated position mode.
Controller activity remains a typed evidence surface for trial semantics, but
cannot invalidate finite selected occurring fish/chaser geometry. Result
construction fails closed unless the fixed policy is
`controller_active_is_orthogonal_position_evidence_v1`; the historical
`chaser_inactive` reason code remains only for reading and auditing already
sealed runs.

The corrected immutable relative-frame run is:

`analysis/chaser_relative_frame_runs/chaser_relative_frame_keypoint_triad_cohort_20260825_exact_trials_session_time_activity_orthogonal_v3`

Its manifest SHA-256 is
`c3a2229fe8fbcd24b5a4fbeb2dee1c672b5e2c59a615479e6c4f05ae93a3f556`.
Measured relative-position coverage per chaser is now exactly the finite fish
coverage: 59,789 of 59,964 frames in pre, 29,984 of 29,992 in training, and
58,302 of 59,930 in post.

The corrected radial/near-field successor is:

`analysis/chaser_radial_near_field_runs/goodbatbadbat_chaser_radial_near_field_20260825_exact_session_time_activity_orthogonal_v2`

Its manifest SHA-256 is
`af699f076df54bf4da2f99abc2f9db05f4ed694bfdb46fea74f82b8cddf1a1e1`
and scientific payload SHA-256 is
`620d0a4df457d0249080cdfbba3dbc752fd592f15ddb09c2f3e8f172be372589`.
All six epoch-by-chaser distance/radial series are present. Exact 5 mm/6 mm
hysteretic near-field integration finds no entries in pre or training, then
one post entry per chaser: 1.042 seconds of aggressive-chaser near dwell and
1.650 seconds of inert-chaser near dwell. The plot and deep-audit receipt use
the output stem
`goodbatbadbat_chaser_radial_near_field_exact_session_time_activity_orthogonal_v2`
under the recording's `operations/.../plots` directory.

The earlier radial v1 and the controller/bout/escape products bound to the
superseded relative-frame v2 remain immutable audit evidence but must not be
used as current scientific products. Their dependent successors therefore had
to be republished from relative-frame v3 before scientific review. No selector
or production authority was changed, and required CI remains unrun.

That dependent republish is now complete under the shared immutable run name
`goodbatbadbat_chaser_successors_20260825_exact_trials_session_time_activity_orthogonal_v3`:

- controller trials: manifest
  `8ae5c7b89f986d9480d045020361315f23c154207dafdaab2317dd093ddf87ca`,
  scientific payload
  `cfd52584cf003ef72a513ce351c31588c3d72b65fe14801fa83429eab1037910`;
- generalized bout response: manifest
  `d25e712ab3aac3e1cab29819433e4bf4d262d37a250143c07f5f7460b886a2da`,
  scientific payload
  `9612b6125a8b6df877090745f06223a2c13a19477f5e2ad9fb188583229165be`;
  and
- escape/freeze: manifest
  `8000476b554e4426abb71a79c10c65166d8b3f9621cc7636c63d8ec96330e276`,
  scientific payload
  `54f388f25704792542fe56c6705072a3c4251da5d09e6d91ac1cac6e6f1c7c86`.

The corrected dashboard is under
`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_chaser_successors_20260825_exact_trials_session_time_activity_orthogonal_v3/plots`.
Its PNG SHA-256 is
`b69b254e5c4ff47923d1370ff257db9b4fd8c9364ecd365d1ce5dabb0afb4d1e`,
its PDF SHA-256 is
`73cfb1cf86a20173b5703f6627675d6ae529c8cc4497b5a1131335c0f47bf0a1`,
and its receipt payload SHA-256 is
`cd9745ad00df59c9952889da3d0f93b5be05c6b3bbea328976e02fa9968d5baa`.
The plot was visually inspected and includes four exact trials, all six
epoch-by-chaser bout-response series, response-class counts, and threshold
sensitivity. The v2 dashboard is now superseded for scientific review.

### Detection-centroid canary and detailed plot bundle: 2026-08-25

Detection bounding-box centroids are now exercised as a first-class fish
position provider, not a legacy-only fallback. The immutable relative-frame
canary is:

`analysis/chaser_relative_frame_runs/chaser_relative_frame_detection_bbox_centroid_cohort_20260825_exact_trials_session_time_activity_orthogonal_v3`

Its manifest SHA-256 is
`f89f45b006b050587a6197933a085917236cce78eb95212debd0e929de2c3895`
and prepared payload SHA-256 is
`c7c9fd30efeb44f24083e16e0cc6d62a76c594ace6ab76d36bc95aca6873ee8f`.
Finite relative-position coverage per chaser is 59,964/59,964 frames in pre,
29,992/29,992 in training, and 59,905/59,930 in post. This exceeds the
reviewed-keypoint canary's coverage while retaining its provider identity as
separate evidence.

The corresponding radial/near-field successor is:

`analysis/chaser_radial_near_field_runs/goodbatbadbat_chaser_radial_near_field_detection_bbox_centroid_20260825_exact_session_time_activity_orthogonal_v2`

Its manifest SHA-256 is
`25581e49691ff600711aa34953619eca9517341a09330d2c6cc252f0779dca70`
and scientific payload SHA-256 is
`050724d60444440ba4ae4bae6ac7c5293b11abd6d9ec4c5ef4503a3ab4407206`.
The detection and keypoint providers both find no 5 mm near-field entries in
pre or training and one post entry per chaser. Aggressive-chaser post dwell is
identical at 1.041733052 seconds. Inert-chaser post dwell is 1.658356983
seconds from detections and 1.650024296 seconds from keypoints. Across all six
epoch-by-chaser strata, median fish--chaser distance differs by less than
approximately 0.8 mm. This is one-recording canary agreement, not a provider
equivalence claim.

The detailed immutable plotting recipe is implemented by
`fisheye.utils.plot_chaser_detailed_successors`. It accepts only explicit run
names and performs no selector resolution or Zarr mutation. Before comparing
providers it now requires both radial products to bind their corresponding
relative-frame products; matching semantic-selection and arena-geometry
authorities; matching coordinate and scale policy; and byte-identical chaser,
timestamp, occurrence, behavior-role, and semantic-selection arrays. The two
fish-position provider identities must remain distinct.

The visually inspected v2 bundle is under:

`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_chaser_successors_20260825_exact_trials_session_time_activity_orthogonal_v3/plots/detailed`

It includes PNG and PDF versions of the matched-provider distance CDFs,
generalized bout-response heatmaps, exact trial and escape/freeze details, and
exact-session-time trigger-aligned trial distance traces. The receipt payload
SHA-256 is
`fa5cce7193998ad3eb9d5ef781012221d1d91324b69475c57558cdf29eeef21a`
and its exact source-bindings SHA-256 is
`2de6cb98b4f185184327d3540483639ed2718140e3c41ebddc3cd3146b8e2724`.
The earlier detailed v1 bundle remains immutable but is superseded by v2
because v2 proves the complete comparison alignment contract before render.

The receipt also records plots that cannot yet be produced without inventing
evidence: individual near visits are not persisted by radial successor v1;
escape-onset-aligned distance traces are not persisted by escape successor v2;
ring-entry video is deferred to its sealed video successor; and gaze/bearing
remains blocked on reviewed body-frame and eye-orientation evidence.

Focused outside-sandbox validation passes 14 plotting/radial tests, including
rejection of mismatched chaser arrays and duplicate provider identities.
Required CI remains unrun. All runs and plot receipts are selector-ineligible
candidate evidence and do not change registry or production authority.

### Frozen 84-recording cohort execution plan: 2026-08-25

The recording-local successor chain now has a dedicated cohort planner/runner:

`scripts/py -m fisheye.utils.materialize_composable_chaser_successor_cohort`

The planner consumes a frozen registry JSON export and resolves mutable
recording-local authorities only during planning. It freezes the exact raw H5
stat identity, canonical stimulus run, keypoint and detection proxy manifests,
compatible motion/bout pair, reviewed arena-geometry selection and digest, and
physical-scale authority for every recording. Each input group's direct
metadata-file SHA-256 is retained and revalidated by the worker. The task has
one canonical payload digest and a contiguous one-based recording axis.

The current read-only GoodBatBadBat inventory contains 84 active recordings:
36 from 2026-08-10, 28 from 2026-08-11, and 20 from 2026-08-12. Thirty-six use
producer protocol hash
`9f27b4084f252eb7ce70bda1bd1056aa93c1d295d90ed65de213d0cc46b5c459`
and 48 use
`b0f637a8dbbb4fe064bc4c86b636dc78bde8d082c050510b5c833b6119e98081`.
All 84 have both first-class position proxies, exact reviewed geometry and
scale authorities, a canonical stimulus source, and a compatible provider
motion/swim-bout pair. Eighty-two use the talk-v2 motion/bout pair. The two
earliest recordings retain explicitly frozen compatible v1 pairs: arena 1
uses the talk-v1 pair and arena 2 uses the canary-v1 pair.

The frozen task generated during this checkpoint is
`/tmp/goodbatbadbat_composable_chaser_cohort_task_20260825.json`, with task
SHA-256
`0cb0a8b77d7f77e851b7f6543da5a760c42ab2743949476504eba7afd7dced8b`.
It reports 83 recordings ready for the complete chain and the deeply audited
arena-1 recording ready for plot-only reuse. The `/tmp` location is diagnostic
evidence, not a durable cluster handle; the submitter copies a revalidated
task into its durable run directory before submission.

Each worker serially performs, or exactly reuses after validation: historical
semantic stimulus publication; v1 compatibility and v2 immutable epoch
publication; protocol-semantic selection; keypoint and detection relative
frames; controller-trial, generalized bout-response, and escape/freeze
successors; both radial/near-field providers; paired-provider exact-epoch
spatial occupancy; and dashboard, detailed, and spatial-occupancy plot bundles.
Dynamic epoch-manifest identity is read only after the preceding immutable
publication completes. No `latest`, `latest_complete`, or other scientific
selector is resolved during execution.

The LSF renderer is:

`scripts/submit_composable_chaser_successors_bsub.sh`

It requires an absolute clean commit-pinned cluster worktree and full 40-character
Palette commit, copies and revalidates the task, and emits one array worker per
recording. Every worker refuses execution outside LSF, owns one entire
analysis Zarr, uses node-local scratch, writes only selector-ineligible
recording-local products and external plot/worker receipts, and never updates
SQLite or activates a selector. This path does not invoke the superseded
monolithic chaser-analysis submitter.

A real no-write resume check against task 81 revalidated and reused all nine
completed publication stages and planned only the two missing cohort-layout
plot recipes. A second real no-write check against task 1 rendered the full
11-stage chain with its exact v1 motion/bout exception. Combined focused
outside-sandbox validation passes 20 cohort, plot, and radial tests.

The original 11-stage implementation was committed as
`900c97ce5b90b4c2462c24433bab2afb596299df` and deployed in a clean,
commit-pinned cluster worktree. LSF array `153742886` completed 80 recordings
and failed closed on all four cameras from
`2026-08-12T21-14-36Z`. Those four archives encode legacy step 0 as
`[468, 30469]` while step 1 begins at `30469`; exact half-open endpoint
authority cannot be recovered from that equality. They retain the semantic
products that were safe to publish, but no relative-frame or dependent
successor was produced. The 80 successful receipts all report
`complete_selector_ineligible`, exact commit/task/safety bindings, and no
nonzero return code.

The original cohort publication produced 400 PNGs, 400 PDFs, and 160 external
plot receipts below:

`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_chaser_successors_20260825_exact_trials_session_time_activity_orthogonal_v3/cohort_plots`

Required CI remains unrun. These executions are experimental, immutable,
selector-ineligible evidence; they did not update SQLite, activate a selector,
or change production authority and are not merge-ready.

### Exact protocol-epoch spatial occupancy heatmaps: 2026-08-25

The missing per-epoch two-dimensional occupancy product is now a first-class
paired-provider successor rather than a reuse of the historical
detection-only occupancy surface. It is published below:

`analysis/chaser_spatial_occupancy_runs/goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260825_exact_epochs_v1`

The successor binds explicit immutable keypoint and detection relative-frame
runs, the exact protocol-semantic selection, and both reviewed-geometry-bound
radial successors. It requires identical acquisition-frame, timestamp,
semantic-selection, chaser, geometry, coordinate, and physical-scale
authorities while requiring the two fish-position provider identities to stay
distinct. It uses one shared 2 mm arena-centered physical grid, excludes
finite points outside the reviewed circular arena, and stores integer counts,
conditional valid-in-arena density, candidate-epoch-normalized occupancy,
coverage, invalid-position counts, and out-of-arena counts for exactly
`chaser_pre`, `chaser_training`, and `chaser_post`. Missing positions are not
interpolated. Exact semantic rows must all exist in the relative-frame source;
additional selected source rows outside the semantic epoch union remain
explicit evidence but are excluded from the epoch histograms.

The deeply audited plot recipe produces a 3-by-3 figure: keypoint and detection
occupancy rows plus a detection-minus-keypoint difference row, with pre,
training, and post columns on byte-identical physical bins. Titles report
whole-epoch coverage so conditional spatial normalization cannot hide missing
positions. PNG/PDF receipts bind the exact source manifest and scientific
payload and remain selector-ineligible.

Implementation commit `5b2bd068186281206b546c8a927b11d2ea83a14a` passed
Ruff, `py_compile`, `bash -n`, `git diff --check`, and 28 focused
outside-sandbox tests. A real no-write computation canary on task 81 produced a
42-by-42 grid with scientific payload
`5bdd96b8c014a7344263e1834392731ade2407eb5310994c3f5d7b1d0cb90a62`.
The selector-ineligible write/plot canary was LSF job `153743120`; it completed
in 53 seconds and its output was visually reviewed.

LSF array `153743122` then ran only the 80 scientifically resolvable indices
`1-76,81-84` from the frozen task SHA-256
`0cb0a8b77d7f77e851b7f6543da5a760c42ab2743949476504eba7afd7dced8b`.
All 80 elements completed `DONE`. Receipt audit found 80 expected and no
unexpected indices, 80 unique worker payloads, 13 stages per recording, exact
commit/task/safety bindings, and no nonzero return code. Plot audit found 80
unique source manifests and scientific payloads, 80 PNGs, 80 PDFs, and 80 plot
receipts. Independent rehashing matched all 160 output file sizes and SHA-256
digests. The plot root now contains 480 PNGs, 480 PDFs, and 240 plot receipts
across the five dashboard/detailed figures plus the new spatial-occupancy
figure for each successful recording.

The array receipts are under:

`/groups/johnson/johnsonlab/jeremy/operations/goodbatbadbat_spatial_occupancy_followup_20260825/composable_chaser_successors_goodbatbadbat80_occupancy_5b2bd068_20260825`

The four ambiguous indices `77-80` were never submitted to the occupancy
follow-up. This evidence remains selector-ineligible and required CI remains
unrun.

### Deferred publication-performance successor

- [ ] Replace repeated whole-archive publication work with independently
      consolidated immutable run stores (or equivalent subtree metadata
      generations) plus one small atomic selector manifest.
- [ ] Emit a content-addressed publication receipt bound to the exact source
      manifest, candidate physical inventory, schema/tool version, and
      consolidated metadata generation. Permit later consumers and interrupted
      retries to reuse it only after revalidating those bindings.
- [ ] Evaluate reflink or storage-native immutable copies and safe batching of
      candidate publications before one final visibility update.
- [ ] Make strict relative-frame source validation expose a targeted,
      streaming scientific-array audit mode for consumers that require only
      frame-axis position/selection evidence. Preserve the exact manifest and
      array-content proofs without materializing all frame-by-chaser arrays in
      memory. Occupancy task 2 (`2026-08-10T17-20-55Z_arena_2_goodbatbadbat`)
      completed correctly but took 955 seconds and peaked at 11,477 MB while
      the other 79 follow-up tasks completed much sooner; this is optimization
      evidence, not permission to weaken deep validation.

This optimization is deliberately outside the plot-critical path. The
2026-08-25 epoch-v2 canary spent 1.7 seconds rematerializing the scientific
payload but about 82 seconds planning/staging and 94 seconds in atomic
publication, including 66 seconds in archive-root metadata finalization.
Optimization must preserve fail-closed identity, direct/consolidated metadata
equivalence, and selector isolation.

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
