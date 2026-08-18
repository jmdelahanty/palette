# Provider-Aware Spatial Analytics Implementation Checklist

<!-- contract-meta
status: accepted-implementation-checklist
decision_date: 2026-08-18
implementation: pending
promotion_status: selector-ineligible-only
-->

Purpose: implement a reusable stimulus-selection, trajectory, occupancy,
contrast, cohort, and plotting foundation over explicit subject-position
providers. The first canary compares detection- and keypoint-derived position
without making either provider a new scientific default.

This checklist narrows the next implementation slice of:

- [Composable Stimulus Analysis and Plot Recipe Design](composable_stimulus_analysis_and_plot_recipes_design.md);
- [Position, Body-Frame, and Motion Provider Design](position_body_frame_and_motion_provider_design.md);
- [Subject Position Storage Contract v1](subject_position_storage_contract_v1.md); and
- [Derived Analysis Run Contract](derived_analysis_run_contract.md).

Existing immutable stimulus, detection, position, tracking, occupancy, motion,
and visualization runs remain unchanged. New publications are immutable
successors or new run families. No production selector, registry authority, or
provider default may change during this checklist's canary phases.

## Accepted first-slice decisions

- [x] Start with `detection_bbox_centroid.v1` and
      `keypoint_anatomical_triad_mean.v1` as two separate explicit offers.
- [x] Keep component-mask and subject-body-mask position providers out of the
      first analytics canary while new mask labels and model evidence are being
      developed. Preserve their existing runs and comparison evidence.
- [x] Use one valid tracked subject sample per acquisition frame as the
      scientific occupancy sample unit. Never count arbitrary raw detection
      rows as independent exposure samples.
- [x] Require an exact observation-to-track-sample projection. Duplicate valid
      subjects in a single-subject recording fail closed; they are not reduced
      with first-row, highest-confidence, or mean-position heuristics.
- [x] Keep position and heading independent. Detection-derived trajectory,
      occupancy, speed, acceleration, and bouts may be computed without a
      heading provider.
- [x] Do not join the 264 detection-only canary observations to the keypoint
      body-frame rowset by cardinality or nearest row. Each position provider
      owns its exact tracking and linear-motion lineage.
- [x] Use source-camera pixels only as an explicitly labeled diagnostic
      coordinate product. Scientific cross-recording occupancy uses an exact
      selected camera/arena transform and a persisted arena-millimetre grid.
- [x] Fail closed for the scientific millimetre product when scale, extent,
      coordinate, selected geometry, or transform authority is missing or
      stale. Do not infer scale from raster dimensions or dish diameter.
- [x] Treat every stimulus step as an atomic state. `SOLID_BLACK` does not
      imply `baseline`, `pre`, or `post`; saved compositions assign those roles.
- [x] Materialize exact frame membership before calculating a metric. Metrics
      and renderers do not independently re-resolve stimulus steps.
- [x] Keep scientific normalization separate from display normalization.
      Per-panel maximum scaling is never an input to a contrast.
- [x] Make the recording-balanced cohort view primary for cohort comparison.
      Pooled-frame products remain separately labeled descriptive outputs.
- [x] Keep mask-derived heading, body-mask heading, gaze, turn-toward,
      circling, and provider promotion outside this implementation slice.

## Phase 0: freeze the implementation contracts

### Selection and frame-set contract

- [ ] Name and version the immutable selection-expression and resolved-frame-
      set schemas without changing `stimulus_epoch_runs` v2.
- [ ] Bind every selection to one exact recording, stimulus run, acquisition
      frame domain, source-video metadata record, acquisition-clock authority,
      and source metadata digest.
- [ ] Support exact atomic-step references and exact interval-annotation
      references. Persist predicate text only together with its concrete
      resolved members.
- [ ] Support the narrow v1 expression vocabulary: exact member,
      `union`, `intersection`, and `difference`.
- [ ] Represent all intervals as ordered, de-duplicated, half-open acquisition
      frame intervals `[start_frame, end_frame)`.
- [ ] Define overlap behavior explicitly: a resolved frame contributes at most
      once to a pooled metric while all source-membership evidence is retained.
- [ ] Support `keep_occurrences` and `pool_intervals` as distinct aggregation
      policies. Preserve occurrence identity in either case.
- [ ] Support directional leading and trailing trims. For the existing
      nominal-frame-clock v1 policy, remove `ceil(seconds * fps)` frames and
      persist requested seconds, effective frame count, and rounding policy.
- [ ] Reject negative trims, trims that invert an interval, incompatible
      timelines, unresolved predicates, and unsupported expression operators.
- [ ] Make roles such as `baseline`, `treatment`, and `control` explicit saved
      metadata. Never infer them from step mode, order, or display label.
- [ ] Canonicalize and digest the requested expression independently from the
      resolved frame set so stale stimulus resolution is detectable.

### Provider-track binding contract

- [ ] Define a typed input handle for one exact
      `analysis/subject_position_runs/track_sample/<run>` or for an exact
      observation-position run plus its immutable observation-to-track
      projection.
- [ ] Require exact `track_sample_key`, acquisition frame, subject/track
      identity, coordinate descriptor, provider ID, estimator digest, source
      manifest, and recording-timing authority.
- [ ] Require uniqueness of `(subject_track_identity, acquisition_frame)` for
      the first single-subject profile.
- [ ] Preserve provider-present, provider-valid, in-selection, transform-valid,
      and in-grid states separately.
- [ ] Publish or bind selector-ineligible track successors for both detection
      and keypoint position. Do not reuse a body-frame source as position
      evidence.
- [ ] Reject implicit provider fallback, selector lookup, same-length joins,
      reordered keys, duplicate keys, stale manifests, and cross-recording
      composition.

### Scientific spatial-grid contract

- [ ] Freeze one versioned GoodBatBadBat arena-millimetre grid profile before
      writing a canary. Record exact x/y edges as float64 arrays.
- [ ] Choose the fixed grid extent and bin width from declared arena geometry
      and bounded canary evidence, not the observed position minima/maxima.
- [ ] Define bin membership as left-closed/right-open, with the final outer
      edge inclusive. Persist the edge policy.
- [ ] Record the selected arena geometry, physical scale, camera-to-arena
      transform, coordinate descriptor, and every authority digest used to
      project source positions.
- [ ] Record out-of-grid finite samples separately. Do not clip them into edge
      bins or silently expand the grid.
- [ ] Give any camera-pixel diagnostic grid a different policy ID and prevent
      it from entering millimetre-grid cohort contrasts.

## Phase 1: pure composable selection compiler

- [ ] Implement pure schema models, canonical JSON, and digest helpers for
      atomic references, annotations, expressions, resolved intervals, source
      memberships, occurrences, and assigned roles.
- [ ] Generalize `resolved_epoch_selection` through the new compiler while
      retaining its current compatibility behavior for maintained epoch-v2
      runs.
- [ ] Resolve exact canonical stimulus steps rather than relying only on
      `pre_event`, `training_event`, and `post_event` aliases.
- [ ] Make compilation deterministic under equivalent input mapping order.
- [ ] Persist both requested and resolved selection representations.
- [ ] Add a mixed `SOLID_BLACK -> CHASER_PRESENTATION -> SOLID_BLACK` fixture
      with distinct `black_before`, `chaser`, `black_after`, and `all_black`
      compositions.
- [ ] Prove that the two black steps remain separate occurrences unless a
      saved expression explicitly pools them.
- [ ] Add tests for boundaries, gaps, trims, empty results, overlap,
      de-duplication, set algebra, role preservation, and stale-source
      rejection.

## Phase 2: generic trajectory product

- [ ] Define one versioned selector-ineligible trajectory schema over the
      `track_sample` row axis.
- [ ] Materialize exact source track-sample indices/keys, acquisition frames,
      occurrence and selection membership, provider position, arena-mm
      position, and validity/reason evidence.
- [ ] Keep trajectory position unsmoothed in v1. Any smoothed trajectory is a
      separately identified derived method.
- [ ] Preserve selected expected-frame count, source-row count, valid-position
      count, transform-valid count, in-grid count, and missing/invalid counts.
- [ ] Bind exact position-provider, track projection, selection, timing,
      geometry, transform, and software identities in a content manifest.
- [ ] Publish one detection-position and one keypoint-position trajectory for
      the same GoodBatBadBat selections without selecting a winner.
- [ ] Validate source preservation and direct/consolidated metadata equality.

## Phase 3: provider-aware occupancy v2

- [ ] Move scientific histogram computation out of
      `visualization/plot_detection_epoch_heatmaps.py` into a provider-neutral
      analysis module. Keep existing occupancy-v1 readers and runs unchanged.
- [ ] Define a new provider-neutral run family and schema; do not label it
      `detection_occupancy` when it accepts other position providers.
- [ ] Consume only an exact validated trajectory/track-sample input and exact
      resolved frame set.
- [ ] Materialize, per occurrence and pooled selection:
      - raw bin counts;
      - valid in-grid sample count;
      - occupancy fraction of valid in-grid samples;
      - occupancy time in seconds under the bound timing policy;
      - expected selected frames;
      - provider-present and provider-valid counts;
      - transform-invalid and out-of-grid counts; and
      - exact x/y grid edges.
- [ ] Exclude invalid or non-finite provider positions from spatial bins and
      the occupancy-fraction denominator, while reporting their coverage
      against all expected selected frames.
- [ ] Do not interpolate missing positions, substitute another provider, clip
      finite out-of-grid points, or normalize each panel by its maximum.
- [ ] Require count conservation:
      `sum(bin_counts) == valid_in_grid_sample_count`.
- [ ] Require fraction conservation within floating tolerance when the
      denominator is nonzero; define all-zero/NaN behavior for an empty valid
      selection explicitly.
- [ ] Bind every position, tracking, selection, geometry, transform, timing,
      grid, validity, and configuration source by exact path and digest.
- [ ] Publish detection and keypoint occupancy canaries for identical saved
      `black_before`, `chaser`, and `black_after` selections and the same
      millimetre grid.

## Phase 4: strict recording-level occupancy contrasts

- [ ] Implement a narrow v1 contrast algebra with `difference` as the first
      operation: `treatment occupancy_fraction - baseline occupancy_fraction`.
- [ ] Require named arms and preserve every contributing selection, role,
      occurrence, source step, and source occupancy manifest.
- [ ] Require both arms to agree on provider run and estimator, track-sample
      policy, coordinate frame, transform, geometry, grid edges, denominator,
      normalization, recording, subject, and timing authority.
- [ ] Reject ordinary scientific contrasts between detection and keypoint
      providers. Cross-provider sensitivity belongs in an explicitly labeled
      comparison product.
- [ ] Publish `chaser - black_before` and `black_after - black_before` canary
      contrasts separately for detection and keypoint position.
- [ ] Store result arrays and exact references; do not overwrite, average, or
      duplicate the immutable source occupancy runs.
- [ ] Test rejection for mismatched provider, source run, grid, extent,
      coordinate frame, geometry, sample unit, denominator, overlap policy,
      timing, and stale lineage.

## Phase 5: cohort products

- [ ] Define one recording-level scalar/vector summary contract suitable for
      cohort concatenation without reopening plot artifacts.
- [ ] Freeze the cohort input manifest before campaign submission. Bind every
      recording, subject-track unit, provider, metric run, selection, contrast,
      and manifest digest.
- [ ] Treat `(recording_identity, subject_track_identity)` as the first
      recording-balanced experimental unit. A repeated or erroneous subject
      UUID must not silently collapse distinct recordings.
- [ ] Publish pooled-frame descriptive occupancy separately from the primary
      recording-balanced cohort product.
- [ ] Require a stated aggregation policy for unequal valid-frame coverage.
      Report coverage and do not silently reweight recordings by frame count.
- [ ] Keep detection and keypoint cohort products separate and comparable by
      exact provider label. Do not average providers.
- [ ] Export tidy per-recording/per-subject tables with metric values,
      validity, coverage, selection roles, provider identity, and source
      digests.
- [ ] Add campaign accounting for expected, succeeded, failed, blocked,
      missing, and stale recording products before cohort publication.

## Phase 6: plot recipes and Marimo inspection

- [ ] Define canonical plot-recipe JSON with independent scientific-analysis
      and render signatures.
- [ ] Implement generic trajectory, occupancy-panel, and occupancy-contrast
      recipes over exact immutable products.
- [ ] Keep labels, colormap, facets, panel order, and display scaling in the
      recipe rather than the scientific arrays.
- [ ] Generalize chaser markers into an optional annotation provider over the
      resolved frame set. Bind semantic behavior labels independently from
      display color.
- [ ] Recreate the current pre/chaser/post occupancy presentation as a recipe
      over the generic products rather than a second occupancy computation.
- [ ] Publish immutable plot-artifact attempts with recipe, PNG/spec/media
      hashes, source product digests, and consolidated metadata validation.
- [ ] Add recording-level Marimo discovery for exact selector-ineligible
      detection and keypoint offers, trajectories, occupancy panels, and
      contrasts.
- [ ] Show expected frames, valid coverage, out-of-grid count, provider ID,
      selection membership, grid policy, and selector/promotion status beside
      every plot.
- [ ] Prevent a read-only visualization or annotation failure from mutating a
      scientific product or production selector.

## Phase 7: provider sensitivity for motion and swim bouts

- [ ] Materialize or bind detection- and keypoint-position track successors
      with identical tracking policy where scientifically compatible.
- [ ] Compute speed, path length, and acceleration from each provider's own
      position track and exact timing authority.
- [ ] Run the same versioned bout-segmentation policy independently over each
      compatible provider-motion run.
- [ ] Compare coverage, speed and acceleration distributions, bout count,
      duration, path length, peak speed, inter-bout intervals, and
      pre/chaser/post contrasts without selecting a provider.
- [ ] Preserve algorithm identity separately from provider identity so a
      changed position source is not mistaken for a changed bout algorithm.
- [ ] Do not require or synthesize heading for these linear-motion analyses.
- [ ] Block rather than join keypoint body-frame rows onto detection-only
      observations. Heading-dependent comparisons remain deferred.
- [ ] Add recording- and camera-stratified checks for provider-dependent bias
      before any future default-promotion discussion.

## Canary acceptance

Use `2026-08-10T17-20-55Z_arena_2_goodbatbadbat` first because its four
position providers and immutable provider-comparison evidence already exist.
This checklist's canary uses only detection and keypoint position.

- [ ] Preflight exact source manifests, selector-ineligible state, coordinate
      graph, geometry, transform, acquisition clock, stimulus run, and frame
      domain without writing.
- [ ] Freeze the selection specs and millimetre-grid profile in the canary
      plan before materialization.
- [ ] Prove detection and keypoint calculations use identical resolved frame
      sets and grid edges while retaining their own validity and row lineage.
- [ ] Review trajectories and occupancy panels against recording playback for
      representative pre-, chaser-, and post-period frames.
- [ ] Quantify coverage and position-provider sensitivity for occupancy,
      contrasts, speed, acceleration, and bouts.
- [ ] Check for systematic step/state-dependent disagreement; do not replace
      this with review of weak mask-model disagreements.
- [ ] Record exact canary run IDs, manifests, Palette commit, commands, test
      results, and a timestamped decision.
- [ ] Keep all canary runs selector-ineligible and leave production/default
      provider policy unchanged.

## Production and integration gates

- [ ] Add focused pure/in-memory tests before real-Zarr integration tests.
- [ ] Run real-Zarr and Marimo checks outside the Codex sandbox according to
      `AGENTS.md`.
- [ ] Preserve existing occupancy-v1, GoodCopBadCop compatibility, chaser, and
      provider-canary tests.
- [ ] Validate immutable retries, manifest tampering, stale lineage, direct vs
      consolidated reads, and final consolidated visibility.
- [ ] Update the storage-contract catalog, analysis-offer capability registry,
      recording-local discovery, and registry projection only after the
      scientific run contracts are stable.
- [ ] Keep parallel workers away from SQLite. If registry projection is added,
      use immutable receipts and one dependent serial finalizer.
- [ ] Pass every required CI check before integration, shared-checkout update,
      production selector activation, campaign publication, or any claim of
      merge readiness.
- [ ] Require a separate timestamped provider-promotion decision after
      multi-recording evidence. Successful materialization alone cannot make
      keypoints or detections the GoodBatBadBat default.

## Explicitly deferred

- Retraining, promoting, or scientifically adjudicating the current subject-
  mask position providers.
- Visual disagreement review for weak current mask predictions.
- Component-mask-triad or whole-body-mask production analytics campaigns.
- Mask-derived anatomical heading and full-body-mask heading.
- Gaze, turn-toward, circling, predicted-miss, and other heading-dependent
  escape analyses.
- Arbitrary user-defined formulas, weighted anatomical points, arbitrary
  contrast expressions, and implicit provider fallback.
- Production provider selection or an 84-recording provider campaign before
  the canary, required CI, and a separate promotion decision are complete.

## Completion evidence to record

- Exact branch and commit.
- Exact source and output run paths and manifest digests.
- Selection-expression and resolved-frame-set digests.
- Position provider, tracking, timing, geometry, transform, and grid policy
  identities.
- Expected, present, valid, transformed, in-grid, and missing sample counts.
- Count/fraction conservation results.
- Detection-versus-keypoint occupancy, contrast, motion, and bout sensitivity.
- Focused, adjacent, integration, Marimo, and required-CI commands/results.
- Confirmation that source recordings, immutable inputs, existing analysis
  runs, selectors, registry authority, and provider defaults were not changed.
