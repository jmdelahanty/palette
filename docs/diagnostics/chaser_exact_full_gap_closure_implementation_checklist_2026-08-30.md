# Exact chaser analytics full gap-closure implementation checklist

<!-- contract-meta
version: 3
status: implementation-in-progress
implementation: phase-2-epoch-and-alignment-local-validation
last_verified: 2026-08-30
audited_commit: c89d689c36bc6088e5abd5495861c533caa25649
-->

## Purpose

Close every currently identified gap between the receipt-bound exact chaser
successors, the established chaser scientific products, recording-local
Marimo inspection, static publication, and cohort execution.

This document is an implementation plan, not evidence that the listed work is
complete. It was reconciled against the active PR-78 worktree after eleven
parallel read-only Luna xhigh audits covering:

- epoch behavior;
- fish heading and alignment by distance;
- event-aligned escape-distance trajectories;
- individual near-field visits;
- ring-entry static views and media;
- response regimes;
- persisted-but-unmounted views;
- full-profile and static-artifact composition;
- shared contracts and receipts;
- cohort execution; and
- dependency ordering.

The fastest safe path is intentionally split into three kinds of work:

1. mount arrays that are already sealed;
2. compose an existing scientific producer with a small new exact extension;
3. add a new immutable successor only where identity-bearing rows or samples
   are currently discarded.

## Validated Core Behavior composition addendum — 2026-08-31

The full-profile work in this checklist must not become a plot-specific or
monolithic data authority. The recording-level composition direction is now
documented in
[`validated_recording_behavior_composition_design_2026-08-31.md`](../validated_recording_behavior_composition_design_2026-08-31.md).

That design distinguishes exact source authorities, independently governed
scientific successors, validation receipts, composition receipts, and
render-only projections. It proposes one small digest-bound validated
recording-behavior bundle that can expose Core Behavior and exact chaser
capabilities to any number of visualization adapters without copying their
scientific arrays.

For the current GoodBatBadBat workflow, the intended initial composition uses
the exact chaser-bound keypoint-provider motion and its same-track swim bouts as
the corresponding Core Behavior source, while retaining detection centroids as
a separately identified first-class position provider. Eye, gaze, and tail
remain optional exact capabilities; they are never attached through an
independent `latest` choice. The 2026-08-31 historical gaze batch remains
unaccepted because of upstream segmentation quality and must appear as a scoped
unavailable/review-required capability rather than invalidating unrelated
chaser products.

Phase 8 below should therefore consume or extend that generic composition
interface. The existing full-chaser-profile applicability envelope remains a
readiness/module-status view over independently sealed products; it must not
duplicate their arrays, replace child receipts, or become a new numerical
source.

The first interface slice is now implemented in
`fisheye.analysis_workflows.validated_recording_behavior_bundle`, with the
explicit-input planner at
`fisheye.utils.plan_validated_recording_behavior_bundle`. A read-only
GoodBatBadBat smoke resolved 10 exact source bindings, 11 scientific children,
17 complete capabilities, and four explicitly unavailable mask/gaze/tail
capabilities. The final smoke was in-memory only and also proved exact
direct/consolidated metadata equivalence for the provider-motion subtree. An
earlier `/tmp` specimen predates that binding and is obsolete diagnostic
evidence; durable materialization, consumer migration, commit, required CI,
and any deployment or promotion remain unchecked work.

## Implementation addendum — 2026-08-30

The first reader-only slice is implemented locally on
`agent/palette/chaser-gap-closure-phase1-20260830`, stacked on audited commit
`34e87b945b6224090be7ba759664865782a44484`. It does not create or migrate a
scientific successor. The local work adds:

- receipt-targeted distance CDF, ordinary/geometric radial-mass, configured
  wall-excluded, and selection-index views;
- valid-row and candidate-row same-quadrant views;
- valid-in-arena versus candidate-epoch occupancy modes at exact 2 mm and
  aligned exact-count-summed 4 mm display resolution;
- a conditional anatomical fish-heading route, one sample per acquisition
  frame, with no motion/detection fallback; and
- explorer spec v8 with exact consumed-array, bin, denominator, body-authority,
  and display-parameter provenance.

Static-publisher parity for the new geometric/wall-excluded, same-quadrant,
candidate-normalized, and fish-heading views remains incomplete. Full required
CI and integration are also still outstanding; this addendum is not
merge-readiness evidence.

Local validation on 2026-08-30:

- 134 passed and 15 expected failures across the exact-chaser receipt,
  source-handle, successor, explorer, and current static-publication regression
  set;
- changed-module Ruff and `git diff --check` passed; and
- `scripts/py -m marimo check apps/marimo/palette_explorer.py` passed outside
  the sandbox.

Real receipt-bound recording canary on 2026-08-30:

- recording:
  `2026-08-10T17-20-55Z_arena_1_goodbatbadbat`;
- exact composition receipt SHA-256:
  `1e3299e10710a02386ea8f7a65d8273236d1d9231a9194ee27c49f6924a9afc3`;
- exact spatial bundle:
  `analysis/chaser_spatial_occupancy_runs/goodbatbadbat_chaser_spatial_occupancy_keypoint_detection_20260827_body_frame_projection_receipt_bound_v4`;
- explorer spec schema version: 8;
- `radial_near_field`: four figures, both radial children targeted-rehashed;
- `distance_distributions`: four figures, all required CDF, ordinary,
  geometric, wall-excluded, and selection-index arrays targeted-rehashed;
- `same_quadrant_occupancy`: two figures, both denominator surfaces and counts
  targeted-rehashed;
- `spatial_occupancy`: one 21-trace figure, spatial and relative-frame arrays
  targeted-rehashed; and
- `fish_heading`: one four-panel figure, exact body source-row and heading
  arrays targeted-rehashed on the keypoint relative child only.

Every figure passed Plotly JSON validation. Per-route receipt-bound load time
was 1.29-3.74 seconds; render time was 0.01-0.12 seconds. No selector was read
or changed and no scientific successor was recomputed.

### Phase 2/3 implementation addendum — 2026-08-30

The next local slice is implemented on
`agent/palette/chaser-epoch-alignment-phase2-20260830`, stacked on
`c89d689c36bc6088e5abd5495861c533caa25649`. It adds two independently
governed exact children rather than expanding one shared lineage:

- the existing semantic-v2 provider epoch-behavior product now has a strict
  source handle, semantic-v2-only admission, filtered-speed publication guard,
  exact discovery, receipt-targeted array roster, Marimo route, provenance,
  exact child receipt, and cohort stage;
- `palette.chaser.body_alignment_by_distance.v1` persists frame evidence and
  semantic-epoch x chaser x exact 5 mm distance-bin summaries from the base
  keypoint distance surface and accepted anatomical body frame;
- the alignment child explicitly retains distance-invalid/body-valid evidence,
  forbids shortest-axis truncation, body-origin distance substitution,
  interpolation, viewer rebinning, and motion-heading fallback;
- static PNG/PDF alignment publication and the Marimo alignment route share
  `validate_persisted_body_alignment_summary`, so their bins, support, and
  statistics enter through the same conservation checks;
- each alignment artifact receipt binds the exact child manifest/payload,
  targeted verified array roster, full bin/angle/denominator/display recipe,
  and output hashes; and
- exact projection receipts use closed versions v5-v8 for alignment-only,
  gaze+alignment, epoch+alignment, and gaze+epoch+alignment compositions.

The explorer spec is version 10 and conditionally discovers exactly one
source-compatible alignment child. The cohort runner now materializes and
seals semantic-v2 epoch behavior and body alignment, composes them into the
versioned epoch+alignment projection receipt v7 without overwriting an older
composition receipt, and produces the receipt-bound static alignment bundle.

The widened affected exact-chaser regression set passes locally (159 tests),
along with changed-module Ruff, `py_compile`, `git diff --check`, and Marimo
validation. A read-only receipt-bound dry run against
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat` also passed without writing the
archive: 149,936 acquisition frames expanded to 299,872 exact frame-chaser
rows, then 90 summary rows across three semantic epochs, two chasers, and
fifteen zero-anchored 5 mm bins. The planned scientific payload SHA-256 is
`e8dd52bf3751f9a747b0ede9d857de139ac88536e1a3764ff45c7cf068d49d8b` and
the distance-bin recipe SHA-256 is
`2021219f051e95d529c0b00cadf286a7e5117ea5697eba9bcc34095f22dff3a0`.

That read-only plan is validation evidence, not a published canary. A real
immutable recording canary, full required CI, cross-recording alignment
export, and the separate epoch-spatial child remain outstanding. This
addendum is not merge-readiness, deployment, selector, or
production-authority evidence.

## Definition of complete

A product is not closed merely because an algorithm or plot exists. Every
product in scope must satisfy all three layers:

- scientific product: versioned persisted arrays or tables with exact identity,
  denominators, validity, parameters, and source lineage;
- immutable publication: manifest, payload digest, independent child receipt,
  selector-ineligible status, and exact composition binding;
- consumer surface: receipt-bound Marimo route, static artifact or report
  registration, provenance display, and fail-closed tests.

A product may be described as cohort-ready only after its recording-local
contract, export row unit, clustering policy, frozen membership, canary, and
required CI checks are complete.

## Audited state and closure decision

| Product | Audited state at 34e87b94 | Closure decision |
|---|---|---|
| Distance CDF and distributions | CDF, radial observed mass, geometric expected mass, and 5 mm wall-excluded twins are already sealed | Reader/render-only package; no successor migration |
| Candidate-epoch-normalized occupancy | occupancy_fraction_candidate_epoch is already sealed and receipt-rostered | Extend the existing spatial route with an explicit normalization mode |
| Same-quadrant scalar occupancy | Valid- and candidate-denominator fractions are already sealed | Add a panel or route; no successor migration |
| Full fish-by-chaser quadrant matrix | Computed in the provider suite but not sealed by the exact radial child | Add a versioned quadrant-summary successor to close configured-zone parity |
| Fish heading | Exact sparse body-frame projection and body_heading_deg are implemented, but the exact viewer requests only bearing arrays | Add a separate closed heading roster and conditional route so older bearing-only capability remains explicit |
| Alignment by distance | Framewise bearing exists; the established viewer rebins scientifically in Marimo | Add a sealed epoch x chaser x distance-bin summary; Marimo renders it without rebinning |
| Epoch speed, bouts, and IBI | Semantic-v2 provider materializer already exists | Integrate that producer; do not rewrite epoch/bout math |
| Epoch wall and center behavior | Explicitly omitted from semantic-v2 because it lacks a selected position/geometry authority | Add a separate exact position-and-geometry-bound epoch-spatial successor |
| Event-aligned escape trajectories | Escape v3 computes event x reference x time internally but persists only reduced medians | Add immutable escape-events v4 with aligned event samples |
| Individual near-field visits | Existing scientific products retain only aggregates; visualization reconstructs visits privately | Add immutable visit rows plus ragged aligned samples |
| Ring-entry static views | Useful legacy renderer exists but consumes unsealed reconstructed visits | Render only from the new visit successor and publish receipt-bound artifacts/specs |
| Ring-entry video/media | Existing animation is synthetic and has no durable media/frame-map receipt | Add a separate immutable media publication contract after visit identity |
| Response regimes | v1 producer exists, but no exact Marimo/export path and state semantics are incomplete | Mount v1 only as labeled descriptive compatibility; complete with a v2 state/transition successor |
| Full-profile readiness | Full-profile envelope and applicability planner exist but are not in the maintained exact runner/viewer | Bind them into execution and add a readiness/module-binding route |
| Static artifact gallery | PNG machinery persists useful metadata, but the generic reader does not verify the full selected scientific binding | Add receipt-bound artifact verification and provenance |
| Legacy pre/post polar snapshots | Historical reference output exists; exact body-bearing views supersede its scientific purpose | Keep as a verified reference artifact, never revive raw legacy discovery |
| Debug/details tables | Historical explorer exposed useful evidence tables; exact provenance is narrower | Add receipt-bound diagnostic tables to provenance, not a second scientific loader |
| Gaze tracking | Conditional exact route exists on this branch | Treat missing eye capability as applicability evidence, not as a fallback or new gap |

## Safety invariants

- [ ] Every source is an exact immutable run path plus manifest or payload
      digest. No latest, latest_complete, selected, authoritative, default,
      sorted-child, or raw-child fallback is permitted.
- [ ] Preserve independent child receipts because position, body frame, motion,
      trials, bouts, events, visits, and media have different lineages.
- [ ] A composition receipt binds those children; it does not replace their
      individual receipts or become a cache/selector authority.
- [ ] New schema meaning is additive. Preserve v1/v3 artifacts and readers as
      explicit compatibility surfaces; never reinterpret or overwrite them.
- [ ] Marimo is read-only and render-only. It may filter, select, and perform a
      bounded display projection, but must not assign trials, segment visits,
      rebin scientific summaries, classify regimes, interpolate rows, or
      recompute scientific denominators.
- [ ] Missing acquisition rows remain missing. Sparse body-frame rows project
      by exact acquisition frame with no interpolation or nearest-row fill.
- [ ] Detection/bounding-box centroids remain a first-class position provider.
      They are not a body-heading provider and must not acquire a velocity,
      mask, or travel-direction heading fallback.
- [ ] Body heading is anatomical and keypoint/body-frame sourced. Missing or
      invalid body evidence blocks only heading-dependent products; it does not
      invalidate detection-based distance or spatial products.
- [ ] Trial gaps remain retained evidence but are not active trial members.
      Event/trial attachment occurs only through exact persisted row identity.
- [ ] Epoch windows, settle policy, inclusive/exclusive endpoints, trial
      membership, provider roles, and object roles come from persisted records.
- [ ] Published speed products use an explicit verified speed level and valid
      sample/transition masks. Raw centroid speed is not silently accepted for
      publication.
- [ ] Every numerator is published beside its scientific denominator, support,
      dropout/censor counts, and validity/reason policy.
- [ ] Frame, bout, event, and visit rows are descriptive evidence. Cohort
      inference aggregates at fish/recording/session and records its clustering
      unit; no naive pooled-row p-values are allowed.
- [ ] Object role and color are separate layers. Role styling is derived from
      persisted role codes and never inferred from blue/red pixels or chaser
      array order.
- [ ] Static object-centered and moving-chaser-centered coordinate frames are
      separate products and cannot be mixed in one unlabeled panel or clip.
- [ ] Published immutable Zarr readers use consolidated metadata. Writers use
      unconsolidated access until payload, attrs, manifests, selectors, and
      receipts are final, then consolidate as the last visibility step.
- [ ] Selector activation, registry authority, deployment, and production
      publication remain separate operations after validation and required CI.

## Dependency order

    exact semantic epochs + exact provider positions + reviewed geometry
                                 |
              +------------------+-------------------+
              |                                      |
       exact body-frame                         exact motion/bouts
       sparse projection                       and controller trials
              |                                      |
       fish heading +                         epoch behavior
       alignment summary                      response regimes
              |                                      |
              +---------- exact relative frame ------+
                                 |
                    +------------+-------------+
                    |                          |
              escape-events v4          near-field visits v1
                                               |
                                  ring static/specs/media
                    |                          |
                    +------------+-------------+
                                 |
                  per-product receipts + profile envelope
                                 |
                    Marimo / static report / export
                                 |
                    frozen cohort canary and LSF fanout

The persisted-view reader package can proceed immediately and independently.
Epoch semantic-v2 integration, alignment-summary contract work, response-regime
contract work, and visit/escape schema design can proceed in parallel. Ring
media must wait for visit identity and exact frame mapping.

## Phase 0 — Freeze the compatibility and route inventory

- [ ] Add a machine-readable legacy-to-exact route matrix with one of:
      ported, conditional, renamed-not-equivalent, missing-reader,
      missing-successor, upstream-blocked, or retired.
- [ ] Record that body_bearing_polar is anatomical-body evidence and is not an
      alias for the old motion-heading egocentric view.
- [ ] Record that trajectory_overlays is not an alias for configured quadrant
      occupancy or a full detection-occupancy product.
- [ ] Record that escape_freeze is a diagnostic canary and is not an alias for
      chaser_escape_events.
- [ ] Record that scalar same-quadrant fraction is not the full 4 x 4 joint
      quadrant table.
- [ ] Keep exact routes free of scientific controls. Any controls must be
      display-only and their effect must be captured in figure provenance.
- [ ] Add a deferred-route inventory test so missing products cannot disappear
      silently during later notebook refactors.

Primary anchors:

- apps/marimo/components/analysis_catalog.py:186-273
- apps/marimo/components/chaser_exact/provider.py:88-171
- tests/unit/fisheye/test_marimo_chaser_exact_provider_adapter.py:324-354
- apps/marimo/components/goodcopbadcop_chaser.py:2248-2443
- docs/diagnostics/chaser_exact_successor_interactive_visualization_implementation_checklist_2026-08-27.md:329-446

Acceptance:

- [ ] Every historical scientific surface has an explicit disposition.
- [ ] No exact analysis ID obtains an implicit legacy loader.
- [ ] A test fails if a catalog route has no provider route, load contract,
      renderer, or explicit deferred/retired record.

## Phase 1 — Mount the already-sealed products first

This phase is the shortest path to additional useful plots and must not create a
new scientific successor.

### 1A. Distance distributions and geometric controls

Persisted evidence:

- src/fisheye/analysis_workflows/chaser_radial_near_field_successor.py:307-392
- src/fisheye/analysis/provider_chaser_position_suite.py:639-695
- docs/chaser_radial_occupancy_contract.md:42-60

- [x] Add DISTANCE_DISTRIBUTION_ARRAYS beside
      RADIAL_NEAR_FIELD_ARRAYS in
      apps/marimo/components/chaser_exact/array_requirements.py:5-21.
- [x] Require exact epoch role/window, behavior role, chaser identity,
      candidate frame count, valid distance count, and wall-excluded valid
      count.
- [x] Require the persisted CDF threshold and fraction arrays.
- [x] Require ordinary observed count/fraction, geometric expected available
      area/fraction, and selection index.
- [x] Require the corresponding configured wall-excluded observed, expected, and
      selection-index arrays.
- [x] Add distance_distributions to the exact route registry at
      apps/marimo/components/chaser_exact/provider.py:88-171.
- [x] Extend targeted receipt-bound loading at
      apps/marimo/components/chaser_exact/projection.py:675-723 for both the
      keypoint and detection radial children.
- [x] Add a dedicated renderer adjacent to radial_near_field.py.
- [x] Group only by exact epoch-window x behavior x chaser identity.
- [x] Plot persisted CDF thresholds and persisted radial bin edges; prohibit
      interpolation, reconstruction, and rebinning.
- [x] Show ordinary and wall-excluded denominators separately.
- [x] Label the geometric expectation and configured perimeter policy directly.
- [x] Include consumed arrays, thresholds, edges, providers, ordinary/wall
      denominators, and no-rebin policy in figure metadata.
- [ ] Share one pure validated projection/parameter builder with
      src/fisheye/utils/plot_chaser_detailed_successors.py:645-891 and
      :1634-1782 so static and interactive views cannot drift scientifically.

Acceptance:

- [ ] Mutating any consumed CDF/radial array fails receipt-bound loading.
- [x] Keypoint and detection providers render as independently labeled
      position providers.
- [ ] Static and interactive fixtures agree on strata, bins, denominators, and
      curve values.

### 1B. Candidate-epoch-normalized spatial occupancy

Persisted evidence:

- apps/marimo/components/chaser_exact/array_requirements.py:23-38
- src/fisheye/analysis_workflows/chaser_spatial_occupancy_successor.py:347-353
- src/fisheye/analysis_workflows/chaser_spatial_occupancy_successor.py:466-480
- src/fisheye/visualization/chaser_spatial_occupancy_display.py:32-262

- [x] Extend the existing spatial_occupancy route instead of adding a second
      scientific route.
- [x] Add an explicit display mode for density_valid_in_arena versus
      fraction_candidate_epoch.
- [x] Preserve separate units, color bars, denominator annotations, and scale
      defaults; never place both quantities on an unlabeled common scale.
- [x] Reuse SpatialOccupancyDisplaySurface for both native 2 mm and aligned
      4 mm displays.
- [x] Verify coarsening preserves mass for candidate fraction and density for
      the valid-in-arena surface.
- [x] Add the display-only normalization mode to the explorer spec and record
      it in figure metadata.
- [x] Either add a candidate-normalized static output or explicitly record
      static parity as incomplete.

Acceptance:

- [x] Low per-bin percentages are visibly interpretable as a probability mass
      spread over the arena rather than mistaken for missing occupancy.
- [x] Candidate-normalized panels show candidate count, in-arena count,
      missing/out-of-arena counts, and coverage.

### 1C. Same-quadrant summaries

Persisted scalar evidence:

- src/fisheye/analysis/provider_chaser_position_suite.py:449-472
- src/fisheye/analysis/provider_chaser_position_suite.py:583-605
- src/fisheye/analysis_workflows/chaser_radial_near_field_successor.py:317-337

- [x] Add valid-denominator and candidate-denominator same-quadrant fractions
      to radial_near_field or a dedicated same_quadrant_occupancy route.
- [x] Show valid distance frame count, same-quadrant valid frame count, and
      candidate frame count beside the fractions.
- [x] Treat the full 4 x 4 fish-quadrant x chaser-quadrant table as a separate
      schema task because it is computed at
      provider_chaser_position_suite.py:697-719 but not sealed.
- [ ] Create a versioned exact quadrant
      successor with quadrant identities, counts, fractions, candidate/valid
      denominators, geometry digest, and role/provider identity.
- [x] Do not revive an unclustered p-value title in the recording-local view.

### 1D. Receipt and spec impact

- [x] Keep exact_chaser_projection_receipt v2 unchanged for views that consume
      only its existing radial/spatial children
      (exact_chaser_projection_receipt.py:27-42).
- [x] Add closed route-specific consumed-array rosters and targeted rehash
      records; do not widen a generic permissive array list.
- [x] Make an explicit explorer-spec version decision when new display
      parameters are added; update fixtures and schema checks together.

## Phase 2 — Complete epoch behavior without duplicating epoch math

The semantic-v2 recording-local producer already owns exact epoch, motion,
bout, IBI, and denominator semantics:

- src/fisheye/analysis_workflows/materializers/provider_epoch_behavior_summary.py:98-121
- :354-449 for sealed semantic windows
- :533-789 for fish and bout metrics/denominators
- :909-1097 for exact source bindings
- :1173-1240 for persisted tables and the explicit spatial omission
- :1318-1595 for validation and atomic publication

### 2A. Integrate the existing semantic-v2 motion/bout product

- [x] Admit only semantic schema version 2 for the new exact route.
- [x] Reject or quarantine raw speed for published summaries even though
      SUPPORTED_SPEED_LEVELS currently includes raw at line 121.
- [x] Require exact provider-motion and same-track swim-bout handles.
- [x] Preserve sample_valid, transition_valid, valid tracked duration, bout
      denominator, semantic role/hash/step identity, and source interval
      digest in every row/view.
- [x] Add an exact loader and catalog route; do not use the legacy dashboard
      loader that expects per_epoch_chaser.
- [x] Remove viewer-side epoch recomputation from scientific admission. If
      retained for diagnostics, label it unsealed and keep it unavailable from
      exact routes.
- [x] Render speed/path summaries, valid duration/coverage, bout counts/rates,
      bout kinematics distributions, and IBI distributions from persisted
      tables.
- [x] Show source speed level and proxy-alignment limitations directly in the
      panel.
- [x] Add the exact semantic-v2 product to the composable cohort runner and
      its child-receipt composition.

### 2B. Add a separate epoch-spatial extension

Do not mutate provider motion into a position authority. Create a separately
versioned component, provisionally:

    palette.chaser.epoch_spatial_behavior.v1

Dependencies:

- exact semantic selection;
- one explicitly chosen fish-position provider;
- reviewed circular arena geometry and scale;
- exact acquisition-frame identity.

Persist per semantic epoch and provider:

- [ ] epoch role, window ID, semantic identity/hash, start/end convention;
- [ ] candidate, finite-valid, in-arena, out-of-arena, and invalid counts;
- [ ] valid tracked duration;
- [ ] center-distance count, mean, median, P05, P95, and maximum;
- [ ] persisted center-distance histogram edges, counts, and normalization;
- [ ] arena radius and configured wall-band width;
- [ ] wall-band count, fraction of valid/in-arena rows, and valid wall time;
- [ ] provider identity/digest, geometry digest, coordinate convention, and
      every denominator/validity reason.

- [ ] Reuse legacy chaser_epoch_behavior_summary.py:572-596 and :672-750 only
      as a field/parity reference, not as a source-tolerance or selector path.
- [ ] Compose semantic motion/bouts and epoch-spatial rows in the viewer by
      exact semantic epoch identity. Do not force them into one lineage.
- [ ] Permit keypoint and detection position providers as separate selectable
      products.

Acceptance:

- [ ] The semantic-v2 panel remains valid when no position provider exists and
      visibly states spatial_metrics_omitted.
- [ ] The combined full epoch panel appears only when exact semantic IDs and
      epoch bounds match across the two children.
- [ ] Wall/center metrics are never derived from motion speed arrays.
- [ ] No wall-clock denominator fallback exists.

### 2C. Epoch cohort/export guard

- [ ] Extend the provider cohort export with exact source/version/semantic
      lineage and optional epoch-spatial rows.
- [ ] Freeze recording/session/acquisition-batch columns before statistics.
- [ ] Either use the declared clustered statistics policy or require an
      EXPLORATORY ONLY receipt and watermark for Mean +/- SEM figures.
- [ ] Reject mixed schema versions, stale rows, raw published speed, and
      missing semantic digests.

## Phase 3 — Fish heading and alignment by distance

### 3A. Record the corrected current state

An older audit found body_frame=None, but PR 78 has already advanced beyond
that state. The current exact implementation:

- accepts one exact body_frame_run_name at
  chaser_proxy_relative_frame_adapter.py:736-746;
- projects sparse body rows onto exact acquisition frames without interpolation
  at :563-689;
- binds body run/manifest/recipe/projection evidence at :651-688 and
  :983-1040;
- persists body_heading_deg, body_heading_valid, body bearing, body-relative
  coordinates, reason codes, and source rows at
  chaser_relative_frame_storage.py:926-1013; and
- derives and passes the exact body run in the composable cohort runner at
  materialize_composable_chaser_successor_cohort.py:370-419 and :1314-1315.

The remaining gaps are consumer coverage, cohort backfill, and a persisted
alignment aggregation.

### 3B. Add the exact fish-heading route

- [x] Add a separate closed _BODY_HEADING_ARRAY_NAMES roster at
      apps/marimo/components/chaser_exact/projection.py so bearing-only
      capability remains backward-compatible. It includes body source row ID
      and validity plus heading value, validity, and reason code.
- [x] Add a conditional fish_heading route available only on the keypoint
      relative child with a valid body extension.
- [x] Validate that frame-level body heading repeats exactly across flattened
      chaser rows, collapse it once per acquisition frame, and never count the
      same fish-heading sample once per chaser.
- [x] Plot the persisted heading distribution by exact semantic epoch; do not
      source heading from velocity or track motion.
- [x] Show valid, missing-source, and present-invalid counts and the exact body
      run/recipe digest.
- [x] Keep detection provider distance/spatial routes available while heading
      remains explicitly unavailable for detection.
- [x] Verify a real canary exact keypoint relative-frame child with body
      extension through its reusable validation receipt before cohort fanout.

### 3C. Add a body-alignment summary successor

Create a separately versioned immutable summary, provisionally:

    palette.chaser.body_alignment_by_distance.v1

Source:

- exact keypoint relative-frame child with body extension;
- exact semantic selection;
- the base physical distance arrays from that same relative child.

Persist frame evidence sufficient for audit:

- [x] acquisition frame, epoch/window identity, chaser identity and role;
- [x] body source row ID, heading/bearing validity and reason;
- [x] body bearing, alignment_cos = cos(bearing), and
      lateral_sin = sin(bearing);
- [x] base physical distance, distance validity, and reason.

Persist scientific summary rows keyed by epoch x chaser x distance bin:

- [x] exact bin start/end/center in mm and bin recipe digest;
- [x] candidate row count, jointly valid row count, and invalid-reason counts;
- [x] mean alignment, mean absolute bearing, circular mean/resultant length,
      and declared descriptive spread/quantiles;
- [x] provider/body/relative/semantic source handles and digests.

- [x] Define distance as the base relative_distance_physical requiring the
      selected position provider; do not silently substitute body-origin
      distance.
- [x] Reject source-axis length mismatch instead of reproducing the legacy
      shortest-axis truncation in chaser_egocentric_bearing.py:296-300.
- [x] Preserve the canonical camera +Y-down to world +Y-up heading convention
      and atan2(left, forward) bearing convention.
- [x] Load persisted summary bins in Marimo. Remove the scientific group-by
      and viewer-selected rebinning at
      apps/marimo/components/goodcopbadcop_chaser.py:2391-2403.
- [x] Permit only plot filtering/sampling in the interactive layer.

Acceptance:

- [x] Closed-form synthetic fixtures cover 0, +/-90, and +/-180 degree
      heading/bearing/alignment values.
- [x] Missing body rows never acquire a motion-heading fallback.
- [x] Fish-position-invalid/body-valid policy is explicit and tested.
- [x] Static and interactive views use identical persisted bins and support.
- [ ] Cohort exports aggregate by recording/session rather than pooling
      framewise alignment rows.

## Phase 4 — Persist event-aligned escape-distance trajectories

Current v3 evidence:

- src/fisheye/analysis/chaser_escape_events.py:115-116 declares v3;
- :493-545 computes event x reference x time samples in memory;
- :549-581 reduces those samples;
- :1057-1071 persists event scalars; and
- :1108-1117 persists only epoch x reference x time median traces.

Add immutable schema v4 while preserving v3:

    palette.chaser_escape_events.v4

### 4A. Aligned sample contract

Retain every v3 group and add an aligned_traces group with:

- [ ] time_frame_offset and time_s;
- [ ] sample_acquisition_frame_id or camera_frame_id;
- [ ] sample timestamp_ns and timestamp validity;
- [ ] distance_mm and delta_distance_mm on event x reference x time;
- [ ] sample validity mask and sample invalid-reason code;
- [ ] event bout ID, event row ID, epoch identity, start frame, provider/chaser
      identity, and reference identity;
- [ ] exact trial row ID/ordinal or explicit no-membership sentinel plus
      membership source;
- [ ] trace usability, event-level censor code, valid fraction, dropout
      fraction, and baseline validity.

- [ ] Persist both absolute and onset-baseline-subtracted distance so reduced
      medians, gain, and recapture can be independently reproduced.
- [ ] Preserve unusable event rows. Trace usability may exclude a row from a
      trajectory summary but must not remove it from event counts/rates.
- [ ] Use explicit censor codes including ok, out_of_recording,
      crosses_epoch_boundary, dropout_exceeds_threshold, no_finite_baseline,
      invalid_reference, missing_frame_identity, and ambiguous_provider.
- [ ] Persist one explicit provider/chaser assignment; fail closed on ambiguous
      multi-chaser assignment.
- [ ] Annotate trial membership without using trial gaps as members.

### 4B. Window convention

- [ ] Freeze sample frames as
      [event_start - pre_frames, event_start + post_frames).
- [ ] Require the first sample to be at or after the effective inclusive epoch
      start and the last sample to be at or before the effective inclusive
      epoch end.
- [ ] Document and test the one-frame compatibility difference if v3's
      stricter boundary behavior is corrected.
- [ ] Freeze onset, baseline, settle trim, dropout threshold, reference order,
      and time conversion in the manifest.

### 4C. Publication and consumer

- [ ] Extend component validation with required v4 shapes, dtypes, axis order,
      censor registry, payload inventory, and digests.
- [ ] Publish through private staging, reopen/deep validation, atomic rename,
      and an independent exact-child receipt.
- [ ] Add explicit v4 support to the profile, surface catalog, runner, exact
      projection composition, and cohort runner.
- [ ] Add an escape_trajectories exact route with bounded event/reference/time
      loading.
- [ ] Render selectable individual event traces plus persisted
      median/quantile summaries, event counts, usable counts, and censor counts.
- [ ] Do not reconstruct aligned samples from source distance data in Marimo.
- [ ] Recompute/backfill v4 from exact v3 dependencies and canonical distance;
      never attempt to recover event samples from v3 medians.

Acceptance:

- [ ] v4 scalar/event counts match v3 under equivalent configuration.
- [ ] v4 reduction reproduces v3 median/gain/recapture or reports every
      intentional boundary difference.
- [ ] Empty-event and all-censored recordings publish valid explicit artifacts.
- [ ] Source frame/timestamp identity survives round-trip and tampering fails.

## Phase 5 — Persist individual near-field visits

Current aggregate products do not retain visit identity:

- chaser_radial_occupancy.py:99-179 and :1334-1401 persist epoch/radial
  aggregates;
- chaser_near_field_occupancy.py:114-220 and :1004-1426 persist aggregate
  entries, dwell, censor counts, density, and CDFs;
- chaser_visit_trajectories.py:98-117 intentionally fails closed, while
  :119-269 privately reconstructs visits for inspection; and
- the private legacy path silently drops visits with fewer than five valid
  samples at :227-235.

Create one scientific successor:

    palette.chaser.near_field_visits.v1

Suggested component family:

    analysis/chaser_near_field_visits_runs/<exact-run>

### 5A. Exact dependencies

- [ ] Exact keypoint or detection relative-frame child with provider identity.
- [ ] Exact semantic selection and epoch records.
- [ ] Exact reviewed arena geometry/scale.
- [ ] Exact radial/near-field policy containing entry and exit radii.
- [ ] Exact body-alignment child only for heading-dependent visit fields.
- [ ] Exact bout-response, controller-trial, and escape-event handles only when
      those attachments are requested.
- [ ] No latest discovery during computation, publication, or viewing.

### 5B. Visit identity and state machine

Persist one row per exact:

    (epoch_window_id, reference_kind, reference_id, visit_ordinal)

- [ ] Define visit_key as a deterministic digest of base run identity, epoch,
      reference kind/identity, parent chaser identity, virtual rotation, and
      ordinal.
- [ ] Persist reference_kind as object, virtual, or dish_center.
- [ ] Persist reference ID/label, chaser and parent-chaser identity, behavior
      role, and virtual rotation.
- [ ] Persist epoch/window identity and effective start/end boundaries.
- [ ] Reject duplicate keys, overlapping visit intervals, unstable reference
      order, and chaser-column-position substitution.

Use the maintained hysteresis semantics in
chaser_near_field_occupancy.py:525-656:

- [ ] Require a known valid outside sample before an observed entry.
- [ ] Entry is a valid crossing strictly below r_in.
- [ ] Exit is a valid crossing strictly above r_out; equality is not a
      crossing.
- [ ] Phase starts inside: retain a censored row with
      phase_start_inside and no observed entry.
- [ ] Invalid gap while active: close/censor with invalid_gap, reset to unknown,
      and never synthesize a resumed-inside entry.
- [ ] Phase ends inside after an observed entry: phase_end_inside.
- [ ] complete is true only for observed entry plus observed valid exit.
- [ ] Persist both first-outside exit frame and last-inside frame.
- [ ] Retain short visits with a quality flag; never reproduce the silent
      fewer-than-five-sample removal.
- [ ] Apply the same versioned visit state machine to every exact selected
      semantic epoch, including pre/post static and random-wander/training
      periods when present; retain epoch role so those contexts never pool
      silently.

### 5C. Visit and ragged-sample payload

Suggested groups:

    epochs/
    references/
    visits/
    samples/
    summary/
    visualizations/

Visit rows:

- [ ] visit key/ordinal, epoch, reference, provider, role, and source handles;
- [ ] entry frame/timestamp/observed flag;
- [ ] last-inside and exit frame/timestamp/observed flag;
- [ ] complete, censor reason, invalid-gap count, quality flag;
- [ ] valid sample count, valid duration, dwell, CPA/min/max distance;
- [ ] path length, displacement, heading/alignment summaries;
- [ ] exact trial, bout, and escape attachments with attachment reason.

Flattened ragged samples using sample_offset and sample_count:

- [ ] base relative row ID;
- [ ] acquisition/camera frame ID and stimulus frame number when available;
- [ ] timestamp_ns and epoch/window identity;
- [ ] source position/detection row ID;
- [ ] fish and reference arena coordinates;
- [ ] canonical relative coordinates and frame_kind;
- [ ] distance, heading, bearing, alignment, and validity/reason arrays;
- [ ] time_from_entry_s computed from timestamp_ns;
- [ ] explicit FPS fallback flag only if timestamps are unavailable;
- [ ] optional speed, bout ID/onset, escape ID, and response-state code.

Do not pad or silently truncate visits. Verify monotonic timestamps, source
array lengths, offsets, and exact row bounds.

### 5D. Summary outputs

Persist summaries by epoch x reference and epoch x real chaser:

- [ ] total, complete, and censored visits;
- [ ] left-boundary, right-boundary, invalid-gap, and short-quality counts;
- [ ] valid samples, valid duration, and entry denominator/rate;
- [ ] complete-dwell mean/median/quantiles;
- [ ] CPA/distance/path/displacement summaries;
- [ ] heading/alignment summaries with body-valid support;
- [ ] bout/escape/turn-escape attachment counts;
- [ ] real-versus-virtual descriptive differences;
- [ ] timestamp coverage, invalid fraction, and geometry QC.

Keep visit as the descriptive row unit. Do not treat multiple visits from one
fish as independent cohort replicates.

### 5E. Marimo and tests

- [ ] Add near_field_visits to the exact catalog and provider.
- [ ] Add epoch, provider/reference, visit, completion/censor, and quality
      selectors.
- [ ] Display the visit table, canonical trajectory, aligned
      distance/heading/alignment traces, and persisted artifacts.
- [ ] If the component is absent, show a typed unavailable reason and never
      call the private legacy collector.
- [ ] Test phase-start, complete, invalid-gap, phase-end, threshold equality,
      no-valid-sample, and short-visit cases.
- [ ] Test exact row/timestamp preservation, duplicate/overlap rejection,
      stable reference ordering, bout/trial/escape joins, and provider roles.
- [ ] Test sealed round-trip, dependency digest mismatch, tampering, collision
      behavior, selector unchanged, and direct/consolidated equivalence.

## Phase 6 — Ring-entry views and media

Ring output is derived visualization evidence, not a replacement for the visit
successor. Keep three products visibly distinct:

1. receipt-bound static PNG and interactive spec;
2. synthetic trajectory animation from persisted visit samples;
3. source-frame review video with an exact video/frame map.

### 6A. Static ring-entry contract

- [ ] Define a versioned visualization contract containing:
  - static pre/post object-centered frame semantics;
  - moving-training/chase chaser-centered frame semantics;
  - ring edges, responsive shell, escape trigger, and hysteresis thresholds;
  - settle trimming, padding, panel/entry caps, and invalid-gap policy;
  - role resolution, layer order, units, and coordinate orientation.
- [ ] Keep the 5/6 mm near-field visit hysteresis distinct from the wider
      15/20 mm response shell. Persist and label both recipes when a panel
      displays both; never call the response shell a near-field visit.
- [ ] Resolve object role from persisted role codes. Missing role evidence
      becomes unknown/non-comparable; never fall back to chaser index zero.
- [ ] Render from persisted visit samples, not collect_ring_entries or
      _collect_visits_unsealed_inspection.
- [ ] Reuse display logic from chaser_ring_traversal.py:598-767 only after its
      inputs are replaced by the sealed visit source.
- [ ] Persist PNGs through write_png_visualization_artifact with contract,
      renderer/version, source paths/runs, signature, parameters, SHA-256, and
      byte length.
- [ ] Persist an interactive spec that points to canonical visit arrays rather
      than duplicating scientific data.
- [ ] Register ring_entry_static in the exact catalog, renderer registry,
      reporting catalog/executor, and Marimo dispatch.

### 6B. Synthetic trajectory animation

- [ ] Label this explicitly as synthetic trajectory media, not source video.
- [ ] Persist visit keys, displayed sample rows, frame convention, stride,
      frames per entry, entry cap, renderer, codec/container, and output hash.
- [ ] Use only persisted visit samples and sealed overlay parameters.
- [ ] Keep scientific visit success independent from optional animation
      encoding success; record media state and reason separately.

### 6C. Source-frame media successor

Create a separate media publication, provisionally:

    palette.chaser.ring_entry_media.v1

- [ ] Resolve a single-video recording through
      palette.source_video_metadata.v2 with canonical locator, dimensions,
      FPS, frame count, codec, pixel format, and conflict checks.
- [ ] Resolve clipped recordings through the finalized collection and
      recording_frame_index.parquet; never sort clip directories.
- [ ] Bind camera serial, clip ID, clip-local frame, recording-frame ID,
      source path, and source-frame index for every output frame.
- [ ] Record both clocks: source frame identity and time relative to ring
      entry.
- [ ] Define inclusive/exclusive clip boundaries, entry padding, no crossing
      epoch/static/chase boundaries, invalid/dropout behavior, entry cap, and
      output frame cap.
- [ ] Declare whether pixels come from the ingest-authoritative full-frame
      stream or a display proxy. Runtime crop availability must not imply crop
      consumption.
- [ ] Record coordinate-space scaling for every overlay.
- [ ] Define output container/codec. A suitable display baseline is H.264 MP4
      with faststart, exact FPS/timeline, declared pixel format, and
      display-only status.
- [ ] Record encoder, preset/CRF, hardware acceleration, scale flags, command,
      ffmpeg/ffprobe versions, dimensions, FPS, frame count, decode status,
      output SHA-256, and byte length.
- [ ] Encode to a temporary path, validate, atomically rename, then publish the
      manifest pointer last.
- [ ] Keep large MP4s outside the Zarr payload while storing their normalized
      discoverable path, digest, byte count, frame map, and source lineage in
      the component visualizations manifest.
- [ ] Add a bounded Marimo stream/download path. The current generic viewer at
      src/fisheye/utils/view_zarr_visualization.py:145-174 accepts only PNG and
      must not be loosened generically.

Acceptance:

- [ ] Static and moving-chaser frame labels cannot be swapped.
- [ ] Synthetic media cannot be presented as source-frame review.
- [ ] Single-video and clipped mappings, nonzero frame bases, epoch boundaries,
      missing frames, wrong clips, tampering, codec/FPS/count, and partial
      publication failure all have adversarial tests.
- [ ] No publication manifest exists when a required selected clip fails.

## Phase 7 — Response regimes

Current v1 is a real producer, not an absent algorithm:

- chaser_response_regimes.py:74-75 declares v1;
- line 92 defines the 2.0 mm/s moving threshold;
- its current immobile/moving thresholds leave a 1-2 mm/s dead band;
- lines 274-397 compute distance-binned profiles and adjacent transitions;
- lines 938-1141 persist a sealed component with
  epoch x chaser x distance-bin summaries.

The gap has two stages.

### 7A. Safe compatibility visibility

- [ ] Add an exact v1 loader only when an immutable exact handle, manifest,
      provider-motion source, speed level, and support arrays validate.
- [ ] Add response_regimes to catalog, registry, provider, projection,
      reporting catalog/executor, and static artifacts.
- [ ] Show immobile fraction, moving fraction, moving-away fraction, fish and
      chaser separation rates, event gain where present, valid support,
      dropout, and source speed level.
- [ ] Label v1 as descriptive compatibility and display its 1-2 mm/s dead-band
      policy.
- [ ] Do not attach confirmatory p-values or imply causal preference.

### 7B. Final v2 state/transition successor

Create:

    palette.chaser_response_regimes.v2

Exact dependencies:

- provider-aware relative frame;
- verified motion arrays and explicit speed level;
- semantic selection;
- exact trial/bout/escape handles where attachment is requested;
- body-alignment and visit handles as optional governed extensions.

- [ ] Freeze an explicit state registry:
      immobile, swim, escape, unknown/censored, plus any deliberate dead-band
      state.
- [ ] Decide and version the threshold/hysteresis policy rather than inheriting
      an accidental dead band.
- [ ] Persist frame x chaser state evidence with distance, fish/chaser radial
      velocity or separation contribution, validity, and censor reason.
- [ ] Persist selection x chaser x distance-bin profiles with numerator,
      valid sample/transition counts, valid seconds, risk-set support, dropout,
      floor-clamp status, and threshold parameters.
- [ ] Persist a state-transition table with from/to state, exact boundary row,
      support, and transition validity.
- [ ] Persist a dwell table with state, start/end, duration, censoring, and
      epoch/trial/visit attachments.
- [ ] Persist a recording summary and optional bout-repertoire table.
- [ ] Distinguish fish motion from chaser/reference motion.
- [ ] Preserve the physical clamp geometry and record when it is active.
- [ ] Do not use the pursuit annulus as a valid null. Mark its known
      closed-loop limitation explicitly.
- [ ] Treat wall distance as a measured mediator/context, not an automatically
      removable nuisance.

Marimo:

- [ ] Render persisted state fractions/curves, separation components,
      transitions, dwell, dropout, and support.
- [ ] Allow epoch/provider/state display filtering only.
- [ ] Show state method/version, thresholds, speed source, denominator, and
      applicability/QC watermark.

Cohort:

- [ ] Export descriptive recording summaries and exact support columns.
- [ ] Aggregate frame/transitions/dwells to fish/recording before inference.
- [ ] Add MetricSpecs only after v2 semantics are frozen, with explicit
      session/acquisition-batch clustering and correction family.

## Phase 8 — Full-profile composition, receipts, and readiness

Existing foundations:

- full_chaser_profile_successor.py:1-35 defines a digest-bound applicability
  and immutable-module envelope;
- :91-150 defines exact module-product bindings and selector-ineligible
  constraints;
- :170 onward prepares the composed profile;
- chaser_profile_applicability.py:464-576 and :788-1014 defines and validates
  plan/readiness evidence;
- exact_chaser_projection_receipt.py:27-42 currently composes seven core,
  optional gaze, and two relative-frame receipts.

### 8A. Keep lineage modular

- [ ] Continue producing one independent receipt per scientific child.
- [ ] Add new child families through a versioned receipt/profile extension,
      not by placing every lineage into one unstructured bundle.
- [ ] Prefer a versioned profile receipt whose child map is validated against a
      closed profile/module schema.
- [ ] Bind exact schema/version, run path, manifest, payload/receipt digest,
      applicability state, dependency order, and selector status per module.
- [ ] Preserve the current projection receipt for the existing interactive
      bundle; introduce v3 only when its closed child grammar changes.
- [ ] Do not make the composition receipt a selector, cache authority, or
      substitute for current child metadata validation.

### 8B. Execute and publish the applicability plan

- [ ] Invoke the full-profile applicability planner in the maintained
      composable runner, not only in proxy-candidate receipts.
- [ ] Persist one recording-local applicability envelope with recording ID,
      profile/version/hash, capability evidence, module states/reasons,
      dependency order/waves, overrides, readiness, and plan digest.
- [ ] Distinguish applicable, complete, inapplicable, missing, invalid,
      review-required, stale, and unrun.
- [ ] Never claim full completion for planned/blocked/stale/unrun modules.
- [ ] Keep reduced profiles explicitly not_claimed_reduced_profile.
- [ ] Resolve or explicitly mark the current virtual_reference_controls
      capability gate in chaser_behavior_full_v3.yaml:106-120.
- [ ] Treat missing body, eye, video, and visit capability as module-specific
      applicability evidence instead of invalidating unrelated position views.

### 8C. Full-profile Marimo route

- [ ] Add full_profile_readiness to the exact catalog/provider.
- [ ] Display module ID, source handle, schema/version, capability state,
      module state/reason, dependency order/wave, profile/plan digest,
      selector eligibility, and temporal alignment class.
- [ ] Make this a composed readiness/dashboard surface, not a numerical
      recomputation layer.
- [ ] Permit navigation from each completed module row to its modular view.

## Phase 9 — Static artifacts, provenance, and diagnostic evidence

### 9A. Receipt-bound static artifact gallery

The current static panel resolves paths and reads bytes at
apps/marimo/components/static_artifacts.py:14-54. The writer at
src/fisheye/shared/plot_artifacts.py:60-181 and :244-289 records substantially
more evidence than the generic reader validates.

- [ ] Resolve previews only through the selected exact run/component
      visualization manifest and publication receipt.
- [ ] Verify normalized path ownership, component ownership, artifact schema,
      MIME, PNG signature, byte length, content SHA-256, manifest entry,
      visualization contract, renderer/version, artifact signature,
      source_paths, and source_runs.
- [ ] Reject path traversal, cross-component artifacts, source-run swaps,
      stale consolidated metadata, oversized payloads, and content tampering.
- [ ] Show verification state, receipt digest, content hash, contract,
      renderer, and source bindings in the gallery.
- [ ] Keep the generic archive gallery opt-in and diagnostic; it is not
      scientific authority.

### 9B. Pre/post polar reference snapshots

- [ ] Keep old chaser_egocentric_bearing pre/post PNGs as historical/reference
      artifacts only.
- [ ] If exposed, attach them to the verified static-artifact/body-bearing
      route and exact component binding.
- [ ] Prefer interactive plots from persisted exact body arrays.
- [ ] Do not revive direct raw GoodCopBadCop dashboard discovery or unsealed
      motion-heading source tolerance.
- [ ] Explicitly label the current two-chaser/pre-post compatibility
      assumptions or generalize the renderer before applying it elsewhere.

### 9C. Provenance and debug tables

- [ ] Extend the exact provenance panel with profile/applicability envelope,
      exact provider/body/motion/trial/event/visit/media bindings, consumed
      arrays, receipt digests, artifact hashes, parameters, and verification
      mode.
- [ ] Add bounded diagnostic tables for semantic windows, visible frame rows,
      provider coverage, body validity/reasons, spatial denominators, epoch
      summaries, event censors, visit censors, and module decisions.
- [ ] Load diagnostics from the same receipt-bound projection; do not open a
      second legacy dashboard data path.
- [ ] Make candidate/unpromoted status visible in every diagnostic export.

## Phase 10 — Modular Marimo integration

Keep the explorer modular. Each scientific product owns:

- a closed required-array roster;
- a typed projection/data object;
- one loader that verifies exact handles and receipts;
- one renderer module;
- route availability/applicability logic;
- figure/display provenance; and
- focused tests.

The notebook remains an orchestrator.

### 10A. Primary edit map

| Concern | Primary location |
|---|---|
| Exact route registry and load flags | apps/marimo/components/chaser_exact/provider.py:71-171 |
| Closed consumed-array rosters | apps/marimo/components/chaser_exact/array_requirements.py:5-120 |
| Projection data object | apps/marimo/components/chaser_exact/projection.py:266-281 |
| Receipt-bound targeted loading | apps/marimo/components/chaser_exact/projection.py:284-330 and :650-735 |
| Catalog labels/descriptions | apps/marimo/components/analysis_catalog.py:186-273 |
| Recording/spec discovery and display parameters | apps/marimo/components/registry.py:841-1055 |
| Top-level dispatch | apps/marimo/palette_explorer.py:1027-1141 |
| Exact provenance | apps/marimo/components/chaser_exact/provenance.py:51-191 |
| Provider adapter tests | tests/unit/fisheye/test_marimo_chaser_exact_provider_adapter.py |
| Discovery/spec tests | tests/unit/fisheye/test_marimo_chaser_exact_successor_component.py |

### 10B. Route checklist

- [x] distance_distributions
- [x] spatial_occupancy candidate-normalized mode
- [x] same_quadrant_occupancy
- [ ] quadrant_joint_occupancy when its successor exists
- [x] epoch_behavior
- [x] fish_heading
- [x] alignment_by_distance
- [ ] escape_trajectories
- [ ] near_field_visits
- [ ] ring_entry_static
- [ ] ring_entry_media
- [ ] response_regimes
- [ ] full_profile_readiness
- [ ] verified_static_artifacts
- [ ] expanded provenance/debug evidence

For every route:

- [ ] Availability derives from exact capability and child validation.
- [ ] Missing capability returns a typed reason, not an empty unexplained
      plot and not a fallback.
- [ ] Load only the arrays needed by the selected renderer.
- [ ] Display parameters are separate from scientific parameters.
- [ ] Exported figures embed source paths/digests, consumed arrays, exact
      persisted bins/thresholds, denominators, and display projection.
- [ ] Route/registry/load/dispatch parity tests enumerate the complete set.

## Phase 11 — Static publication, exports, reports, and statistics

### 11A. Recording-local static products

- [ ] Register each new static visualization ID in
      src/fisheye/reporting/catalog.py and its executor in
      src/fisheye/reporting/execution.py.
- [ ] Render from the same pure validated projection used by Marimo.
- [ ] Persist every bin size, threshold, provider choice, normalization,
      role/style map, display cap, and renderer version in the recipe/receipt.
- [ ] Validate PNG/PDF/media hashes and exact source run digests.
- [ ] Keep diagnostic canaries visibly diagnostic.

### 11B. Cross-recording export contracts

- [ ] Define one exact row unit for every table:
  - recording x epoch for epoch behavior;
  - recording x epoch x chaser x distance bin for alignment/regimes;
  - recording x event for escape evidence;
  - recording x visit for near-field evidence;
  - recording x module for profile readiness.
- [ ] Include schema/method/provider/profile versions, exact source handles,
      denominator, support, validity/censor fields, semantic identity, and
      frozen cohort identity.
- [ ] Require an exact authority-set manifest and exact component handles;
      never rediscover latest children during export.
- [ ] Keep frame/event/visit tables descriptive and separate from the
      recording-level inference table.

### 11C. Statistical admission

- [ ] Add MetricSpecs only after source-table semantics are stable.
- [ ] Declare fish/recording/session/acquisition-batch clustering and repeated
      subject policy explicitly.
- [ ] Reject singleton primary families and naive pooled frame/bout/event/visit
      inference.
- [ ] Record multiple-comparison family and exploratory/confirmatory tier.
- [ ] Require an EXPLORATORY ONLY watermark and receipt whenever clustered
      inference is unavailable.
- [ ] Do not place p-values in ordinary recording-local chart titles.

## Phase 12 — Cohort execution and backfill

### 12A. Freeze membership

- [ ] Use the cohort plan, coverage, and freeze workflow before submission.
- [ ] Record normalized query/hash, registry UUID, consulted-row hashes, exact
      Zarr membership, prerequisite states, exclusions, and manifest hash.
- [ ] Fail on missing/ambiguous metadata, duplicate active Zarrs, multiple
      latest stimulus runs, and non-ok prerequisites unless an explicit
      written exclusion policy permits them.
- [ ] Never silently drop recordings with missing selected metadata.

Prerequisites by product:

- [ ] exact stimulus semantics for every product;
- [ ] exact provider position and geometry for distance/spatial/visits;
- [ ] exact accepted body-frame supplier binding for heading/alignment;
- [ ] verified motion and same-track bouts for epoch/regimes/events;
- [ ] exact controller trials for trial/event attachment;
- [ ] exact bout response for escape v4;
- [ ] exact visits before ring artifacts/media;
- [ ] exact video/frame mapping before source-frame media.

### 12B. Extend the maintained composable runner

The maintained exact cohort runner currently materializes relative, controller,
bout, escape/freeze, radial, spatial, projection receipts, and plot bundles in
materialize_composable_chaser_successor_cohort.py:1306-1826.

- [x] Add semantic-v2 epoch behavior.
- [ ] Add the separately governed epoch-spatial child.
- [x] Add body-alignment summary.
- [ ] Add escape-events v4.
- [ ] Add near-field visits.
- [ ] Add response-regimes v2.
- [ ] Add profile applicability/full-profile envelope.
- [ ] Add independent exact-child receipts for each.
- [ ] Add a versioned composition/profile receipt containing their exact
      bindings.
- [ ] Add static ring artifacts after visits.
- [ ] Keep source-frame media as an optional dependent stage so encoder failure
      cannot corrupt scientific-success state.
- [ ] Add receipt reuse only when exact task, code, parameters, source handles,
      payloads, and expected outputs match.

### 12C. Deployment and LSF

- [ ] Commit the implementation and obtain required green CI first.
- [ ] Deploy a clean commit-pinned worktree with
      scripts/deploy_palette_cluster_worktree.sh; do not move the shared
      /groups checkout for concurrent experiments.
- [ ] Record the absolute PALETTE_GROUPS_REPO and full commit in every job.
- [ ] Run a one-recording canary through every applicable stage.
- [ ] Deep-validate the canary, compare direct/consolidated metadata, verify
      child/composition receipts, and inspect all new figures.
- [ ] Submit the frozen zarr_paths.txt with bounded max-active fanout.
- [ ] Use failure-closed done(job_id) dependencies.
- [ ] Keep array tasks free of registry mutation.
- [ ] Run one serial finalizer after all required tasks succeed; it may record
      derived stage status but must not activate production selectors.
- [ ] Record per-recording complete, blocked, inapplicable, censored, and
      failed reasons rather than presenting one undifferentiated success count.

## Phase 13 — Required validation and CI

### 13A. Cross-cutting contract tests

- [ ] Exact row/order/source/provider mismatch.
- [ ] Cross-recording and cross-camera mismatch.
- [ ] Manifest/payload/receipt tampering.
- [ ] Exact retry equivalence and immutable same-name collision.
- [ ] Direct versus consolidated metadata agreement.
- [ ] Missing body: position views complete, heading views unavailable.
- [ ] Canonical coordinate orientation and angle wrapping.
- [ ] No interpolation, no nearest row, no motion-heading fallback.
- [ ] Exact epoch role/hash/step identity and boundary behavior.
- [ ] Exact trial gaps retained but non-member.
- [ ] Denominator and no-wall-clock fallback.
- [ ] Persisted bins/thresholds and no viewer rebinning.
- [ ] Event boundary/dropout: trace exclusion versus rate inclusion.
- [ ] Visit hysteresis, invalid gaps, censoring, and exact attachments.
- [ ] Static versus moving ring coordinate frames.
- [ ] Video/clip/frame-map identity and media tampering.
- [ ] Profile applicability reason-state distinctions.
- [ ] No-fallback Marimo discovery and complete route parity.

### 13B. Validation commands

Use scripts/py for every Python command.

- [ ] Run py_compile and git diff --check for fast static validation.
- [ ] Run focused in-memory unit suites first.
- [ ] Run all real-Zarr pytest suites outside the sandbox.
- [ ] Run scripts/py -m marimo check outside the sandbox.
- [ ] Run static artifact and media decoder/hash smokes.
- [ ] Run the relevant full unit/integration suite.
- [ ] Push the branch and require every repository-required CI check to finish
      successfully.

Failed, cancelled, timed-out, or skipped required checks remain blocking. A
commit may be handed off with those states only as explicitly incomplete and
must not be described as merge-ready, deployed into the shared checkout, or
used to activate selectors.

## Parallel implementation packages

After the reader-only Phase 1 starts, use disjoint work packages to reduce
merge contention:

| Package | Scope | Can begin | Blocks |
|---|---|---|---|
| A | Persisted distance/CDF/spatial/same-quadrant views | immediately | none |
| B | Semantic-v2 epoch integration | immediately | epoch route/export |
| C | Epoch-spatial successor | after exact position/geometry choice | full epoch view |
| D | Fish-heading route | immediately on PR-78 body branch | heading view |
| E | Alignment-summary successor | after exact body-relative contract | alignment view/export |
| F | Escape-events v4 | after exact trial/bout-response binding | event trajectory view |
| G | Near-field visits v1 | after exact relative/geometry policy | ring products |
| H | Response-regimes v2 | after verified motion/state decision | regime view/export |
| I | Ring static/spec | after G | ring static route |
| J | Ring media | after G and video/frame map | media route |
| K | Full-profile envelope/receipts | contracts can start immediately; binds after B-J | readiness/dashboard |
| L | Cohort exports/statistics | after stable source schemas | cohort publication |

Likely merge hotspots that should be owned by one integration package:

- apps/marimo/components/chaser_exact/provider.py
- apps/marimo/components/chaser_exact/projection.py
- apps/marimo/components/analysis_catalog.py
- apps/marimo/components/registry.py
- apps/marimo/palette_explorer.py
- src/fisheye/utils/materialize_composable_chaser_successor_cohort.py
- src/fisheye/analysis/profiles/chaser_behavior_full_v3.yaml

Scientific packages should add their own renderer/loader/test modules and let
the integration package make the small central registry edits.

## Recommended first execution

To obtain useful plots quickly while preserving the final architecture:

1. land the reader-only distance distributions, candidate-normalized
   occupancy, and same-quadrant scalar views;
2. land the exact fish-heading route using the already implemented body
   extension;
3. integrate semantic-v2 epoch motion/bouts;
4. implement and canary the alignment summary;
5. develop escape v4, visits, and response-regimes v2 in parallel;
6. render static ring artifacts from visits, then add optional media;
7. compose the full-profile/readiness receipt and routes;
8. run one complete canary; and
9. freeze and fan out the cohort only after required CI is green.

This ordering makes currently sealed evidence visible first, avoids another
legacy accommodation layer, and reserves new schemas for the products that
actually lost event or visit identity.

## Final closure ledger

Do not mark this document implementation-complete until every applicable box
below is supported by a passing receipt-bound canary:

- [ ] Distance CDF/distributions and geometric/wall-excluded controls
- [ ] Both spatial occupancy normalization modes
- [ ] Same-quadrant scalar and full joint quadrant product
- [ ] Semantic-v2 epoch speed/bout/IBI behavior
- [ ] Epoch wall/center spatial behavior
- [ ] Exact anatomical fish heading
- [ ] Persisted alignment by distance
- [ ] Event-aligned escape-distance trajectories
- [ ] Individual near-field visits
- [ ] Ring-entry static views/specs
- [ ] Synthetic ring media, if retained
- [ ] Source-frame ring media, when exact video/frame identity is available
- [ ] Response-regime states/transitions/dwells
- [ ] Full-profile readiness/module bindings
- [ ] Receipt-bound static artifact gallery
- [ ] Expanded provenance/debug evidence
- [ ] Recording-local static publication
- [ ] Frozen cohort exports and governed statistics
- [ ] One-recording canary
- [ ] Full eligible-cohort LSF fanout
- [ ] Required CI green
- [ ] Selector/promotion decision made separately
