# Exact Chaser Successor Interactive Visualization Implementation Checklist

<!-- contract-meta
status: implementation-in-progress
implementation: receipt-bound-reader-merged-first-class-gaze-and-distribution-work-in-progress
last_verified: 2026-08-30
-->

## Purpose

Make the completed receipt-bound chaser successors safely discoverable and
interactive in Palette's recording Marimo explorer without rewriting the
scientific Zarr products, resolving a selector, recomputing scientific
membership or metrics, or falling back to legacy/candidate analyses.

This checklist implements the findings in
[`chaser_exact_successor_marimo_status_2026-08-26.md`](chaser_exact_successor_marimo_status_2026-08-26.md).
The first package is a reader compatibility correction. Later packages add
interactive coverage and, if accepted, a separately versioned visualization
descriptor. Do not make those later packages prerequisites for correcting the
current zero-option defect.

## Current evidence and status

- [x] The selector-ineligible scientific cohort completed for 80 eligible
      recordings: one canary plus task indices `2-76,81-84` in LSF job
      `153756073`.
- [x] Indices `77-80` remain explicit protocol-semantic exclusions and were not
      treated as successful cohort members.
- [x] All 80 cohort receipts bind Palette commit
      `65b06a2f6ab4c4c30a92a8248a7ffb1742d70b0c` and task digest
      `bbbcc9710d38c6bd5c0a611bc68b40c24ef908f1faaf93913b26b914a0256509`.
- [x] Independent validation reproduced 240 plot-receipt hashes, 720
      source-validation receipts, and every one of 1,440 PNG/PDF output hashes.
- [x] Scientific Zarr successors persist the arrays required for radial,
      distance, trajectory, and spatial-occupancy interaction.
- [x] Root consolidated metadata exposes the v4 receipt-bound spatial
      successor; stale or missing consolidation is not the observed defect.
- [x] A pre-fix live read-only smoke against
      `2026-08-10T17-20-55Z_arena_1_goodbatbadbat` returns zero exact-successor
      Marimo options.
- [x] The failure is isolated to literal dictionary equality between a minimal
      radial relative-child binding and the spatial bundle's enriched
      receipt-bound representation of the same exact child.
- [x] The original unit fixture used the minimal binding on both sides and did
      not exercise the production receipt-bound shape.
- [x] The local reader correction passes the real-artifact smoke and
      metadata-only discovery for all 80 eligible receipt-bound v4 recordings.
- [x] Reader commit `559c08fdea42f0e4de3985033f95e99917a67a5f`
      passed all 23 required checks and merged through PR 70 as merge commit
      `33bc1c5b7f4d91348fc18ff2a8683e72761ea185`.
- [x] The modular architecture plan merged through PR 68 as merge commit
      `b1495ff7664fc80169aab583db3a765da6439660`.
- [x] The behavior-preserving package extraction is commit
      `345083fb922316767a1b58f105f26c85943d2ef7`; the separate spatial view is
      commit `d215eb686045b59aacf2a201a99fdfdcf12c88ec`.
- [x] The modular/spatial branch plus explicit multi-source choice passed all
      23 required checks and merged through PR 71 as merge commit
      `13318a8b7a16399a70f290ea3bd5ad466f309ae9`.
- [x] The exact controller-trial view passed all 23 required checks and merged
      through PR 72 as merge commit
      `d7b1cde38efce05a106c20882f7cd3b8452cc1d2`.
- [x] The generalized bout-response implementation passes 57 focused tests and
      a read-only live v4 smoke: 1,445 selected bouts, 2,890 bout-by-chaser
      rows, 30 persisted distance-band summaries, and 2,466 directed-valid
      body-frame rows.
- [x] The generalized bout-response view passed all 23 required checks and
      merged through PR 74 as merge commit
      `267553f548d0590958938d1d2227e5c87f19ec8a`.
- [x] The escape/freeze implementation passes 59 focused tests, the 163-test
      Marimo suite (148 passed and 15 expected xfails), Marimo check, and a
      read-only live v4 smoke: four trials, five events, twenty persisted
      threshold-sweep rows, one escape trial, three freeze candidates, and five
      trace-usable events.
- [x] The escape/freeze view passed all 23 required checks and merged through
      PR 75 as merge commit
      `8c6b2d7d1a1b1491098b1f82680c2fd1596edddd`.
- [x] A strict projection receipt now composes seven exact-child receipts and
      two relative-frame receipts without replacing their independent lineage
      records or becoming a selector/cache authority.
- [x] Receipt-bound interactive loads revalidate direct child metadata and
      rehash only arrays consumed by the selected renderer. Exhaustive deep
      audit remains the explicit path when no projection receipt is supplied.
- [x] The receipt-bound reader passed required CI and merged through PR 76 as
      merge commit `e0521c6bcf39859634c8d52bc4a8bc98f73cf721`.
- [x] An 84-entry direct metadata-file census found no eye-angle, subject-shape,
      or modern subject-mask source family in the frozen body-frame cohort.
- [x] The frozen cohort invocation did not request `chaser_gaze_tracking_v2`
      and supplied no eye run or gaze-convention review receipt.
- [x] The modern gaze successor retains exact body-frame gaze/bearing rows,
      summaries, and sustained lock events but does not retain the legacy
      rotated spatial controls or dynamic-lag summaries.
- [x] The paired radial successors already seal the missing CDF and observed /
      geometric-expected distribution arrays; no successor migration is needed
      for the distance-distribution reader.

## Safety invariants

- [x] Continue to require consolidated metadata for immutable published
      artifacts. Do not add an unconsolidated discovery fallback.
- [x] Resolve no `latest`, `latest_complete`, `selected`, `authoritative`,
      `default`, promoted, or name-sorted child.
- [x] Require exact recording identity, run path, manifest digest, completion,
      selector-ineligible state, provider identity, and provider digest.
- [x] Treat extra receipt evidence as a versioned schema to validate, not as
      fields to ignore and not as proof of a different scientific child.
- [x] Do not reopen or alter trial membership, semantic epochs, event classes,
      time bases, geometry, position-provider authority, or missing-data
      policy in visualization code.
- [x] Permit deterministic display projection only. Record its algorithm,
      limits, and missing-data behavior; never feed projected values back into
      scientific metrics or exports.
- [x] Keep missing source rows missing and prohibit interpolation.
- [x] Keep static PNG/PDF bundles and their JSON receipts valid independent
      publications. Interactive availability must not replace or mutate them.
- [x] Keep every new view selector-ineligible and read-only. No registry write,
      authority activation, source mutation, or scientific publication occurs
      while rendering.
- [x] Do not use older GoodCopBadCop, candidate, or legacy views as an implicit
      fallback when an exact receipt-bound bundle fails validation.

## Phase 0 — Freeze the failing production boundary

- [x] Add a production-shaped in-memory fixture whose spatial provider record
      uses exactly:
  - `run_path`;
  - `manifest_sha256`;
  - `validation_receipt_sha256`; and
  - `verification_mode=receipt_bound_targeted_array_rehash_v1`.
- [x] Keep the corresponding radial successor's relative binding in its valid
      minimal two-field form: `run_path` plus `manifest_sha256`.
- [x] Add a regression proving current literal whole-object comparison would
      suppress that otherwise valid bundle.
- [x] Add separate adversarial fixtures for wrong path, wrong manifest digest,
      malformed receipt digest, wrong verification mode, unexpected binding
      schema, provider mismatch, recording mismatch, and incomplete child.
- [x] Freeze the live smoke target, exact run path, and expected zero-option
      pre-fix result in the diagnostic note; do not encode mutable selectors in
      the test.

Primary locations:

- `tests/unit/fisheye/test_marimo_chaser_exact_successor_component.py`
- `apps/marimo/components/registry.py`
- `apps/marimo/components/chaser_exact_successors.py`

## Phase 1 — Define one exact relative-child binding validator

- [x] Add one pure helper shared by discovery and projection loading.
- [x] Define a normalized immutable child identity containing exactly:
  - exact child `run_path`; and
  - lowercase 64-character `manifest_sha256`.
- [x] Define closed accepted binding profiles rather than accepting arbitrary
      mapping supersets:
  - minimal exact-child binding v1; and
  - receipt-bound targeted-array-rehash binding v1.
- [x] For the receipt-bound profile, require a lowercase 64-character
      `validation_receipt_sha256` and the exact supported verification-mode
      identifier.
- [x] Compare normalized scientific identity across the spatial and radial
      manifests while retaining the richer spatial receipt evidence in the
      returned proof.
- [x] Reject disagreement in either core identity field even if provider IDs
      happen to match.
- [x] Reject malformed, missing, duplicated, or unrecognized profile fields
      with a typed reason suitable for tests and diagnostics.
- [x] Do not claim that the external validation receipt was reopened unless the
      reader is given and validates its exact file. The current viewer may
      instead report that the sealed spatial manifest binds its digest while
      independently rehashing the arrays it consumes.
- [x] Keep the helper independent of Marimo UI objects so registry discovery,
      loaders, CLI audits, and tests can share it.

## Phase 2 — Correct discovery and projection loading together

- [x] Replace literal binding equality in
      `discover_exact_chaser_successor_options()` with the shared validator.
- [x] Replace literal binding equality in `_verify_bundle_children()` with the
      same validator; discovery and loading must not have different admission
      grammar.
- [x] Preserve the existing exact-child reopening and manifest validation after
      normalized identity comparison.
- [x] Preserve provider ordering: keypoint first, detection second.
- [x] Preserve distinct provider identity/digest requirements.
- [x] Preserve spatial/radial agreement on semantic selection, epoch records,
      reviewed arena geometry, and relative child identity.
- [x] Include the normalized child identity, receipt digest, verification mode,
      and validation behavior in projection provenance.
- [x] Bump the synthesized explorer-spec schema or renderer version if the
      accepted binding grammar changes its public interpretation.
- [x] Ensure the capability still becomes undiscoverable on any real mismatch;
      the fix is schema-aware equivalence, not permissive fallback.

## Phase 3 — Reader and registry tests

- [x] Add a passing receipt-bound discovery test using the real enriched shape.
- [x] Add a passing load test proving radial and spatial bindings normalize to
      one exact relative child.
- [x] Add a rejection test for a valid receipt digest attached to the wrong
      relative manifest digest.
- [x] Add a rejection test for a correct path/digest with an unsupported
      verification mode.
- [x] Add a rejection test for an enriched binding with a malformed receipt
      digest.
- [x] Add a rejection test for provider-order reversal.
- [x] Add a rejection test for any forbidden selector attribute on the parent.
- [x] Add a rejection test proving missing/stale consolidated metadata does not
      fall through to an unconsolidated traversal.
- [x] Retain the existing extrema-preserving and missing-break display tests.
- [x] Add a catalog test proving only the exact-successor provider owns these
      receipt-bound options; legacy/candidate providers must not claim them.
- [x] Run focused pytest outside the Codex sandbox using `scripts/py` (25
      passed).
- [x] Run `scripts/py -m marimo check apps/marimo/palette_explorer.py` outside
      the Codex sandbox.

## Phase 4 — Real-artifact acceptance

- [x] Use commit-pinned implementation
      `d215eb686045b59aacf2a201a99fdfdcf12c88ec` at
      `/tmp/palette-authority-supplier-clarity-20260827`.
- [x] Reopen the smoke recording with consolidated metadata and require exactly
      one option matching the exact v4 spatial run. Two explicit immutable
      exact bundles are currently visible in the recording; the other is the
      older `20260825_exact_epochs_v1` candidate and is not silently selected.
- [x] Require an explicit operator source choice when discovery exposes more
      than one immutable exact bundle. Keep a sole exact bundle as the
      unambiguous default, and load no projection while the choice is unset.
- [x] Load `radial_near_field` and require deep audit of both spatial/radial
      children.
- [x] Load `distance_traces` and `trajectory_overlays`; require content hashes
      for every relative-frame array the selected projection consumes.
- [x] Require all declared analyses to retain the exact recording ID, provider
      IDs/digests, run paths, child manifest digests, semantic epoch binding,
      and geometry authority.
- [x] Record load time, arrays read, bytes read where measurable, and display
      point counts. The commit-pinned spatial smoke discovered in 0.304 s,
      loaded in 3.758 s, deep-audited 19 spatial plus 61+61 radial arrays
      (324,573 decoded bytes), loaded no relative arrays, and rendered 15,876
      heatmap cells. Performance evidence did not weaken validation.
- [x] Run metadata-only discovery over all 80 eligible recordings and require
      one exact option per recording.
- [ ] Deep-load a deterministic representative sample covering arenas,
      recording dates, body-frame source variants, and the previously long
      task-2 recording.
- [x] Keep indices `77-80` explicitly excluded from successful-cohort claims.
- [x] Write a digest-bound smoke report. It may describe viewer readiness but
      must not activate a selector or alter any recording.

## Phase 4.5 — Establish the modular exact-chaser explorer boundary

The recording explorer remains one supported Marimo entrypoint, but analysis
implementations must be ordinary focused Python components. Do not grow
`palette_explorer.py`, `chaser_exact_successors.py`, or the legacy
`goodcopbadcop_chaser.py` into a new monolith. The durable design is recorded
in [`marimo_explorer_architecture.md`](../marimo_explorer_architecture.md).

- [x] Preserve `apps/marimo/palette_explorer.py` as the thin application shell
      for workspace selection, provider/analysis selection, generic errors,
      stale-load protection, and final layout.
- [x] Define one exact-chaser provider adapter with closed maps for available
      analysis IDs, projection loaders, optional controls, and renderers.
- [x] Replace exact-chaser analysis-specific top-level load/render branches
      with one provider-adapter invocation; adding a later exact analysis must
      not require another notebook branch.
- [x] Extract the current exact component into focused modules for shared
      projection, radial/near-field, distance, trajectory, and provenance.
- [x] Keep `apps/marimo/components/chaser_exact_successors.py` as a temporary
      compatibility facade that re-exports the supported public functions.
- [x] Put spatial occupancy in its own focused module.
- [x] Put escape/freeze in its own focused renderer plus audited discovery,
      contract, and loader modules. Controller-trial and generalized
      bout-response retain their separate focused modules.
- [ ] Put the later full-profile view in its own focused module.
- [x] Keep legacy GoodCopBadCop code isolated. Do not add new exact-successor
      logic, fallback behavior, or scientific interpretation to that surface.
- [x] Keep discovery metadata-only. Importing a provider or listing analysis
      IDs must not read scientific arrays.
- [x] Load only the selected analysis projection and key any cache by exact
      archive, run, manifest, renderer, and display-parameter identities.
- [x] Prevent late asynchronous results from rendering under a newer recording,
      provider, source, or analysis selection.
- [x] Add focused tests per module plus one facade/provider/top-level routing
      test proving closed dispatch and typed unavailable behavior.
- [x] Land the mechanical extraction as a distinct commit from the first new
      spatial-occupancy implementation so review can distinguish structural
      movement from behavior: `345083fb` then `d215eb68`.

## Phase 5 — Mount the remaining persisted scientific views

Implement these as separate reviewable packages after the reader unblock. Each
package must use persisted successor arrays and exact child bindings; it must
not derive a new scientific product in the UI.

### 5A — Spatial-occupancy heatmaps

- [x] Add an exact `spatial_occupancy` analysis ID.
- [x] Read persisted `occupancy_density_valid_in_arena`, bin edges, arena-bin
      mask, epoch/provider codes, and candidate/validity denominators.
- [x] Reproduce provider/epoch ordering and the detection-minus-keypoint
      display from sealed arrays.
- [x] Bind the 2 mm grid policy, reviewed circle, coordinate orientation,
      missing-row policy, color normalization, and coverage annotations.
- [x] Treat color scale and point/texture resolution as display parameters;
      never renormalize scientific occupancy in Marimo.
- [ ] Extract one shared scientific display projection used by the static and
      Marimo heatmap renderers so validation, bin coordinates, denominators,
      normalization, and scale derivation cannot drift.
- [ ] Keep conditional valid-in-arena occupancy primary and expose the already
      persisted candidate-epoch-normalized surface as an explicitly
      coverage-sensitive companion; do not derive either surface in Marimo.

### 5A.1 — Exact per-epoch distance distributions

- [ ] Add a closed `distance_distributions` exact analysis route rather than
      reviving legacy `hist_density` or candidate-provider viewers.
- [ ] Add a receipt-bound targeted roster for the exact `cdf_*` arrays and
      radial observed/expected probability-mass arrays from both provider
      successors.
- [ ] Render paired-provider CDFs by exact epoch x behavior x chaser stratum.
- [ ] Render persisted observed radial probability mass against the persisted
      moving-reference geometric expected fraction on exact stored bin edges.
- [ ] Expose the 5 mm wall-excluded observed/expected twins explicitly and
      preserve their distinct denominators.
- [ ] Share the scientific projection/parameter builder with the current static
      detailed CDF recipe; do not recompute a histogram from relative-frame
      rows or permit interactive rebinning.
- [ ] Record child receipt digests, consumed array rosters, exact bin edges,
      denominators, provider identities, and display-only choices in figure
      provenance.

### 5B — Controller-trial and trigger-aligned views

- [x] Anchor the exact controller-trial successor by run path, manifest digest,
      semantic selection, recording identity, and relative-frame source.
- [x] Render exact logged active trial members only; retained trial gaps remain
      evidence but do not become trial members.
- [x] Display full/trial distance traces from persisted membership and timing
      arrays without inferring legacy trial boundaries.
- [x] Preserve exact session-time and relative-frame semantics in labels and
      provenance.

### 5C — Generalized bout response

- [x] Mount persisted bout-response rows, exact bout start/end response
      intervals, bout identities, kinematics, and source validity. The
      successor has no independent post-bout response window, so the viewer
      does not synthesize one.
- [x] Preserve the source swim-bout segmentation and controller-trial binding;
      do not resegment bouts or reassign events in the viewer.
- [x] Expose persisted distance-stratified summaries and raw body-frame
      response rows only where the product declares the required source and
      validity fields; do not re-bin or re-aggregate scientific summaries.
- [x] Keep body-frame missing-source and present-invalid-axis states distinct
      in the callout and display provenance, with no motion-heading fallback.

### 5D — Escape/freeze

- [x] Mount persisted trial/event summaries and reason/validity codes.
- [x] Preserve the exact escape/freeze classifier/version and source window
      definitions.
- [x] Render trial outcomes, event outcome facts, recapture facts, and
      trace-validity reasons without recomputing response classes from
      displayed/downsampled data.
- [ ] Add event-aligned distance-trace trajectories only after a new immutable
      successor version persists the aligned samples and their window identity.
      Version 2 persists recapture outcomes and trace-validity reasons but no
      aligned distance samples, so the current viewer explicitly does not
      reconstruct a trace.

### 5E — First-class exact bearing and gaze

- [ ] Add a keypoint-only anatomical body-bearing polar view from the persisted
      relative-frame body extension. State explicitly that detection bbox
      centroid remains first-class for position/distance but supplies no body
      axis or anatomical bearing.
- [ ] Define first-class gaze as a conditional capability of the exact chaser
      provider, not as a third position provider and not as a legacy fallback.
- [ ] Version the modern gaze successor before cohort materialization to seal
      the rotated spatial null controls and required real-minus-virtual
      recording summaries. Preserve the exact reviewed arena geometry,
      rotation list, collision exclusion, distance/accessibility gates, and
      null denominators in provenance.
- [ ] Decide and persist the required dynamic tracking surface. The legacy
      component includes zero-lag and causal best-lag gain; the current modern
      successor does not. Do not leave the catalog promising a metric absent
      from the payload.
- [ ] Keep lock occupancy descriptive: persist and display its exact valid,
      accessible, and locked counts alongside the fraction.
- [ ] Keep sustained lock events as contiguous exact-frame intervals with
      timestamp-derived duration, explicit 0.10-second default, sample count,
      and median absolute error. Do not bridge invalid gaps in the viewer.
- [ ] Add a closed gaze array roster, exact discovery proof, projection loader,
      renderer, and typed unavailable state. Require the exact recording,
      keypoint relative-frame/body extension, semantic selection, reviewed eye
      run logical digest, and gaze-convention receipt digest.
- [ ] Add exact gaze-child validation receipts and a versioned projection
      composition schema that requires gaze for a gaze-capable receipt. A
      supplied v1 composition receipt must not silently fall back to a deep
      audit for gaze.
- [ ] Add body-bearing polar, gaze-versus-bearing, gaze-error, tracking,
      lock-fraction, lock-event, and real-versus-rotated-control panels from
      persisted arrays only.
- [ ] Add cohort planning fields for an explicit eye-angle run and exact
      convention receipt per recording; never resolve an eye selector or infer
      biological direction.
- [ ] Materialize gaze only after modern subject-mask -> subject-shape ->
      compact-v7 eye-angle prerequisites and their review/authority decisions
      exist for the cohort.

### 5F — Full profile

- [ ] Define an exact full-profile bundle or readiness envelope that binds the
      controller, bout-response, escape/freeze, radial, spatial, and relative
      children required by the composed dashboard.
- [ ] Do not infer cross-module compatibility from a shared run-name string.
- [ ] Include gaze only when one complete gaze successor is explicitly bound to
      the same recording and coordinate/time authorities; retain a truthful
      partial profile when gaze is unavailable.
- [ ] Represent unavailable modules explicitly rather than substituting a
      legacy view.

## Phase 6 — Decide and implement durable interactive descriptors

This is a design gate, not a prerequisite for the reader compatibility fix.

- [ ] Decide whether runtime synthesis from sealed Zarr manifests remains the
      only interactive descriptor or whether generic viewers need a persisted
      descriptor.
- [ ] If runtime synthesis remains authoritative:
  - version the renderer and spec schema;
  - record the exact display algorithms and limits in the generated spec and
    figure metadata;
  - expose the application/software commit in operational provenance; and
  - test deterministic spec generation from the same sealed manifests.
- [ ] If a persisted descriptor is required, publish it as a separate immutable
      visualization artifact or versioned visualization run. Do not add fields
      to or rewrite the completed scientific successor.
- [ ] A persisted descriptor must bind:
  - recording ID and exact anchor run path;
  - anchor and child manifest digests;
  - renderer ID/version and supported analysis IDs;
  - display-only algorithms, limits, missing-data policy, and interpolation
    prohibition;
  - required source-validation or plot-receipt digests when those files are
    actual inputs;
  - software commit/environment identity; and
  - its own canonical payload digest.
- [ ] Keep plot-receipt JSON as a static export receipt. Do not call it an
      interactive spec merely because it contains plot parameters.
- [ ] If a descriptor is stored inside a recording Zarr, use a new immutable
      visualization publication, consolidate only after all writes complete,
      and validate the new consolidated generation before discovery.

## Phase 7 — Performance and interactive safety

- [x] Keep discovery metadata-only and consolidated.
- [x] Retain exhaustive deep audit as the explicit no-receipt reader path.
- [x] For receipt-bound views, hash only the exact arrays required by the chosen
      view and record which declarations were checked.
- [x] Preserve missing-data breaks during trace reduction.
- [x] Preserve source order and coordinate extrema during trajectory reduction.
- [x] Put hard, recorded display limits on points, traces, and panels.
- [ ] Cancel stale Marimo loads when a recording/analysis selection changes;
      never let a late result render under another selection.
- [ ] Cache only by exact archive identity, run path, manifest digest, renderer
      version, and display-parameter digest.
- [ ] Never let a cache hit bypass manifest, consolidated-generation, or
      recording-identity validation.
- [ ] Keep viewer failures explicit and typed. A hidden capability is safer
      than a legacy fallback, but the UI should explain the rejected proof when
      requested by an operator.

Receipt-bound implementation evidence on the live target
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`:

- the projection receipt validates seven exact immutable children and two
  relative-frame children by exact path and receipt digest;
- the escape/freeze load fell from the 12.971-second exhaustive audit to 4.151
  seconds;
- it loaded and rehashed exactly 53 escape/freeze arrays, loaded zero arrays
  from spatial, radial, controller, bout, or relative children, and then
  completed renderer-level scientific validation in 0.001 seconds;
- its selection and figure provenance retain the projection-receipt digest,
  each consumed child receipt digest, verification mode, and verified array
  roster; and
- this is read-only experimental evidence. The temporary `/tmp` composition
  receipt is not a durable cohort publication and does not make this branch
  merge-ready before required CI.

One warm-cache all-route acceptance sweep then passed with these bounded loads:
radial 1.721 s (15+15 arrays), distance 3.153 s (13+13 relative arrays),
trajectory 2.900 s (13+13 relative arrays), spatial 1.327 s (14 arrays),
controller 3.511 s (25 controller plus 13+13 relative arrays), generalized
bout response 2.803 s (38 bout arrays), and escape/freeze 3.275 s (53 escape
arrays). Metadata-only dependencies loaded zero arrays in every applicable
route.

## Phase 8 — Documentation, CI, and release evidence

- [ ] Update the status note from `implementation: partial` only after the live
      smoke and required CI pass.
- [ ] Document exactly which static figure families have interactive
      equivalents and which remain static-only.
- [x] Update recording-explorer architecture documentation with the modular
      exact-successor provider/package plan.
- [ ] Record the durable interactive-descriptor decision separately when that
      Phase 6 design gate is resolved.
- [ ] Add a required boundary test covering producer-shaped receipt-bound Zarr
      metadata -> consolidated discovery -> projection loader -> Marimo output.
- [ ] Run all required CI checks. Failed, cancelled, skipped-after-failure, or
      timed-out checks remain blocking.
- [ ] Do not merge, update the shared `/groups` checkout, or describe the branch
      as complete/merge-ready before every required check is green.
- [ ] If a commit-pinned experimental deployment is used for real-artifact
      smokes, keep it selector-ineligible and record the exact deployment path
      and commit.
- [ ] Preserve the existing scientific and static-publication receipts; viewer
      release evidence is additive.

## Acceptance matrix

| Surface | Required evidence | Failure behavior |
| --- | --- | --- |
| Discovery | consolidated exact anchor, closed binding profiles, exact child/provider identities | capability absent with typed diagnostic; no fallback |
| Radial/near field | deep-audited persisted radial/spatial arrays and shared epoch/geometry proof | panel unavailable |
| Distance traces | targeted relative-array hashes, exact session time, missing-break preservation | panel unavailable; no interpolation |
| Trajectories | targeted position/chaser hashes, reviewed arena, exact epochs | panel unavailable; no alternate provider |
| Spatial heatmaps | persisted density, denominators, bin edges/mask, fixed grid/orientation | panel unavailable; no viewer recomputation |
| Distance distributions | paired persisted CDF and observed/geometric expected arrays on exact bins | panel unavailable; no framewise histogram |
| Body bearing | exact keypoint body extension and anatomical sign convention | keypoint panel unavailable; no detection substitution |
| Gaze | exact eye payload/review, body bearing, semantic selection, rotated-control successor and receipt child | capability unavailable; no legacy or deep-audit fallback under a supplied v1 receipt |
| Trial/bout/escape | exact module manifest and source bindings; persisted membership/classification | module explicitly unavailable |
| Full profile | exact cross-module digest bundle/readiness envelope | partial modules labeled; no run-name inference |
| Static artifacts | canonical plot receipt and file SHA-256 | static file rejected independently of Marimo |
| Persisted interactive descriptor, if adopted | immutable spec, renderer/version, exact source digests, self-digest | descriptor rejected; scientific Zarr remains valid |

## Definition of done

The immediate reader package is done only when the real smoke target discovers
exactly one receipt-bound option and every currently declared exact analysis loads
through consolidated exact-path reads, production-shaped and adversarial tests
pass, Marimo check passes, and every required CI check is green. This does not
claim full visualization parity.

Full interactive parity is done only when every intended static scientific
figure family has an exact persisted-array adapter or is explicitly documented
as static-only/deferred, all 80 eligible recordings pass metadata discovery,
representative deep-load smokes pass, and no viewer path resolves a selector,
falls back to legacy/candidate data, mutates a recording, or recomputes a
scientific claim from display-projected values.
