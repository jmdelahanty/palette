# Exact Chaser Successor Interactive Visualization Implementation Checklist

<!-- contract-meta
status: implementation-in-progress
implementation: reader-merged-modular-spatial-locally-validated-ci-pending
last_verified: 2026-08-29
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
- [ ] Required CI for the new modular/spatial branch has not run. Local and
      real-artifact validation do not make that branch merge-ready.

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
- [ ] Put later controller-trial, generalized bout-response, escape/freeze, and
      full-profile views in their own focused modules.
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

### 5B — Controller-trial and trigger-aligned views

- [ ] Anchor the exact controller-trial successor by run path, manifest digest,
      semantic selection, recording identity, and relative-frame source.
- [ ] Render exact logged active trial members only; retained trial gaps remain
      evidence but do not become trial members.
- [ ] Display full/trial distance traces from persisted membership and timing
      arrays without inferring legacy trial boundaries.
- [ ] Preserve exact session-time and relative-frame semantics in labels and
      provenance.

### 5C — Generalized bout response

- [ ] Mount persisted bout-response rows, response windows, bout identities,
      kinematics, and source validity.
- [ ] Preserve the source swim-bout segmentation and controller-trial binding;
      do not resegment bouts or reassign events in the viewer.
- [ ] Expose distance-stratified and body-frame response summaries only where
      the persisted product declares the required source/validity fields.
- [ ] Keep body-frame missing-source and present-invalid-axis states distinct.

### 5D — Escape/freeze

- [ ] Mount persisted trial/event summaries and reason/validity codes.
- [ ] Preserve the exact escape/freeze classifier/version and source window
      definitions.
- [ ] Render trial outcomes and event traces without recomputing response
      classes from displayed/downsampled data.

### 5E — Full profile and optional gaze

- [ ] Define an exact full-profile bundle or readiness envelope that binds the
      controller, bout-response, escape/freeze, radial, spatial, and relative
      children required by the composed dashboard.
- [ ] Do not infer cross-module compatibility from a shared run-name string.
- [ ] Add gaze/controller-trial views only when one complete gaze successor is
      explicitly bound to the same recording, trial axis, and coordinate/time
      authorities.
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

- [ ] Keep discovery metadata-only and consolidated.
- [ ] Deep-audit small successor tables before rendering.
- [ ] For frame-scale arrays, hash only the exact arrays required by the chosen
      view and record which declarations were checked.
- [ ] Preserve missing-data breaks during trace reduction.
- [ ] Preserve source order and coordinate extrema during trajectory reduction.
- [ ] Put hard, recorded display limits on points, traces, panels, and memory.
- [ ] Cancel stale Marimo loads when a recording/analysis selection changes;
      never let a late result render under another selection.
- [ ] Cache only by exact archive identity, run path, manifest digest, renderer
      version, and display-parameter digest.
- [ ] Never let a cache hit bypass manifest, consolidated-generation, or
      recording-identity validation.
- [ ] Keep viewer failures explicit and typed. A hidden capability is safer
      than a legacy fallback, but the UI should explain the rejected proof when
      requested by an operator.

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
| Trial/bout/escape | exact module manifest and source bindings; persisted membership/classification | module explicitly unavailable |
| Full profile | exact cross-module digest bundle/readiness envelope | partial modules labeled; no run-name inference |
| Static artifacts | canonical plot receipt and file SHA-256 | static file rejected independently of Marimo |
| Persisted interactive descriptor, if adopted | immutable spec, renderer/version, exact source digests, self-digest | descriptor rejected; scientific Zarr remains valid |

## Definition of done

The immediate reader package is done only when the real smoke target discovers
exactly one receipt-bound option, all four currently declared analyses load
through consolidated exact-path reads, production-shaped and adversarial tests
pass, Marimo check passes, and every required CI check is green. This does not
claim full visualization parity.

Full interactive parity is done only when every intended static scientific
figure family has an exact persisted-array adapter or is explicitly documented
as static-only/deferred, all 80 eligible recordings pass metadata discovery,
representative deep-load smokes pass, and no viewer path resolves a selector,
falls back to legacy/candidate data, mutates a recording, or recomputes a
scientific claim from display-projected values.
