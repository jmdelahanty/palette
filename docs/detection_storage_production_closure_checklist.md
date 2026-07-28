# Detection Storage Production Closure Checklist

Status: active contract-closure gate; production selectors remain unchanged

Date: 2026-07-27

## Goal

Finish the canonical-detection v1 and refined-detection v1 public contracts
before native writers, review tools, downstream materializers, or selectors are
routed through them. This is the current source of truth for production adoption;
older benchmark phase lists remain historical evidence.

Contract closure is not the same as making a run authoritative. A contract is
closed when every persisted value and state transition has one exact meaning and
an executable validator. Authority changes occur only in a later canary gate.

## Native Canonical Provenance Boundary

The canonical array schema remains
`palette.canonical_detection.run_schema` v1 for both historical conversion and
new detector output. The persisted run-manifest version identifies how those
arrays were produced:

- run-manifest v1 accepts only
  `palette.canonical_detection.legacy_source_evidence` v1; and
- run-manifest v2 accepts only
  `palette.canonical_detection.native_source_evidence` v1.

Native writers must call `build_native_detection_source_evidence()` followed by
`build_native_canonical_detection_run_manifest()`. The evidence binds recording
identity, exact frame and pixel authority records, source dimensions, model
artifact digest, producer identity, and a digest-verified strict-JSON run
provenance document. The existing v1 builder deliberately rejects that evidence,
and the v2 builder deliberately rejects legacy-conversion evidence.

This distinction does not authorize publication. Native output remains
selector-ineligible until Palette's complete publication gate passes and Crimson
accepts the v2 manifest envelope carrying logical schema v1.

## Already Frozen

- [x] Canonical raw schema: nine exact arrays, dtypes, row ordering, geometry,
      stable `instance_key`, and `F+1` `frame_row_offsets`.
- [x] Canonical logical schema v1 is independent of its run-manifest envelope:
      run-manifest v1 binds legacy-conversion evidence, while run-manifest v2
      binds exact native-detector source and provenance evidence.
- [x] Refined schema: 28 full-acquisition arrays, independent instance/source
      offsets, stable refined-row identity, source audit, manual-row semantics,
      and exact reason registries.
- [x] Zero, one, or many detections per frame; no sentinel observation rows.
- [x] `bbox_norm_coords` authority with required exact float32 image-space
      projections.
- [x] Zarr v3 manifests, exact storage declarations, strict JSON, and direct/
      consolidated metadata equivalence.
- [x] Versioned `detection_published_access_aware_v1` physical profile and
      `detection_regular_rollback_v1` rollback profile.
- [x] Full-duration Palette/Crimson evidence supporting the access-aware profile.
- [x] Fail-closed refined-first selection semantics with default-denied raw
      fallback.
- [x] Delta-v2 add, replace, delete, restore, conflict, generation, and immutable
      partition semantics.
- [x] Deterministic base-plus-delta resolution and local selector-ineligible
      compaction.
- [x] Atomic selector-ineligible placement of canonical/refined snapshot pairs in
      recording archives.

## Contract Rules That Must Not Drift

1. Logical schema versions and physical profile IDs evolve independently.
2. Raw detections and compacted refined snapshots are immutable.
3. Interactive edits append immutable delta partitions; they never mutate raw or
   refined snapshots in place.
4. `instance_key` identifies an observation/edit lineage, not an animal or track.
5. Frame lookup is the exact half-open range
   `[frame_row_offsets[f], frame_row_offsets[f+1])`.
6. Count arrays are compatibility derivations from offsets and are not canonical
   v1 arrays.
7. Published immutable arrays use the shared byte-budget planner and named codec
   profile; writers do not choose raw chunk/shard literals.
8. A candidate remains selector-ineligible until its decoded content, direct
   metadata, consolidated metadata, lineage, and downstream completeness all
   validate.
9. An invalid explicit refined request is terminal and never silently falls back
   to raw data.
10. Selector activation is an owner- and generation-guarded final metadata
    transaction; scientific payload is never rewritten during activation.

## Gate A — Mergeable Contract Foundation

- [x] Keep executable array contracts, stage schemas, planners, manifests, and
      validators under `fisheye.shared.zarr`.
- [x] Keep snapshot publication selector-ineligible and registry-free.
- [x] Integrate the reusable detection-snapshot DAG boundary.
- [x] Keep the refined-to-crop handoff as a non-authorizing boundary prototype;
      it is not a crop storage or publication contract.
- [ ] Reconcile this branch with current `sun` and resolve tests without weakening
      either contract.
- [ ] Merge one reviewed checkpoint before another agent routes native writers.

Exit gate: downstream work can import one committed contract API rather than
copying schema, storage, manifest, or selection rules.

## Gate B — Native Canonical Writer

- [x] Add a strict native source-evidence builder and validator without
      weakening or reinterpreting the accepted legacy-conversion manifest v1.
- [ ] Adapt `detect_yolo` logical output to all nine canonical v1 arrays.
- [ ] Derive exact int64 `frame_row_offsets`; do not publish `frame_counts` or
      `n_detections` in the canonical v1 run.
- [ ] Cast canonical geometry once to float32 and derive image-space projections
      from those persisted values.
- [ ] Create arrays only through the shared array factory and resolved storage
      plan.
- [ ] Make each worker own complete non-overlapping physical chunks or shards.
- [ ] Construct on node-local scratch and atomically copy a fresh immutable run
      back to the recording archive.
- [ ] Persist logical schema, storage profile, resolved plans, source identity,
      worker ownership, and complete validation receipts.
- [ ] Retain the existing writer as an explicit rollback/compatibility path.
- [ ] Confirm Crimson accepts canonical logical schema v1 carried by native
      run-manifest v2 before any native run becomes selector-eligible.

Tests:

- [ ] empty run and long empty-frame gaps;
- [ ] one and multiple detections per frame;
- [ ] deterministic keys, ordering, offsets, and derived geometry;
- [ ] exact dtype and unexpected-array rejection;
- [ ] interrupted write, failed validation, failed consolidation, and copy-back
      tombstone behavior; and
- [ ] exact Palette and Crimson reads with no dtype probing.

Exit gate: one selector-ineligible native canary matches the compatibility-built
snapshot logically and physically.

## Gate C — Refined Edit And Compaction Lifecycle

- [ ] Route review saves to bounded delta-v2 partitions instead of mutating a
      refined snapshot.
- [ ] Allocate event sequences, refined row IDs, and manual instance keys under
      one lineage-owned writer boundary.
- [ ] Freeze generation `G`, open `G+1`, and allow editing to continue while `G`
      compacts.
- [ ] Bind compaction derivation evidence into the authoritative publication
      envelope without changing the frozen refined-v1 array contract.
- [ ] Copy the compacted successor to a fresh run group atomically; never replace
      its base.
- [ ] Verify parent identity, retired-ID nonreuse, source-audit changes, rebuilt
      offsets, and exact decoded arrays after shared copy-back.
- [ ] Add crash injection around partition commit, generation rollover,
      compaction, copy-back, consolidation, and acknowledgement.

Exit gate: a manual addition and a geometry correction produce a fully validated
immutable successor while the editor remains available on the next generation.

## Gate D — Downstream Completeness

Before refined authority changes, calculate the exact old/new key and source-
signature difference and require complete successor evidence for every required
dependent family.

- [ ] Freeze the invalidation receipt schema and required family set.
- [ ] Crop: copy matching keyed rows and compute new or changed observations.
- [ ] Keypoints: copy matching keyed rows and recompute invalidated observations.
- [ ] Subject masks: preserve dense editable authority and regenerate invalidated
      derived caches explicitly.
- [ ] Tracking and analysis: either rebuild against the new rowset identity or
      record why the product is unaffected.
- [ ] Training exports: bind the exact refined snapshot and reject stale coverage.
- [ ] Require exact key coverage, source signatures, and product manifest digests;
      row counts alone are insufficient.

Exit gate: activation evidence proves that every required downstream product is
complete for the candidate refined snapshot or is explicitly absent by contract.

## Gate E — Fail-Closed Activation

- [ ] Freeze the exact activation receipt and proof token.
- [x] Freeze the pure activation-intent manifest transformation so the final
      manifest and consolidated metadata are staged while the run remains
      selector-ineligible.
- [ ] Require the promoted physical profile or the named rollback profile.
- [ ] Require complete run state, manifest validity, source binding, decoded
      logical digest, and direct/consolidated equivalence.
- [ ] Require an approved refined authority envelope with intended use.
- [ ] Require the downstream-completeness receipt when the refined rowset or any
      source signature changed.
- [ ] Acquire one owner/generation lease before parent-selector mutation.
- [ ] Publish refined-first selectors according to the frozen selection contract.
- [ ] Make `stage_selector_eligible=true` the literal final commit write.
- [ ] On failure, restore only metadata still owned by the activation attempt and
      retain an ineligible tombstone.
- [ ] Update the registry only after a fresh post-commit validation succeeds.

Exit gate: injected failures at every pre-commit write leave the old authority
usable; an acknowledged commit resolves one exact new authority in Palette and
Crimson.

## Gate F — Canary And Default Adoption

- [ ] Run one raw-only selector-ineligible native canary.
- [ ] Run one refined delta/compaction selector-ineligible canary.
- [ ] Validate both through Palette and Crimson exact-schema readers.
- [ ] Perform one explicitly approved selector activation with rollback evidence.
- [ ] Confirm registry, selector, run manifest, and consolidated metadata agree.
- [ ] Confirm no stale downstream product is selectable.
- [ ] Observe publication time, peak RSS, object count, and read behavior; this is
      a regression check, not a new profile search.
- [ ] Make the v1 writer path the default only after the canary passes.

## Parallel-Work Boundary

The shared-storage owner freezes schemas, planners, manifests, validation, and
activation evidence. Producer/DAG work may map native values and dependencies to
those APIs, but must not duplicate or reinterpret their policies. Until Gate A is
merged, all new output remains selector-ineligible and registry-free.
