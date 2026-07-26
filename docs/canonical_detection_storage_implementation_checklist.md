# Canonical Detection Storage Implementation Checklist

Status: active; production adoption blocked on the paired Crimson full-archive
gate

Date established: 2026-07-23

Last updated: 2026-07-25

## Goal

Define and adopt one future-facing, versioned storage contract for canonical
Palette detections. The contract must give Palette writers, training promotion,
benchmarks, publication validation, and Crimson one exact interpretation of
every array under `detect_runs/<run>`.

This checklist separates logical-schema decisions from physical-storage tuning.
A chunk or shard benchmark must not silently redefine dtype, coordinate meaning,
row identity, or requiredness.

## Scope

Included:

- canonical raw detections in `detect_runs`;
- frame-to-observation indexing used to read canonical detections;
- shared logical contracts, stage bindings, storage plans, and manifests;
- write, read, publication, metadata, and object-count benchmarks;
- `detect_yolo` adoption after the contract and benchmark gates pass;
- explicit Crimson handling for the new contract and existing archives.

Deferred:

- `detection_artifact_runs`, except for inventory and compatibility awareness;
- precision experiments below `float32`;
- migration of detection quality, refined detection, and training outputs until
  the canonical raw-detection slice is complete;
- unrelated stage writers.

## Locked Decisions

- [x] Logical array schema and physical storage profile are separate contracts.
- [x] Storage planning derives row depth from uncompressed bytes per row, not
      from one universal frame/row count.
- [x] Canonical detection continuous geometry targets exact `float32`.
- [x] Current `float64` detection geometry is an explicit transition/legacy
      representation, not the new target.
- [x] `float16`, normalized `uint16`, and fixed-point integer geometry are
      deferred until canonical storage specifications and consumers are complete.
- [x] Any later representation change requires a new schema version or
      representation ID plus numerical and downstream-behavior validation.
- [x] `detection_artifact_runs` is immutable, selector-ineligible quarantined
      evidence and is not a first-wave implementation or benchmark target.
- [x] Immutable published arrays should normally be sharded; actively mutable
      dense authorities remain chunked unless their editing model changes.
- [x] Published immutable profiles use validated consolidated metadata as their
      external read surface; mutable/in-progress stores use direct metadata.
- [x] Defer Pydantic manifest models until at least one additional stage reveals
      the genuinely shared envelope; keep exact array validation independent of
      serialization-framework coercion.

## Phase 0 — Foundation And Checkpoint

- [x] Create the byte-budget `StoragePlan` foundation.
- [x] Add versioned logical `ArrayContract` types and initial examples.
- [x] Add common benchmark-envelope types.
- [x] Census `ArraySpec` declarations and physical array-creation sites.
- [x] Generate the detection-family schema inventory.
- [x] Classify `detection_artifact_runs` as deferred quarantined evidence.
- [x] Record `float32` as the first canonical detection-geometry dtype.
- [x] Commit the dtype decision and this execution checklist as a checkpoint
      (`eb94f885`).

Exit gate:

- [x] Working tree was clean at named checkpoint `eb94f885` before the producer
      and consumer census began.

## Phase 1 — Canonical Detection Consumer And Producer Census

Trace every current producer and consumer of the canonical detection arrays.
Record code paths separately from documentation and historical compatibility.

- [x] Census Palette writers for every `detect_runs/<run>` array.
- [x] Census Palette readers, validators, refiners, crop builders, exporters,
      training promotion, diagnostics, and review tools.
- [x] Census Crimson reads and current typed-probe fallbacks through the shared
      Palette–Crimson contract or Crimson source review.
- [x] Record which consumers read whole arrays, frame windows, individual frames,
      or observation rows.
- [x] Record whether readers assume rows are ordered contiguously by frame.
- [x] Record every missing-value, empty-run, fill-value, and sentinel assumption.
- [x] Record all current dtype alternatives as compatibility evidence rather
      than candidate canonical dtypes.

Resolve these questions explicitly:

- [x] Is `bbox_norm_coords` the authoritative geometry representation?
- [x] Are `bbox_img_xyxy` and `centers_img_xy` required materialized derivatives,
      optional caches, or reader-derived values?
- [x] Are `frame_counts` and `n_detections` true aliases, and if so which name is
      canonical?
- [x] Should canonical storage add `frame_row_offsets` with shape
      `[n_frames + 1]` for direct frame-to-row lookup?
- [x] What ordering guarantee applies to detection rows?
- [x] What are the legal class-ID range, signedness, and missing-value rules?
- [x] Which arrays are required for an empty but valid detection run?

Deliverable:

- [x] Check in one authoritative array-role table with columns for path, dtype,
      shape, authority, requiredness, access pattern, producer, and consumers.

Exit gate:

- [x] No unresolved authority, alias, indexing, ordering, dtype, or sentinel
      question remains for the first canonical schema.

Phase 1 evidence and decisions are recorded in
[`diagnostics/canonical_detection_producer_consumer_census_2026-07-23.md`](diagnostics/canonical_detection_producer_consumer_census_2026-07-23.md).

## Phase 2 — Versioned Canonical Detection Stage Schema

- [x] Add a versioned stage/run schema type that binds concrete paths to logical
      `ArrayContract` versions.
- [x] Define the canonical run schema ID and version.
- [x] Define symbolic dimensions such as `n_frames` and `n_instances`.
- [x] Define exact contracts for all accepted canonical arrays.
- [x] Use exact `float32` for bounding boxes and centers.
- [x] Define axis names, coordinate spaces, units, fill/null semantics, and
      requiredness.
- [x] Define row identity through unique `instance_key` and full-acquisition
      frame lineage; retain recording-bound key derivation validation in the
      publication contract.
- [x] Define frame-index bounds and the accepted row-ordering invariant.
- [x] Define instance/offset cardinality invariants.
- [x] Exclude `frame_counts` and `n_detections` from canonical bindings.
- [ ] Define count derivation only in explicit compatibility adapters.
- [x] Define the valid zero-observation representation.
- [x] Serialize the schema and concrete bindings into JSON-safe manifest records.
- [ ] Represent existing `float64` archives through an explicit compatibility
      adapter/profile rather than a union in the canonical contract.

Tests:

- [x] Exact dtype acceptance and rejection.
- [x] Shape and symbolic-dimension validation.
- [x] Frame-bound and row-order validation.
- [x] Instance/offset consistency.
- [ ] Compatibility count derivation from offsets.
- [x] Derived-array consistency where required.
- [x] Empty-run validation.
- [x] Manifest round-trip and stable schema identity.

Exit gate:

- [x] The canonical v1 schema can validate an in-memory canonical run without
      importing or invoking a production writer.

## Phase 3 — Storage Intents And Planning Report

- [x] Assign an access pattern to every canonical array.
- [x] Assign immutable write mode to raw canonical detection publication.
- [x] Define access units for per-observation and per-frame reads.
- [x] Produce a `StoragePlan` for every array from exact shape and dtype.
- [x] Verify that inner chunks are derived from byte budgets.
- [x] Verify that outer shards contain whole inner chunks.
- [x] Estimate logical bytes, chunk count, shard count, and metadata/object count.
- [x] Produce a representative report for approximately `1.18M` frames and its
      observed detection-row cardinality.
- [x] Record any small arrays that intentionally remain a single chunk/object.
- [x] Confirm planned worker partitions own complete, non-overlapping physical
      chunks and shards.

Exit gate:

- [x] Every canonical array has an explainable plan tied to its logical schema,
      access pattern, and byte size; no writer-specific row literal is required.

Phase 3 evidence is recorded in
[`diagnostics/canonical_detection_storage_plan_2026-07-24.md`](diagnostics/canonical_detection_storage_plan_2026-07-24.md).

## Phase 4 — Safe Detection Storage Benchmarks

Benchmark only disposable stores derived from the noncanonical safe copy. Never
write benchmark candidates into a canonical recording or training Zarr.

- [x] Record source-store identity and checksums.
- [x] Materialize a controlled `float32` benchmark input from current `float64`
      geometry and record the conversion provenance.
- [x] Validate the canonical source before timed writes and require exact
      destination digests before accepting a result.
- [x] Sweep `128 KiB`, `512 KiB`, `1 MiB`, and `2 MiB` inner-chunk targets in
      the first bounded cluster repetition.
- [x] Resolve `8 MiB`, `32 MiB`, `128 MiB`, and `512 MiB` shard targets and
      deduplicate the targets that produce identical physical plans at 200k.
- [x] Complete an initial same-input regular-versus-indexed-sharding smoke.
- [x] Measure sequential write time, logical bytes, and peak memory.
- [x] Measure sharded publication/copy throughput; publisher-specific peak RSS
      remains a separate profiling refinement.
- [x] Measure process-first and same-process warm complete-offset reads without
      incorrectly labeling uncontrolled OS/filesystem cache state as cold.
- [x] Measure frame windows through `frame_row_offsets` plus the corresponding
      instance slices.
- [x] Measure individual-frame indexed reads.
- [x] Measure contiguous observation-row windows used by downstream joins.
- [x] Measure consolidated and direct metadata open separately in the smoke
      harness.
- [x] Measure final object count and on-disk bytes in the smoke harness.
- [ ] Run representative Crimson/Mac/VPN reads or record that validation as an
      explicit cross-repository gate.

Acceptance requirements:

- [x] No decoded-value or dtype mismatch in the initial 200k-frame A/B smoke.
- [x] No unsafe overlapping parallel writes; the first matrix uses one writer
      owning complete chunks or shards.
- [ ] No unacceptable per-frame read amplification.
- [x] Object count is materially lower than the regular small-chunk layout at
      the 200k scale.
- [x] Peak writer memory is bounded and unchanged across the full-duration
      candidates; publisher-specific peak RSS remains to be captured.
- [x] The candidate result is reproducible and recorded in the common benchmark
      envelope.

Exit gate:

- [ ] One storage profile is selected for canonical detections with evidence;
      alternatives and rejected candidates remain recorded.

Initial smoke evidence and its limitations are recorded in
[`diagnostics/canonical_detection_storage_benchmark_smoke_2026-07-24.md`](diagnostics/canonical_detection_storage_benchmark_smoke_2026-07-24.md).

The first commit-pinned cluster lifecycle smoke is recorded in
[`diagnostics/canonical_detection_storage_cluster_smoke_2026-07-24.md`](diagnostics/canonical_detection_storage_cluster_smoke_2026-07-24.md).

Cluster matrix implementation and the required stage-to-scratch,
local-compute, publish-back lifecycle are tracked in
[`canonical_detection_storage_cluster_benchmark_checklist.md`](canonical_detection_storage_cluster_benchmark_checklist.md).

The corrected five-repetition access-aware result and exact Crimson handoff are
recorded in
[`diagnostics/canonical_detection_storage_access_aware_result_2026-07-24.md`](diagnostics/canonical_detection_storage_access_aware_result_2026-07-24.md).

## Remaining Completion Order — Frozen 2026-07-25

The remaining work follows the order below even though the older phase numbers
place production integration before consumer adoption. The paired consumer gate
must pass before Phase 5 changes `detect_yolo`.

### Checkpoint A — Paired Full-Analysis Fixtures

Fixture scope is product-complete for the frozen Crimson workload, not a copy
of every historical run. The versioned allowlist is
[`canonical_detection_full_analysis_sleepyfish_cam2010095_v1.json`](../configs/benchmarks/canonical_detection_full_analysis_sleepyfish_cam2010095_v1.json).
It currently selects 12 maintained product trees and 602 direct metadata files;
the source archive's consolidated hierarchy has 6,534 entries. Every selected
path and omission is therefore explicit and reviewable. Add a missing dependency
to the allowlist only when a consumer requires it; do not silently widen the
fixture to all historical runs.

At the pinned Crimson commit, subject-mask initialization reads the selected
run's `frame_indices`, `detection_indices`, `source_crop_row_ids`, optional
`available_channels`, its source crop's frame/detection/ROI coordinate columns,
and component contour stores. Pixel loading prefers `masks_roi`, then checks
bitpacked and RLE compatibility surfaces. The fixture spec therefore validates
that this selected run has dense `uint8 masks_roi (1169010,4,512,512)` with
chunks `(256,1,512,512)` and top-level `bytes` plus Zstandard codecs, and that
`mask_bitpacked` and `mask_rle` are absent. The builder copies that run unchanged;
mask optimization is intentionally outside this detection-layout comparison.

Plan mode is the default and performs no payload writes:

```bash
scripts/py -m fisheye.diagnostics.build_canonical_detection_full_analysis_fixtures \
  --spec configs/benchmarks/canonical_detection_full_analysis_sleepyfish_cam2010095_v1.json \
  --benchmark-root /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks \
  --destination /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/full_analysis/sleepyfish_cam2010095_v1
```

Run `--apply` only from a clean, commit-pinned cluster deployment after reviewing
the complete plan. Apply mode requires
`--expected-palette-commit <full-40-character-commit>` and an existing
node-local `--scratch-root`; it refuses a dirty or mismatched checkout. Assembly,
consolidation, and validation happen on scratch. The completed pair is copied
back to a fresh shared-storage incomplete sibling, content-verified, opened
through direct and consolidated metadata, frozen, and atomically renamed. The
implementation probes reflink isolation only between the new incomplete
benchmark base and its sibling; it never reflinks or hardlinks a production
source. `--pair-copy-mode copy` forces an ordinary independent scratch copy.

Frozen Crimson dependency:

- [x] Pin the fixture contract to Crimson commit
      `dadd9d779f0737c9643f15e3831a7c514bf99665` on branch
      `agent/phase5o4-full-analysis-fixture-contract`.
- [x] Verify the contract document SHA-256 is
      `aa64a94de7096b6a22e53d76357a619ca92bc5296b38f0549202fd67aee36a86`.
- [x] Freeze the explicit run name as
      `crimson_storage_fixture_sleepyfish_cam2010095_v1` under
      `detect_runs`.
- [x] Keep Crimson's production cache at 64 MiB for the first layout-only
      comparison.

Fixture input and safety plan:

- [ ] Record the maintained Sleepyfish source-archive metadata fingerprint
      before any copy.
- [x] Enumerate the exact selected nondetection runs required for refined
      keypoints, refined subject masks, subject shape, eye geometry, motion,
      eye-angle and tail-kinematics timelines, and crop geometry.
- [x] Confirm the exact refined-subject-mask arrays opened by Crimson before
      copying the selected dense run.
- [x] Record and validate that the current selected subject-mask run stores unsharded dense
      `uint8 masks_roi` with chunks `(256,1,512,512)`, lacks a compact mask
      cache, and is not silently optimized as part of the detection comparison.
- [ ] Probe server-side reflink/clone and hardlink behavior using disposable
      benchmark files only.
- [x] Never reflink, hardlink, chmod, or otherwise share mutable inode state
      directly with a production archive.
- [x] If physical sharing is used, create one independent immutable benchmark
      base first and share only between the two benchmark fixtures; disclose
      that relationship in both manifests.
- [x] Fall back to verified ordinary copies if safe clone/link semantics are
      unavailable or would change the benchmark interpretation.

Builder implementation:

- [x] Add separate plan and apply modes; plan mode performs no payload writes.
- [x] Accept only a fresh destination below
      `.palette_benchmarks/canonical_detection_storage/full_analysis`.
- [x] Reject production recording, registry, selector, training, and existing
      destination paths.
- [x] Stage each archive under a unique visibly incomplete sibling and install
      it only after validation.
- [x] Assemble and validate on node-local scratch, then copy and verify the
      completed pair back to shared benchmark storage before atomic install.
- [x] Preserve the source-video association without copying or transcoding the
      video.
- [x] Copy only the maintained required nondetection selections, their parent
      group envelopes, and required archive metadata—not every historical run.
- [x] Install the regular and hybrid candidate trees under the same exact
      canonical detection run name.
- [x] Mark both roots `benchmark_only=true`, `canonical=false`,
      `registry_registered=false`, and `selector_eligible=false`.
- [x] Permit benchmark-local selectors only; the Crimson invocation must still
      supply `--detection-run` explicitly.
- [x] Generate and validate inline Zarr v3 consolidated metadata after the
      complete direct-metadata tree is installed.
- [x] Require consolidated and direct declarations to agree for the run and all
      nine detection arrays.
- [x] Freeze successful fixture trees read-only and preserve failed attempts as
      explicitly incomplete evidence or remove them only through an explicit
      cleanup command.

Pair validation and manifest:

- [x] Validate the nine exact canonical array dtypes and shapes.
- [x] Validate offsets start at zero, are nondecreasing, have shape `(F+1,)`,
      and terminate at `N`.
- [x] Validate decoded detection values and array fingerprints are identical
      between regular and hybrid fixtures.
- [x] Validate all included nondetection direct metadata and payload bytes are
      identical between fixtures.
- [x] Normalize consolidated inventories and prove only detection physical
      layout declarations and necessarily regenerated consolidated bytes differ.
- [x] Record source archive, video, Palette commit, Crimson contract commit and
      digest, candidate fingerprints, copy/clone method, inventories, and
      validation results in each fixture manifest.
- [x] Recheck the maintained source fingerprint immediately before atomic
      publication.
- [x] Confirm zero registry, production-selector, and training-artifact updates.

Publication exit gate:

- [ ] Publish immutable `regular.zarr` and `hybrid.zarr` with complete manifests
      and provide their mounted paths to Crimson.

### Checkpoint B — Crimson Full-Archive Gate

Stage 1, storage layout only:

- [ ] Run regular and hybrid with the unchanged 64 MiB cache in five fresh,
      balanced processes.
- [ ] Require exact explicit-run selection, zero dtype/fallback probes, one
      retained offsets read, and identical required-product/frame identity.
- [ ] Measure all required simultaneous products, Ready time, first overlay,
      offset initialization, deterministic seeks, rapid-seek cancellation,
      3,500-frame forward/reverse traversal, shutdown, physical reads, and RSS.
- [ ] Apply the frozen correctness, latency, transfer, cancellation, deadline,
      and 2 GiB RSS gates without changing thresholds after observation.
- [ ] Run one native-30-FPS GUI correctness smoke for each accepted fixture.

Stage 2, cache policy only:

- [ ] Run only if the hybrid passes every Stage 1 gate.
- [ ] Compare hybrid at 16 MiB and 64 MiB in five fresh processes with layout
      and read-ahead held fixed.
- [ ] Select 16 MiB only if it passes the frozen absolute and relative gates;
      otherwise retain 64 MiB.
- [ ] Preserve complete raw and reduced evidence; neither result automatically
      promotes Palette's writer profile.

Consumer exit gate:

- [ ] Crimson accepts the hybrid physical layout and one bounded cache policy
      through the frozen full-archive workload.

### Checkpoint C — Versioned Physical-Profile Promotion

- [ ] Review the complete Palette cluster and Crimson full-archive evidence.
- [ ] Add one exact versioned canonical-detection physical profile; do not
      mutate the generic benchmark candidate in place.
- [ ] Bind Zarr v3, immutable indexed sharding, access-aware chunk budgets,
      outer-shard budget, codec chain, shard-index chain, and consolidated
      metadata requirements.
- [ ] Use approximately 128 KiB inner chunks for `WINDOWED` instance columns,
      1 MiB inner chunks for the `EAGER` offsets array, and 8 MiB outer shards
      only if the consumer gate approves that exact layout.
- [ ] Preserve the regular control and rejected candidates as benchmark
      evidence rather than supported aliases of the promoted profile.
- [ ] Add manifest round-trip and resolved-plan tests for the promoted identity.

Profile exit gate:

- [ ] One reviewed profile ID maps deterministically from every canonical
      detection array contract to its exact chunks, shards, and codecs.

### Checkpoint D — Production Writer Integration

- [ ] Route `detect_yolo` canonical array creation through the shared
      schema/storage owner.
- [ ] Write all nine canonical arrays at exact v1 dtypes, including authoritative
      `int64 frame_row_offsets`.
- [ ] Support empty runs, empty frames, one detection, and multiple detections
      per frame without sentinel observation rows.
- [ ] Exclude `frame_counts` and `n_detections` from canonical publication;
      derive them only in explicit compatibility adapters.
- [ ] Compute on node-local scratch and make every parallel writer own complete,
      nonoverlapping physical chunks or shards.
- [ ] Validate logical schema, row ordering, unique `instance_key`, geometry,
      offsets, candidate plan, codecs, and decoded readback before publication.
- [ ] Record logical-schema ID, physical-profile ID, resolved array plans,
      worker ownership, source identity, and validation receipts in provenance.
- [ ] Publish a fresh immutable run through the atomic publisher.
- [ ] Validate direct metadata before the final consolidated-metadata generation.
- [ ] Update production selectors only after payload, metadata, manifest, and
      Crimson compatibility validation succeed.
- [ ] Leave `detection_artifact_runs` and historical archives unchanged.

Production validation:

- [ ] Deterministic in-memory tests for schema and failure paths.
- [ ] Outside-sandbox real-Zarr tests for regular and nonempty publications.
- [ ] Empty-run and empty-frame tests.
- [ ] Multiple-detection-per-frame and row-order tests.
- [ ] Exact dtype rejection and offsets corruption tests.
- [ ] Partial-write, failed-consolidation, and pre-selector rollback tests.
- [ ] Parallel physical-ownership validation.
- [ ] Consolidated-only Crimson readback with no dtype probing.
- [ ] Selector-ineligible production canary followed by explicit reviewed
      promotion.

Canonical completion gate:

- [ ] A newly promoted `detect_yolo` run conforms to the logical and physical
      contracts, survives failure testing, and is read through Crimson's exact
      canonical adapter.

### Checkpoint E — Compatibility, Migration, And Expansion

- [ ] Keep historical `float64`, `frame_counts`, and `n_detections` behavior in
      explicit legacy adapters, not in canonical v1.
- [ ] Inventory old archives only after new production publication is stable.
- [ ] Decide separately whether important old runs need migration or may remain
      adapter-readable.
- [ ] Begin `detect_quality_runs`, refinement, and training storage contracts
      only after the canonical raw-detection completion gate passes.
- [ ] Treat refined-subject-mask storage as its own next-family project: retain
      dense editable authority and design explicit immutable compact display and
      training publications rather than silently changing it in this work.

## Phase 5 — `detect_yolo` Production Integration

- [ ] Route canonical array creation through the shared schema/storage owner.
- [ ] Remove raw canonical `chunks=` and `shards=` literals from the migrated
      writer path.
- [ ] Cast continuous canonical geometry to exact `float32` before publication.
- [ ] Preserve the approved row ordering and frame index.
- [ ] Validate every logical contract before run completion.
- [ ] Validate cross-array stage invariants before selector eligibility.
- [ ] Record resolved schema and storage-profile identities in provenance.
- [ ] Keep Dask/parallel writes aligned to whole physical chunks or shards.
- [ ] Update selectors only after payload and direct metadata validation.
- [ ] Consolidate and validate the published metadata generation as the final
      immutable visibility step.
- [ ] Leave `detection_artifact_runs` behavior unchanged.

Validation:

- [ ] Deterministic in-memory unit coverage.
- [ ] Focused real-Zarr integration test outside the sandbox.
- [ ] Empty-detection run.
- [ ] Representative nonempty run.
- [ ] Failure/rollback before selector publication.
- [ ] Consolidated-manifest validation.
- [ ] Read-back through the shared logical contract.

Exit gate:

- [ ] Newly published canonical detect runs conform to the versioned stage
      schema and selected physical storage profile.

## Phase 6 — Crimson And Consumer Adoption

- [ ] Publish concrete schema/capability bindings at the archive root.
- [ ] Ensure consolidated metadata exposes the selected paths, shapes, dtypes,
      chunk grids, shards, and codecs.
- [ ] Add one Crimson adapter for the new canonical schema version.
- [ ] Remove repeated dtype probing for that version.
- [ ] Retain an explicit `float64` legacy adapter for historical archives that
      require it.
- [ ] Validate whole-array, windowed, and per-frame reads against the workload
      used in the benchmark.
- [ ] Validate useful error reporting for unsupported schema versions.

Exit gate:

- [ ] Palette and Crimson agree on one exact current canonical contract without
      heuristic dtype or path discovery.

## Phase 7 — Existing-Archive Migration Decision

- [ ] Inventory canonical detect runs by schema, dtype, chunking, sharding, and
      consumer importance.
- [ ] Decide which archives remain readable through compatibility adapters.
- [ ] Decide whether any active archives justify physical migration.
- [ ] If migration is justified, build a copy-on-write migration tool with source
      fingerprints, destination validation, receipts, and rollback-safe
      publication.
- [ ] Never rewrite canonical source stores in place for benchmarking.
- [ ] Validate migrated results through Palette and Crimson.
- [ ] Record migration coverage and remaining compatibility population.

Exit gate:

- [ ] Every relevant old archive has either a supported adapter or a validated
      migration disposition.

## Phase 8 — Expand The Pattern

After canonical raw detection is complete, repeat the same sequence—consumer
census, logical schema, storage plan, benchmark, writer integration, consumer
adoption—for:

- [ ] immutable `detect_quality_runs` snapshots;
- [ ] refined-detection immutable bases and sparse review deltas;
- [ ] refined-detection publication snapshots;
- [ ] detection training promotion/export surfaces;
- [ ] shared detection lineage propagated into crops, keypoints, masks, and
      tracking.

Do not copy canonical-detection array assumptions blindly. Each surface must
declare its own authority, edit model, access pattern, and dimensions while
reusing shared logical contracts where the semantics are genuinely identical.

## Working Agreement

- [x] Keep each phase or independently reviewable subphase in its own commit.
- [ ] Regenerate deterministic inventories in the same commit as generator
      changes.
- [x] Do not mix dtype experiments into chunk/shard profile benchmarks.
- [x] Do not change a production writer before its logical and benchmark exit
      gates pass.
- [x] Run Zarr pytest/integration validation outside the sandbox according to
      `AGENTS.md`.
- [x] Update this checklist at every checkpoint so completed and deferred work
      remains visible.

## Immediate Next Action

- [x] Complete and document the first bounded 200k-frame cluster lifecycle
      smoke without promoting a profile.
- [x] Add the missing random/sequential/indexed read workloads and complete
      four balanced 200k-frame repetitions.
- [x] Run the frozen cross-workflow reduction preview, collect repetition 5
      without changing the workload, and apply the predeclared gates.
- [x] Carry the selected 128 KiB-inner / 8 MiB-target-shard plan and the regular
      1 MiB control to full-duration validation without promoting either.
- [x] Implement and benchmark the access-aware hybrid identified by the
      full-duration result before beginning HTTP/Crimson promotion testing.
- [x] Run the standalone exact-dtype, persisted-offset, cache, file-range, and
      UI workloads through Crimson on the actual Mac/SMB mount.
- [x] Complete the controlled Crimson cache/read-ahead checkpoint; retain the
      production 64 MiB policy until full-archive validation.
- [ ] Implement the fail-closed paired fixture builder beginning at Checkpoint A;
      do not promote a profile or modify `detect_yolo` before Checkpoint B passes.
