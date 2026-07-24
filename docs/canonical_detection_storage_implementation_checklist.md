# Canonical Detection Storage Implementation Checklist

Status: active implementation checklist

Date established: 2026-07-23

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
- [ ] Sweep `128 KiB`, `512 KiB`, `1 MiB`, and `2 MiB` inner-chunk targets.
- [ ] Sweep appropriate `8 MiB`, `32 MiB`, `128 MiB`, and `512 MiB` shard targets.
- [x] Complete an initial same-input regular-versus-indexed-sharding smoke.
- [ ] Measure sequential write throughput and peak memory.
- [ ] Measure sharded publication/copy throughput and peak memory.
- [ ] Measure cold and warm full-array reads.
- [ ] Measure frame-window reads.
- [ ] Measure individual-frame indexed reads.
- [ ] Measure observation-row reads used by downstream joins.
- [x] Measure consolidated and direct metadata open separately in the smoke
      harness.
- [x] Measure final object count and on-disk bytes in the smoke harness.
- [ ] Run representative Crimson/Mac/VPN reads or record that validation as an
      explicit cross-repository gate.

Acceptance requirements:

- [x] No decoded-value or dtype mismatch in the initial 200k-frame A/B smoke.
- [ ] No unsafe overlapping parallel writes.
- [ ] No unacceptable per-frame read amplification.
- [ ] Object count is materially lower than the regular small-chunk layout.
- [ ] Peak memory is bounded for the intended writer and publisher hosts.
- [ ] The chosen result is reproducible and recorded in the common benchmark
      envelope.

Exit gate:

- [ ] One storage profile is selected for canonical detections with evidence;
      alternatives and rejected candidates remain recorded.

Initial smoke evidence and its limitations are recorded in
[`diagnostics/canonical_detection_storage_benchmark_smoke_2026-07-24.md`](diagnostics/canonical_detection_storage_benchmark_smoke_2026-07-24.md).

Cluster matrix implementation and the required stage-to-scratch,
local-compute, publish-back lifecycle are tracked in
[`canonical_detection_storage_cluster_benchmark_checklist.md`](canonical_detection_storage_cluster_benchmark_checklist.md).

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

- [ ] Keep each phase or independently reviewable subphase in its own commit.
- [ ] Regenerate deterministic inventories in the same commit as generator
      changes.
- [ ] Do not mix dtype experiments into chunk/shard profile benchmarks.
- [ ] Do not change a production writer before its logical and benchmark exit
      gates pass.
- [ ] Run Zarr pytest/integration validation outside the sandbox according to
      `AGENTS.md`.
- [ ] Update this checklist at every checkpoint so completed and deferred work
      remains visible.

## Immediate Next Action

- [ ] Commit the accepted dtype decision and this checklist.
- [ ] Begin Phase 1 with the canonical detection consumer/producer census.
