# Derived Analytics Storage Contract Audit And Implementation Checklist

Date: 2026-08-03

Status: active implementation checklist; read-only census complete; no production
profiles, selectors, or registries changed by this checkpoint

## Purpose

This document records the current contract coverage for Palette analytics
outputs and defines the implementation sequence for closing the remaining
logical-schema, physical-storage, publication, selection, and consumer gaps.

It follows the detection, crop, keypoint, and subject-mask contract work. Those
upstream observation families are not re-audited here, except for the
`body_frame_runs` and `keypoint_quality_runs` companion analytics that already
demonstrate the newer shared schema and byte-planner design.

The audit used the `sun` checkout at commit `83ac49be` and made no repository or
production-data changes.

## Executive Decision

Palette does **not** yet have uniform production-grade storage contracts for
all analytics outputs.

Seven maintained array-bearing derived families have an executable catalog,
scientific schemas, source-lineage validation, and shared atomic publication:

- track kinematics;
- swim bouts;
- bout kinematics;
- eye angles;
- subject shape;
- tail kinematics; and
- stimulus response.

However, none of those seven derives every array's physical layout through the
shared byte-budgeted storage planner. Only track kinematics and eye angles have
serialized registry publication. Tail-posture views and bout classification are
maintained derived stages with meaningful individual contracts but sit outside
the executable storage catalog. Stimulus epochs, occupancy outputs, and the
chaser analysis suite use additional stage-local schemas and writers. Several
legacy or in-place writers remain unclassified.

The target is not one scientific schema for every datatype. The target is one
contract system in which every persisted output is classified and every
scientific array has:

1. an exact logical identity;
2. an exact persisted representation;
3. an access- and byte-aware physical plan;
4. a safe publication and selection lifecycle; and
5. a real consumer and benchmark workload.

## Definition Of A Complete Analytics Storage Contract

A maintained analytics family is complete only when all of the following are
true.

### Logical schema

- Every array path is bound to a versioned logical contract.
- Dtype, rank, named axes, shape constraints, units, coordinate domain, and
  fill/null/sentinel semantics are exact.
- Required, optional, derived-cache, compatibility, and forbidden arrays are
  explicit.
- Row identity and source lineage are exact and fail closed.
- Unexpected arrays or attributes cannot silently become scientific authority.

### Physical storage

- Each array declares an access class and write mode.
- Inner chunks and outer shards are derived from uncompressed bytes and access
  shape rather than a shared row-count literal.
- The Zarr version, codec chain, compression level, checksum, shard-index
  configuration, and object estimates are manifest-bound.
- Parallel writers own complete, non-overlapping physical chunks or shards.

### Publication and selection

- Compute occurs in node-local scratch when appropriate.
- Publication is immutable, validated, consolidated, and atomic.
- Completion, selector eligibility, authority selection, and registry
  projection are separate fail-closed steps.
- A completed but selector-ineligible candidate cannot be reused implicitly.
- Copy verification is strong enough for the scientific risk of the family.

### Consumer and performance evidence

- Palette and Crimson readers select one exact schema version without dtype or
  path probing for current data.
- Legacy fallbacks require an explicit compatibility mode.
- Each writer has a deterministic benchmark covering its real access patterns.
- Logical equality, publication cost, object count, transfer, latency, and peak
  RSS are recorded before a physical profile is promoted.

## Current Coverage

| Surface | Current contract position | Main remaining work |
| --- | --- | --- |
| Track kinematics | Strong logical, lineage, materializer, and registry contract | Replace stage-specific fixed row grids with byte planning and benchmark the real consumer |
| Subject shape | Strong typed semantic surface, content manifest, lineage, and atomic sharded publication | Byte-plan heterogeneous semantic arrays; add registry projection |
| Tail kinematics | Strong typed surface, coordinate manifest, and whole-shard worker ownership | Replace fixed 262,144-row policy; add registry projection |
| Eye angles | Strong semantic dense layout and registry publication | Close dtype validation for every array and migrate fixed grids to byte planning |
| Swim bouts | Strong event/frame-axis semantics and compact tables | Add a closed whole-run array manifest, authoritative reader selection, byte planning, and registry projection |
| Bout kinematics | Exact structured table builders and compact columnar storage | Add a closed manifest, byte planning, registry projection, and consumer benchmark |
| Stimulus response | Cataloged and atomically published | Freeze exact compact table fields and dtypes; current output is data-dependent |
| Body frame | Exact ten-array schema, byte planner, strict manifest, and consolidated publication | Move from selector-ineligible companion evidence to an explicitly governed production lifecycle when authorized |
| Keypoint quality | Exact diagnostic schema, byte planner, strict manifest, and consolidated publication | Add explicit production lifecycle/registry policy when authorized |
| Tail-posture view | Exact typed arrays and guarded candidate lifecycle | Add to the executable catalog/materializer boundary and replace mask-derived row chunk helpers |
| Bout classification | Frozen required fields and guarded activation | Freeze exact per-field storage declarations and add to catalog/DAG/materializer boundary |
| Track visualization | Shared PNG/spec byte artifact contract | Keep classified as an artifact rather than a numeric scientific run |
| Stimulus epochs | Explicit typed window columns | Add catalog ownership, byte planning, manifest validation, and atomic publication |
| Detection/session occupancy | Scientific schema IDs and direct writers | Freeze exact arrays and lifecycle; adopt planner and atomic publisher |
| Chaser-distance base | Hardened immutable base and guarded activation | Add to the central analytics catalog and byte planner |
| Chaser components | Protocol-neutral schemas increasingly exist | Add independently payload-bound component seals and atomic component publication |
| Cross-recording Parquet exports | Versioned table/manifest contracts | Make generations atomic and exclusive; freeze exact Arrow dtypes where required |
| Core workflow exports | Declared planning nodes for sampled kinematics, summaries, eye traces, and tail traces | Implement adapters and exact export contracts |
| Legacy/in-place outputs | `speed_runs`, swim-bout statistics, and stimulus/chaser mutation paths remain | Classify as legacy/maintenance or migrate; do not leave them implicitly current |

## Evidence And Known Disagreements

The executable catalog explicitly says physical policy remains stage-owned and
exposes `byte_planner_adopted` as the migration boundary
(`src/fisheye/analysis_workflows/storage_contract_catalog.py:1-10,20-35`). Its
seven entries are at
`src/fisheye/analysis_workflows/storage_contract_catalog.py:75-175`.

The registry catalog contains three additional derived stages—track-kinematics
visualization, tail-posture view, and bout classification—outside that storage
catalog (`src/fisheye/registry/stage_catalog.py:194-255`). Availability patches
tail-posture and classification in separately
(`src/fisheye/analysis_workflows/availability.py:16-23`).

The shared columnar helper is an earlier-generation physical policy: it uses
4,096-row one-dimensional chunks, 1,024-row multidimensional chunks, and
262,144-row requested shards
(`src/fisheye/shared/zarr/columnar.py:31-94`). It passes chunks and shards to
array creation but does not freeze an explicit codec chain there
(`src/fisheye/shared/zarr/columnar.py:141-155`).

Compact stimulus-response fields and dtypes are inferred from the current row
payload (`src/fisheye/analysis/stimulus_response.py:1730-1817`). The older
`analysis_stage_arrays` declarations cover inputs and hierarchical output
variants, not the exact compact persisted surface
(`src/fisheye/shared/zarr/analysis_stage_arrays.py:1-7`).

The canonical analytics matrix currently documents the seven maintained
families (`docs/analytics_storage_schema_matrix.md:31-49`). That description is
useful but incomplete as a full writer census. The earlier reconciliation
correctly leaves the per-array inventory, byte-planner migration, and consumer
benchmarks open
(`docs/derived_analytics_storage_reconciliation_2026-08-01.md:102-124`).

## Correctness Blockers Before Physical Optimization

### Authoritative run discovery

The generic workflow availability path follows completion pointers but does not
require `stage_selector_eligible`. The atomic publisher records completion and
pointers before final selector activation. An interrupted activation can
therefore leave a completed, ineligible candidate discoverable for reuse.

Evidence:

- `src/fisheye/analysis_workflows/materializers/atomic_run_publisher.py:873-907`
- `src/fisheye/analysis_workflows/availability.py:332-401`
- `src/fisheye/utils/execute_analysis_workflow.py:221`
- canonical fail-closed resolver precedent:
  `src/fisheye/shared/zarr_run_completion.py:394`

### Maintained-reader fallback behavior

Swim-bout discovery can use lexical fallback without proving completion or
eligibility, and some eye-angle consumers read `latest` directly.

Evidence:

- `src/fisheye/analysis/swim_bout_io.py:158,954`
- `src/fisheye/analysis/eye_angle_io.py:470,587`
- `src/fisheye/visualization/visualize_eye_angle_overlays.py:707`

### Cross-recording export generations

Parquet parts are written directly into the final generation directory. An
overwrite does not first remove or isolate older parts. Validation checks the
manifest-listed set, while at least one consumer globs every Parquet part. A
rerun with fewer parts can therefore validate one set and read another.

Evidence:

- `src/fisheye/utils/export_cross_recording_analytics.py:5382-5435,5606-5613`
- `src/fisheye/analytics_exports/validation.py:87-178`
- `src/fisheye/utils/plot_cross_recording_bout_kinematics.py:39-51`

### Chaser component publication

The base chaser-distance run is hardened, but downstream components are
intentionally rejected until they gain independent payload-bound seals. Code
after the guard still contains delete-and-rewrite plus direct pointer updates,
which must not become authoritative publication behavior.

Evidence:

- `src/fisheye/analysis/chaser_distance_io.py:675`
- `src/fisheye/analysis/chaser_quadrant_occupancy.py:1123-1140,1403`
- `src/fisheye/utils/export_cross_recording_analytics.py:122-149`

## Implementation Checklist

### Phase 0 — Preserve the audit baseline

- [x] Inventory the seven cataloged maintained derived families.
- [x] Identify derived stage-catalog entries outside the storage catalog.
- [x] Inventory companion analytics, chaser components, exports, and legacy
      writers.
- [x] Separate logical, physical, lifecycle, consumer, and benchmark concerns.
- [ ] Add a machine-readable analytics coverage report generated from the live
      writer, stage, and storage catalogs.
- [ ] Make CI fail when a newly maintained array-bearing analytics stage has no
      explicit coverage classification.

### Phase 1 — Fix fail-closed lifecycle behavior

- [ ] Change generic workflow availability to use the canonical completion and
      eligibility resolver.
- [ ] Reject selector-ineligible candidates during implicit reuse.
- [ ] Permit statusless or lexical legacy discovery only behind an explicit
      inspection/compatibility policy.
- [ ] Add crash-window tests for completion written before activation.
- [ ] Migrate swim-bout run discovery to exact completion and eligibility
      checks.
- [ ] Migrate eye-angle readers and overlay tooling away from unqualified
      `latest` lookup.
- [ ] Confirm explicit run selection fails terminally instead of falling back.

### Phase 2 — Make export generations atomic

- [ ] Write each Parquet export generation into a hidden temporary sibling.
- [ ] Validate the complete part inventory, schemas, row counts, and content
      digests before publication.
- [ ] Atomically rename the validated generation into its immutable final path.
- [ ] Make the manifest enumerate the only allowed part files.
- [ ] Reject extra, missing, duplicate, or digest-mismatched parts.
- [ ] Change consumers to read the manifest-selected files rather than globbing
      a directory.
- [ ] Require validation before registry indexing or activation.
- [ ] Add a regression test in which a replacement generation has fewer parts.

### Phase 3 — Complete the executable analytics catalog

- [ ] Add `tail_posture_view` with its exact parent, schema, method, materializer,
      physical owner, registry mode, and planner status.
- [ ] Add `bout_classification` with the same executable declarations.
- [ ] Decide and record whether each of the following is a maintained
      scientific authority, an embedded component, a visualization/cache, an
      export, maintenance output, or legacy:
    - [ ] stimulus epochs;
    - [ ] detection/session occupancy;
    - [ ] chaser-distance base;
    - [ ] each chaser component in the versioned analysis profile;
    - [ ] registered-detection gate/QC outputs;
    - [ ] speed runs;
    - [ ] swim-bout statistics;
    - [ ] in-place chaser-state interpolation.
- [ ] Add canonical run-parent mappings for maintained families.
- [ ] Remove availability-only local parent declarations after catalog adoption.
- [ ] Add tests proving stage catalog, storage catalog, run-parent catalog, and
      materializer ownership agree.

### Phase 4 — Freeze exact logical schemas

- [ ] Introduce or reuse one exact analytics stage-schema representation that
      binds concrete paths to versioned `ArrayContract` identities.
- [ ] Require exact dtype, axes, shape, units, coordinates, null/fill semantics,
      access class, mutability, and authority role for every array.
- [ ] Freeze a new compact stimulus-response schema with exact tables, columns,
      dtypes, required/optional fields, and string encodings.
- [ ] Reject data-dependent schema expansion in current stimulus-response
      publication.
- [ ] Complete exact dtype validation for every eye-angle semantic array.
- [ ] Add closed whole-run manifests for swim bouts and bout kinematics.
- [ ] Freeze exact per-field bout-classification storage declarations.
- [ ] Bind tail-posture arrays to explicit analytics contracts rather than a
      subject-mask chunk helper.
- [ ] Define exact Arrow dtypes for export columns where cross-language stability
      requires them.
- [ ] Add recomputed-digest tampering and unexpected-field tests for every new
      manifest.

### Phase 5 — Seal chaser components independently

- [ ] Define an immutable component manifest envelope with source-run identity,
      schema, parameters, array declarations, content digests, and completion.
- [ ] Give each component its own hidden-copy/validate/rename publication step.
- [ ] Remove delete-and-rewrite publication of visible components.
- [ ] Bind component selectors to exact manifest digests.
- [ ] Record every requested component's output identity and validation result
      in the chaser runner receipt.
- [ ] Keep export of unsealed component tables fail closed.
- [ ] Add recovery tests for interrupted component publication and stale
      pointers.

### Phase 6 — Adopt byte-aware physical planning

- [ ] Define an `ArrayIntent` for every maintained analytics array.
- [ ] Classify access as eager, windowed, per-row, indexed, bulk-scan, or
      artifact-byte-stream.
- [ ] Classify write mode as immutable, append/whole-shard-owned, or editable.
- [ ] Derive chunks from dtype and per-row/per-unit shape, not a universal row
      count.
- [ ] Derive shards from target bytes and immutable whole-shard ownership.
- [ ] Pin Zarr v3 and the exact bytes/compression/checksum/index codec chain in
      the storage profile and manifest.
- [ ] Preserve independently readable inner chunks inside shards.
- [ ] Record requested/effective chunks and shards plus object estimates.
- [ ] Add planner parity tests for short recordings, million-frame recordings,
      empty tables, wide matrices, and fixed-width text arrays.

Recommended migration order:

1. eye angles;
2. tail kinematics;
3. subject shape;
4. tail-posture view and bout classification;
5. track kinematics;
6. shared columnar storage, thereby covering swim bouts and bout kinematics;
7. stimulus response after its exact schema is frozen;
8. stimulus/occupancy/chaser families after their authority classification and
   lifecycle are complete.

### Phase 7 — Complete publication and registry projection

- [ ] Add serialized registry completion/invalidation projection for swim bouts,
      bout kinematics, subject shape, tail kinematics, and stimulus response.
- [ ] Decide production selection policy for body frame and keypoint quality;
      do not infer activation from their selector-ineligible canaries.
- [ ] Standardize copy-integrity policy across maintained families.
- [ ] Require direct and consolidated metadata equivalence before visibility.
- [ ] Ensure completion, eligibility, selector activation, and registry state
      cannot disagree after recovery.
- [ ] Add idempotent finalizer and retry tests.

### Phase 8 — Implement declared workflow outputs

- [ ] Add the missing stimulus-response execution adapter.
- [ ] Add execution adapters for tail-posture view and bout classification.
- [ ] Implement exact contracts and publishers for:
    - [ ] sampled kinematic exports;
    - [ ] activity/spatial summaries;
    - [ ] eye traces;
    - [ ] tail traces.
- [ ] Keep recording-local Zarr authorities separate from immutable
      cross-recording Parquet query products.
- [ ] Bind each export to exact selected recording-local manifests.

### Phase 9 — Benchmark and promote one family at a time

- [ ] Define a deterministic writer/publisher/reader workload for every
      maintained family.
- [ ] Measure node-local compute, validation, consolidation, copy, and atomic
      publication separately.
- [ ] Measure apparent/allocated bytes, object count, compressed transfer,
      latency distributions, throughput, CPU, and peak RSS.
- [ ] Exercise real access patterns: eager small arrays, random frame/row,
      windowed playback, indexed ranges, full scans, and exports.
- [ ] Include representative short (~200,000-frame) and full-duration
      (~1,000,000-frame) recordings where the row axis is frame-like.
- [ ] Validate exact decoded equality between old and candidate layouts.
- [ ] Run Palette readers and Crimson adapters where the family is user-facing
      in Crimson.
- [ ] Promote a versioned physical profile only when correctness gates pass and
      the candidate materially improves its intended workload without a
      significant regression elsewhere.
- [ ] Retain the previous profile as an explicit rollback reader until the
      migration window closes.

### Phase 10 — Retire compatibility debt deliberately

- [ ] Mark each legacy writer and reader with an explicit compatibility status.
- [ ] Prevent legacy outputs from becoming implicit current authority.
- [ ] Provide migration/republication tools where historical scientific value
      justifies them.
- [ ] Remove legacy fallbacks only after inventorying affected archives and
      proving supported replacements.
- [ ] Update the canonical analytics matrix from executable catalogs rather than
      copying version numbers by hand.

## Promotion Gates

No analytics physical profile or selector should be promoted until all relevant
gates pass:

- exact logical schema and dtype validation;
- source-lineage and identity validation;
- direct/consolidated metadata equivalence;
- immutable atomic publication and recovery tests;
- selector/registry agreement;
- exact decoded equality against accepted source data;
- bounded object count and request amplification;
- real-consumer correctness and cancellation behavior;
- benchmark evidence for both representative short and long recordings; and
- an explicit versioned rollback path.

## Immediate Next Checkpoint

Begin with correctness rather than rechunking:

1. make workflow availability require completion **and** selector eligibility;
2. migrate swim-bout and eye-angle implicit readers to the canonical resolver;
3. make Parquet export generations atomic and manifest-exclusive; and
4. add the associated recovery and stale-part regression tests.

After that checkpoint, freeze the exact compact stimulus-response schema and
expand the executable analytics catalog before beginning physical-profile
migrations.
