# Canonical Detection Storage Implementation Checklist

Status: active; logical canonical detection accepted by Crimson, consumer
residency gates passed, physical-profile promotion deferred, and production
writer adoption still blocked

Date established: 2026-07-23

Last updated: 2026-07-26

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

## Remaining Completion Order — Updated 2026-07-26

The remaining work follows the order below even though the older phase numbers
place production integration before consumer adoption. The paired consumer gate
must pass before Phase 5 changes `detect_yolo`. A bounded integration fixture
now precedes the frozen full-duration gate; it does not replace or satisfy that
gate.

### Checkpoint A0 — 2,048-Frame Full-Product Integration Fixture

The first full-duration publication attempt stopped before copying any source
payload. It spent most of its time hashing the complete selected subject-mask
tree, then failed when strict JSON serialization encountered the source root's
legacy non-finite `imageio_metadata.nframes=+inf` attribute. No final fixture,
registry entry, production selector, or training artifact was created. Failed
scratch and shared incomplete evidence remains subject to explicit cleanup.

The corrective integration spec is
[`canonical_detection_integration_sleepyfish_cam2010095_2048_v1.json`](../configs/benchmarks/canonical_detection_integration_sleepyfish_cam2010095_2048_v1.json).
It selects camera frames `[0, 2048)` without rebasing and retains the same 12
maintained product trees as the full-duration spec. Array payloads are copied
as exact logical prefixes across four declared axis classes:

- 2,048 of 1,188,000 camera-frame rows;
- the observation-row prefix derived by summing the selected source
  `frame_counts` and validating it against `frame_indices`;
- 69 of 39,214 identity-indexed per-second rows; and
- four CSR contour point prefixes derived from each selected `ptr`/`len`
  endpoint.

Small constants and channel tables are copied in full. Any undeclared leading
axis larger than 2,048 fails closed. Source aggregate provenance is retained
and explicitly labeled as unrecomputed; it is not misrepresented as a new
scientific publication.

Implementation:

- [x] Classify the bounded pair as `integration_fixture` and record that it is
      invalid for full-duration startup, object-count, cache-pressure, long-
      traversal, and promotion gates.
- [x] Keep source camera-frame identity unchanged with a required zero-based
      prefix; introduce no rebasing contract.
- [x] Derive the observation-row stop from the source frame-count authority and
      require every selected frame-index/count surface to match it exactly.
- [x] Slice dense masks, crop geometry, keypoints, shape, eye, tail, motion,
      timeline, and CSR contour payloads through declared axis cardinalities.
- [x] Preserve the observed contour empty-row sentinel `ptr=-1, len=0`; derive
      point endpoints from contiguous positive spans rather than assuming every
      row has a nonnegative pointer.
- [x] Slice the two full-duration canonical-detection candidates to `F=2048`,
      require `frame_row_offsets.shape == (2049,)`, and preserve exact instance
      keys and decoded values.
- [x] Regenerate regular and hybrid detection arrays through their existing
      physical profiles so only canonical-detection storage metadata differs.
- [x] Replace complete selected-tree payload hashing with deterministic hashes
      of every copied metadata node and exact logical array block in the
      selected prefix, followed by direct regular/hybrid and post-copy logical
      equivalence checks.
- [x] Omit stale source consolidated metadata, normalize non-finite direct
      attributes to JSON `null` only in benchmark copies, and record the exact
      source metadata path, JSON pointer, symbolic original value, and
      replacement.
- [x] Require strict JSON, inline consolidated metadata, direct/consolidated
      agreement, immutable publication, and selector/registry ineligibility.
- [x] Cover multi-detection frames, `F+1` offsets, frame/row arrays, dense
      masks, a per-second timeline, CSR points, source immutability, and paired
      equivalence in a focused real-Zarr test.
- [x] Run the real-source plan from a clean commit-pinned deployment.
- [x] Publish the immutable 2,048-frame `regular.zarr` and `hybrid.zarr` pair
      through node-local scratch and provide both paths and manifests to
      Crimson.
- [x] Run Crimson's quick schema/open/readiness/overlay/cancellation gate using
      only frame IDs in `[0, 2048)`.

Palette publication completed on 2026-07-26 from commit
`4bf96646f873517bbcf921f78af151b41ce0ed78` in LSF job `153174098`. The
immutable pair is at
`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/full_analysis/sleepyfish_cam2010095_integration_2048_v1`.
Its `pair_manifest.json`, per-layout manifests, and `publication_receipt.json`
are the handoff authorities. The job completed in 3 minutes 7 seconds with a
670,112 KiB peak RSS. The pair contains 507 exact selected logical array
slices (2,178,127,952 logical bytes per layout) in approximately 104.6 MB of
shared files across both layouts. Each layout passed strict JSON,
direct/consolidated metadata agreement, source-postcopy logical hashing, and
row/CSR relationship validation. The only regular/hybrid physical metadata
differences are the canonical detection run group and its nine contract
arrays; all nondetection consolidated metadata and decoded detection values
match exactly.

An earlier bounded attempt, LSF job `153174091` at commit `b99c554a`, built and
validated both scratch stores but failed the paired gate because equivalent
relationship path lists inherited different filesystem iteration order. No
final destination was installed. Its failed-run record and incomplete evidence
remain preserved; commit `4bf96646` canonicalizes those evidence lists and adds
the regression assertion. This was an evidence-ordering defect, not a payload
or relationship defect.

Integration exit gate:

- [x] Palette and Crimson both accept the bounded pair for functional
      integration. No result from this checkpoint is cited as full-duration
      promotion evidence.

Crimson completed the bounded gate on 2026-07-26. Regular and hybrid headless
runs produced identical detection digests and zero stale publications; both
Metal GUI smokes reached frame 300 with zero skipped or late frames; and the
three focused consumer tests passed. This closes compatibility and application
behavior only. It does not establish full-duration storage performance.

### Checkpoint A — Paired Full-Analysis Fixtures

This is now the active scalability checkpoint. The bounded integration pair
passed, so Palette can publish one full-duration pair for Crimson's balanced
five-process comparison. Do not restore the original repeated complete
dense-mask tree hashes as the source-integrity mechanism.

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
back to a fresh shared-storage incomplete sibling, validation-sampled, opened
through direct and consolidated metadata, frozen, and atomically renamed. The
implementation probes reflink isolation only between the new incomplete
benchmark base and its sibling; it never reflinks or hardlinks a production
source. `--pair-copy-mode copy` forces an ordinary independent scratch copy.

The full-duration builder intentionally uses a different validation budget
from the bounded logical-slice fixture. Copy operations must complete without
error; all selected direct Zarr metadata is hashed exactly before and after;
every selected array is checked at deterministic origin, midpoint, and endpoint
coordinates; all nine detection arrays are fully decoded and hashed; and the
complete frame/count, sparse second-index, and contour CSR relationships are
validated on scratch and again after the shared copy. It does not recursively
hash every nondetection payload object. The original implementation performed
that high-fanout walk repeatedly and spent 25 minutes before copying any data.
The manifest records this distinction explicitly rather than claiming a full
nondetection content hash.

The declared full axes are 1,188,000 camera frames, 1,169,010 maintained
analysis rows, 39,214 sparse per-second rows, and four contour point stores.
`second_indices` is strictly increasing and unique, but not dense: it spans
0–39,599 with 386 gaps. The fixture preserves those values and does not relabel
them as a dense identity axis.

Use the LSF wrapper for the node-local preflight and publication job. It is
render-only by default, records the exact commit, scratch capacity, reflink
isolation, resource usage, and job status, and preserves scratch evidence after
a failed publication:

```bash
scripts/submit_canonical_detection_full_analysis_fixture_bsub.sh \
  --spec configs/benchmarks/canonical_detection_full_analysis_sleepyfish_cam2010095_v1.json \
  --benchmark-root /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks \
  --destination /groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/full_analysis/sleepyfish_cam2010095_v1 \
  --run-id sleepyfish_cam2010095_v1_<commit> \
  --palette-repo <commit-pinned-groups-worktree> \
  --preflight-only
```

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
- [x] Probe node-local reflink/clone behavior using disposable
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
- [x] Construct both nondetection trees from one independent scratch base and
      require exact direct metadata plus identical deterministic array samples
      before and after shared publication.
- [x] Normalize consolidated inventories and prove only detection physical
      layout declarations and necessarily regenerated consolidated bytes differ.
- [x] Record source archive, video, Palette commit, Crimson contract commit and
      digest, candidate fingerprints, copy/clone method, exact metadata
      inventory, sample ledger, and validation results in each fixture
      manifest.
- [x] Recheck the maintained source fingerprint immediately before atomic
      publication.
- [x] Confirm zero registry, production-selector, and training-artifact updates.

Publication exit gate:

- [x] Publish immutable `regular.zarr` and `hybrid.zarr` with complete manifests
      and provide their mounted paths to Crimson.

Palette LSF job `153174149` completed at commit
`fcdc67e764a8ddbe318cab7be19f2c3ab7f5fdb5` in 8,010 seconds and atomically
published the read-only pair at
`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/full_analysis/sleepyfish_cam2010095_v1`.
The publication receipt records exact direct/consolidated opens, decoded
detection equality, deterministic nondetection samples, complete frame/count
and CSR relationships, unchanged source evidence, and zero registry, selector,
training-artifact, or profile changes. The pair-manifest SHA-256 is
`25e49003d63f74e5c7f1aa940aa77acee8df0153476847afeb99b98238574432`; the
publication-receipt SHA-256 is
`22763ffce084446cfa797b567ebd7bb66d682e7e1f1f031e7af781c575fdcd1d`.

### Checkpoint B — Crimson Full-Archive Gate

Stage 1, storage layout only:

- [x] Run regular and hybrid with the unchanged 64 MiB cache in five fresh,
      balanced processes.
- [x] Require exact explicit-run selection, zero dtype/fallback probes, one
      retained offsets read, and identical required-product/frame identity.
- [x] Measure all required simultaneous products, Ready time, first overlay,
      offset initialization, deterministic seeks, rapid-seek cancellation,
      3,500-frame forward/reverse traversal, shutdown, physical reads, and RSS.
- [x] Apply the frozen correctness, latency, transfer, cancellation, deadline,
      and 2 GiB RSS gates without changing thresholds after observation.
- [ ] Run one native-30-FPS GUI correctness smoke for each accepted fixture.

Stage 1 accepted the logical canonical-detection contract and the persisted
offset access model. It did not promote the 128 KiB hybrid physical profile:
nondetection initialization and scheduling dominated the full-application
result, and a separate consumer-strategy question remained for the small
decoded detection hot set.

Residency strategy gate:

- [x] Freeze the strategy contract at Crimson parent commit
      `81433985a6be17ae490e674db2b0043360db6b02`.
- [x] Run the 20-process isolated paged/resident comparison against the existing
      regular and hybrid fixtures.
- [x] Run the ten-process hybrid full-archive interference comparison.
- [x] Require one retained offset read, exact paged/resident values, atomic
      resident visibility, zero stale publications, bounded cancellation,
      bounded RSS, zero post-warmup deadline misses, and no maintained-product
      initialization regression above the frozen limit.
- [x] Accept byte-budgeted UI-column residency for separate production-policy
      review while leaving production residency disabled.
- [x] Cancel the original 25-store physical matrix and defer a reduced matrix
      containing only the 128 KiB hybrid, one 8,192-row-aligned candidate, and
      the genuine 1 MiB unsharded control.

The immutable Crimson evidence handoff is commit
`b7a241e853ce08cb2c3d58a48ecd4f0b497afa61` on branch
`codex/phase5o5-residency-verdict-20260726`. Its handoff-manifest SHA-256 is
`5ede0755c86351d7db20b22d5da86d76d9e44a111facea20abfb0901786fa982`, the
full-archive aggregate SHA-256 is
`e42fce1de8346f321fec71512055ab2f1b9971372bb19267ae817c39bc7ed8ae`, and the
gate-document SHA-256 is
`83ce7870443a11346dfed10bea084ad58827ebac4acea4fd93e9b6dbc71405aa`.

This evidence is immutable but was executed from a dirty Crimson development
worktree at `34ff3c3`, not from a clean reproducible source revision. The handoff
records exact source and binary hashes for the executed surfaces. Palette must
preserve that limitation in every citation and must not describe the benchmark
as a clean-commit reproduction.

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

The residency gate does not close this physical-layout exit gate. It validates
a consumer policy over pageable storage; no Palette profile has been selected.

### Checkpoint C — Versioned Physical-Profile Promotion

Deferred until the refined-detection semantic-selection work is complete. Do
not infer profile promotion from the passing residency gate or rewrite the
failed historical Stage 1 decision.

Practical candidate disposition (2026-07-26): the access-aware 128 KiB/1 MiB
inner, 8 MiB outer hybrid is the leading production candidate. The Palette
five-repetition comparison reduced payload objects from 88 to 16, median
publication time from 1.192 to 0.514 seconds, complete PRFS reader time from
66.035 to 58.501 seconds, random-frame p95 from 25.24 to 19.81 milliseconds,
and raised sequential throughput from 41,933 to 47,792 FPS
(`docs/diagnostics/canonical_detection_storage_access_aware_result_2026-07-24.md:18-31,84-102`).
Crimson's later full-application and residency work found no correctness or
deadline reason to reject it, but the original frozen full-application gate did
not pass and no profile was promoted. Treat the hybrid as a candidate, not a
default.

The reduced three-candidate optimization matrix remains deferred and is no
longer a prerequisite to a practical promotion decision. First run a paired
regular-versus-hybrid check on the frozen immutable refined-detection schema.
Resume the 8,192-row candidate matrix only if that check exposes a material
problem or later optimization evidence is needed.

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
- [ ] Validate one logically identical regular/hybrid immutable refined snapshot
      for exact decoded equality, direct/consolidated metadata equivalence,
      codec support, and the required `F+1` frame-row offset index.
- [ ] Apply a new prospective practical gate: zero correctness differences,
      zero deadline misses, no meaningful readiness/current-frame regression,
      at least 4x fewer payload objects, at least 20% less traversal transfer,
      and no material RSS regression.
- [ ] Publish one selector-ineligible refined canary, validate it in Palette and
      Crimson, and retain the regular profile as rollback evidence before making
      the versioned hybrid profile a writer default.

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
- [ ] Begin implementation of `detect_quality_runs`, refinement, and training
      storage contracts only after the canonical raw-detection completion gate
      passes. A read-only producer/consumer/lifecycle census may proceed now so
      the later contracts do not inherit raw-detection assumptions blindly.
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

The passing Crimson residency gate exposed the first semantic priority for this
expansion: production selection should prefer an explicitly selected,
validated refined/corrected detection authority when available. The raw
`detect_runs` benchmark proves canonical storage and access only; it does not
define refined-run selection or manual-addition semantics.

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
- [x] Publish and validate the fail-closed full-duration paired fixture without
      promoting a profile or modifying `detect_yolo`.
- [x] Record Crimson's immutable residency handoff, passing isolated and
      full-archive gates, cancelled 25-store matrix, deferred three-candidate
      matrix, and dirty-execution limitation.
- [x] Complete the refined-detection producer/consumer/lifecycle census before
      designing its logical schema, edit-delta contract, selection policy, or
      physical profiles.
- [x] Freeze refined-detection snapshot v1 before delta/compactor work: exact
      full and clipped array sets, dtypes, identities, sentinels, dual `F+1`
      indexes, byte-based access rules, Zarr v3 codec chain, consolidated
      metadata gate, and an explicit unpromoted access-aware candidate.
- [x] Complete Crimson's first read-only refined snapshot/storage review. Record
      `ACCEPT WITH REQUIRED CHANGES` and freeze its six blockers: persisted
      manifest envelope, fail-closed selection, cross-snapshot identity,
      clipped binding, zero-frame policy, and separate reason registries.
- [x] Complete Crimson's second read-only review. Three original blockers are
      resolved; narrow executable-validation gaps remain for the metadata
      declaration digest, parsed clipped binding, and reason-code coverage.
- [x] Add an exact direct/consolidated metadata normalizer and derived digest,
      deep clipped-binding parser, canonical reason-registry parser, persisted
      array-code coverage, and one fail-closed publication validator. Also
      reject parent/child recording changes and snapshot-ID reuse.
- [x] Complete a third adversarial read-only review. It found that a recomputed
      payload digest could still hide nested logical/storage mutations, that
      the named publication gate omitted identity validation, and that clipped
      rows were not proven against the bound per-clip artifacts.
- [x] Reconstruct and exactly compare the frozen logical and physical plans,
      make identity validation part of publication, define ordered clipped
      source authorities, require bound per-clip row evidence, and add
      recomputed-digest plus multiple-subject tests.
- [x] Add the read-only current refined-run transition adapter; keep it
      selector-ineligible and report blocked/lossy mappings explicitly.
- [x] Obtain read-only re-review of the hardened gate. Crimson accepted the
      Palette publication checks for nested tampering, root/successor identity,
      clipped evidence, and multiple subjects; its remaining required changes
      are in the Crimson refined-v1 consumer and identity-preserving UI path.
- [x] Add the standalone selector-ineligible shadow writer and validate it with
      a mixed raw/manual two-instance frame through real Zarr v3 consolidated
      metadata. It can write only below `/tmp` or `.palette_benchmarks`, never
      inside a recording archive, and never updates selectors or registries.
- [x] Census representative current runs through the read-only transition. A
      23,287-frame full-acquisition run becomes exact v1 with explicit
      historical source-key initialization and no lossy conversion. The
      1,188,000-frame clipped aggregate is deliberately blocked so its ten
      lineage columns cannot be discarded by the full-acquisition adapter.
- [ ] Run the paired regular-versus-access-aware refined snapshot canary and
      apply the pragmatic correctness/object/transfer/readiness/RSS gate before
      promoting a versioned writer profile.
- [ ] Begin detection delta v2 and compactor design only after the snapshot
      contract and its production-transition findings are accepted.

The census is now recorded in
[`diagnostics/refined_detection_producer_consumer_lifecycle_census_2026-07-26.md`](diagnostics/refined_detection_producer_consumer_lifecycle_census_2026-07-26.md).
It finds that the sparse logical authority is sound, but current review remains
single-slot/whole-rewrite compatibility code and the existing detection delta
primitive is insufficient for a general manual addition. Its unchecked contract
decisions are the review and implementation queue; the census itself is complete.

The resulting frozen target is documented in
[`refined_detection_storage_contract_v1.md`](refined_detection_storage_contract_v1.md).
Executable declarations and deterministic validation live in
`refined_detection_schema.py`, `refined_detection_storage.py`, and
`refined_detection_manifest.py`. The manifest-only parser is not a publication
gate: contract-valid snapshots must pass
`validate_refined_detection_publication()` with their exact direct/consolidated
metadata map and decoded arrays. No current writer, selector, registry, delta
schema, compactor, or archive was changed by this checkpoint.
