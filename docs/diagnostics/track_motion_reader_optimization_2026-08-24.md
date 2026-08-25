# Track-Motion Reader Optimization Audit and Implementation Direction

**Date:** 2026-08-24

**Method:** five parallel read-only Luna xhigh audits (reader hot path, Zarr read
amplification, receipt integrity, downstream projections, and test design),
combined with direct inspection of the four-recording production workload and
its runtime telemetry.

**Production evidence:** LSF array `153737380[1-4]`, workflow
`sleepyfish_2026_08_06_authority_downstream_recovery_20260824_v003`, pinned
Palette commit `ffc6b3b8fd28eb8de2b94bb701eda7028427f900`.

**Operation root:**
`/groups/johnson/johnsonlab/jeremy/operations/sleepyfish_2026_08_06_authority_downstream_recovery_20260824_v003`.

**Status:** design recommendation and measured baseline; no reader code was
changed during this audit.

Companion authority-policy audit:
[`crop_contract_split_audit_2026-08-24.md`](crop_contract_split_audit_2026-08-24.md).
This proposal follows that audit's rule: multiple supported publication
profiles may exist, but consumers meet them through one validated authority
interface.  The optimization proposed here is a projection over the existing
track-motion authority, not a new crop profile, publication grammar, adapter,
or bypass flag.

---

## 0. Verdict

The approximately 14–16 minute track reload seen before swim-bout detection is
real and is dominated by repeated whole-publication verification and
unnecessarily broad materialization.  A consumer asking for speed data still
causes the current reader to:

1. bind the complete coordinate and temporal authority;
2. rebuild the complete live motion manifest and hash all published surfaces;
3. copy and hash nearly every track surface, regardless of the requested speed
   levels;
4. call `assert_verified()`, which performs a second complete live binding and
   manifest validation.

The fail-closed intent is correct.  The problem is that the only normal read
shape is an exhaustive scientific audit.  The safe optimization is a **named,
receipt-backed projection reader** that returns a deliberately bounded subset
of the same sealed authority.  The existing exhaustive reader remains the
whole-publication audit path.

The proposed structure is:

```text
one sealed track-motion publication authority
                 |
                 v
one track-motion resolver / validation interface
                 |
        +--------+---------+------------------+
        |                  |                  |
   FULL audit       swim_bouts_v1      track_identity_v1
   projection       projection         projection
```

These projections are consumption shapes, not separate authority grammars.
Callers may not supply arbitrary array paths or request a generic
`skip_validation` mode.

---

## 1. Measured production baseline

### 1.1 Track reload before swim-bout detection

The four array tasks persisted a dedicated wall-time measurement for the
`load_track_kinematics` phase:

| Recording | Reload wall time (s) | Reload wall time |
|---|---:|---:|
| cam2010093 | 888.223545 | 14m 48s |
| cam2010094 | 815.263483 | 13m 35s |
| cam2010095 | 938.357699 | 15m 38s |
| cam2010096 | 919.272407 | 15m 19s |
| **Mean** | **890.279284** | **14m 50s** |
| **Median** | **903.747976** | **15m 04s** |

The four reloads consumed about **59m 21s of cumulative task time**.  Because
the recordings ran in parallel, their contribution to array wall time was the
slowest reload, approximately **15m 38s**.

For historical scale, the July Sleepyfish receipt/proof-reuse canaries recorded
normal exhaustive public-reader times of approximately 151–157 seconds for
1,169,010 rows and 104 motion surfaces.  The current recordings contain roughly
2.75–2.94 million rows and took 815–938 seconds to reload.  Host, cache, code,
and storage conditions differ, so this is not a controlled scaling result; it
does show that the current cost is not a one-off timer anomaly.  The historical
evidence is recorded in `docs/analysis_materializer_runtime_telemetry.md`.

Evidence is in these files below the operation root:

- `logs/core_behavior_array.153737380.1.out:24432`
- `logs/core_behavior_array.153737380.2.out:23074`
- `logs/core_behavior_array.153737380.3.out:23074`
- `logs/core_behavior_array.153737380.4.out:24432`

### 1.2 Track materializer CPU efficiency

The track materializer's `palette.materializer_phase_telemetry` records wall
time, own/child CPU seconds, average effective CPU cores, peak RSS, and process
I/O deltas.  Each production task allocated 16 CPU slots:

| Recording | Wall (s) | CPU (s) | Effective cores | 16-slot efficiency | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|
| cam2010093 | 946.817 | 1,027.837 | 1.086 | 6.78% | 2.89 |
| cam2010094 | 924.612 | 999.356 | 1.081 | 6.76% | 2.72 |
| cam2010095 | 969.130 | 1,047.160 | 1.081 | 6.75% | 2.75 |
| cam2010096 | 1,014.180 | 1,094.871 | 1.080 | 6.75% | 2.83 |

Across the four tasks, the materializer averaged approximately **1.08 effective
CPU cores**, or **6.76% of the 16 allocated slots**.  LSF's whole-job snapshots
were consistent with this result.  The scheduler also displayed an effective
240 GiB memory reservation for each 16-slot task, far above the observed
2.72–2.89 GiB materializer RSS.  Resource requests should be revisited after
the read path is fixed; some later stages may have different parallelism and
must not be sized solely from this phase.

### 1.3 Telemetry coverage gap

There are three telemetry layers, but the current campaign used only two:

| Layer | Current v003 coverage | What it measures |
|---|---|---|
| LSF accounting | yes | aggregate task CPU, memory, slots, and wall time |
| Materializer `PhaseTelemetry` | yes for modern materializers | per-phase CPU, wall, RSS, and process I/O |
| Process-tree sampler | **not used by this custom submission** | two-second CPU-core, RSS, thread, process, and I/O samples |

The standard `scripts/submit_analysis_workflow_bsub.sh` invokes
`fisheye.diagnostics.run_with_resource_telemetry` and writes
`resource_telemetry_summary.json` plus `resource_telemetry_samples.jsonl`.
This v003 campaign called `execute_analysis_workflow` directly, so those files
were not created.  Its materializer telemetry is machine-readable but is
embedded in LSF stdout instead of being linked from the campaign execution
report.

The swim-bout loader records only wall time through `phase_timing`; it does not
currently record CPU or I/O for that phase.  Consequently, we can measure the
reload's 14–16 minute latency precisely but cannot yet separate `/groups`
latency, decompression, hashing, and numerical validation by CPU/I/O evidence.

Relevant implementation:

- `src/fisheye/shared/runtime_telemetry.py:371-510`
- `src/fisheye/analysis_workflows/materializers/track_kinematics.py:1000-1153`
- `src/fisheye/analysis/detect_bouts_multi_level.py:3075-3092`
- `src/fisheye/diagnostics/run_with_resource_telemetry.py:35-299`
- `scripts/submit_analysis_workflow_bsub.sh:168-169,379-416`

---

## 2. Read-amplification diagnosis

### 2.1 Physical and decoded size

The four current track runs contain:

- 106 total arrays per run;
- 104 controlled public track-motion surfaces;
- approximately 1.08–1.16 GiB of decoded logical array data;
- approximately 172–188 MiB of encoded Zarr chunks.

`load_track_kinematics_track()` attempts 59 surfaces and copies 57, representing
approximately 0.70–0.74 GiB decoded and 129–138 MiB encoded per recording.
The reader also traverses both nested movement surfaces and flat compatibility
aliases; aliases are currently hashed as independent arrays.

### 2.2 Repeated full scans

The current call chain is:

```text
load_track_kinematics_track
  -> load_bound_track_motion_run
     -> rebuild complete live motion manifest
        -> hash every published surface
           -> array_payload_sha256 reads and then rereads the array
  -> copy 57 surfaces
     -> read each selected array and hash the owned copy
  -> bound_run.assert_verified
     -> repeat complete live binding and manifest validation
```

For the 104 public motion surfaces, the manifest portion alone is approximately:

```text
104 surfaces
  x 2 payload reads per digest (initial + TOCTOU recheck)
  x 2 complete manifest rebuilds
= 416 complete surface reads
```

This excludes the 57 returned-array copies and hashes, physical millimetre/pixel
consistency reads, numeric invariant recomputation, coordinate and temporal
authority validation, and alias handling.

Important source locations:

- `src/fisheye/analysis/track_kinematics_io.py:406-422,501-705`
- `src/fisheye/analysis/track_kinematics.py:668-743,9218-9740,11533-11717`
- `src/fisheye/shared/coordinate_frame_record.py:682-735`

### 2.3 `required_speed_levels` does not bound I/O

`required_speed_levels` currently acts as a post-read presence requirement.
The loader still iterates across every speed level and nearly every optional
surface before checking the requested levels.  A caller asking for one speed
level therefore does not receive a one-level I/O projection.

This behavior needs a counting-reader regression test before it changes.  The
test should fail today by showing that a request such as
`required_speed_levels=("smoothed",)` still reads raw, filtered, averaged,
heading, acceleration, and other unrelated surfaces.

### 2.4 Swim bouts need a bounded subset

The current multi-level swim-bout workflow needs acquisition-frame identity,
validity fields, positions for bout points, four persisted speed levels, and
the corresponding path-distance inputs.  Exponential speed is derived from
filtered speed.  It does not need the full heading, angular derivative,
acceleration, and summary inventory.

A preliminary consumer census places `swim_bouts_v1` at roughly 13–16
canonical arrays plus authority metadata, versus 57 copied arrays today.  The
estimated returned decoded payload is roughly 150 MiB instead of about
0.70–0.74 GiB.  The exact field list and alias resolution must be frozen in the
projection specification and tested before implementation; the estimate is
not itself a contract.

---

## 3. Consumer projection census

The agent audits found several distinct read shapes over the same publication:

| Proposed projection | Required logical content |
|---|---|
| `full_v1` | Existing exhaustive publication and all normal track fields; compatibility/audit path |
| `swim_bouts_v1` | acquisition frames; sample/transition validity; raw, filtered, smoothed, averaged physical speeds; required physical/pixel path distances; positions for bout points; FPS and authority metadata |
| `bout_kinematics_v1` | frames and time; physical/pixel positions; one configured speed and matching path distances; validity; requested heading arrays; authority metadata |
| `visualization_static_v1` | time, positions, validity, displayed speed levels, smoothed acceleration and heading, cumulative distance |
| `visualization_interactive_v1` | broader UI inventory including speed, acceleration, distance, angular derivatives, validity and reason codes |
| `track_identity_v1` | `track_sample_key`, `source_acquisition_frame_index`, `source_instance_key`, FPS and identity/authority metadata |
| `portable_export_v1` | the existing bounded kinematics export selection; currently 21 surfaces and already window-streamed |

The activity/spatial export actually needs a subset of the portable export:
frames, source/sample/transition validity, physical positions, filtered
physical speed, and filtered path distance.  It may merit a smaller named
projection after the first implementation proves the interface.

Relevant consumer sites:

- `src/fisheye/analysis/detect_bouts_multi_level.py:2578-2697,2873-2891,3095-3115`
- `src/fisheye/analysis/bout_kinematics.py:2952-3078,3322-3358`
- `src/fisheye/analysis/plot_track_kinematics.py:1473-1508,1562-1570,1629-1657`
- `src/fisheye/visualization/interactive_track_kinematics.py:1516-1612`
- `src/fisheye/analytics_exports/kinematics_samples.py:96-117,598-715`
- `src/fisheye/analytics_exports/activity_spatial_time_bins.py:1206-1225`
- `src/fisheye/analytics_exports/tail_trace_samples.py:326-390,943-962`

The export readers' bounded `_SELECTED_SURFACES` and `_source_binding()` design
is the closest existing model.  The new work should consolidate that pattern
behind the shared track-motion projection interface instead of creating a
second resolver family.

---

## 4. Integrity boundary: what may and may not be optimized

### 4.1 Keep the exhaustive loader

`load_bound_track_motion_run()` currently proves the live whole-publication
state.  It must remain available for:

- publication validation and selector activation;
- archive-wide scientific audits;
- migration and corruption diagnostics;
- consumers that truly require the full authority surface.

It should not silently become a receipt-trusting partial reader.  Existing
tests intentionally require it to hash every published array.

### 4.2 Current receipts are not normal reader authority

`verify_track_motion_payload_validation_receipt()` explicitly states that it
does not mint reader authority.  It proves a publication checkpoint.  A fast
reader must not reinterpret this existing receipt without a reviewed contract
change.

The current receipt infrastructure provides strong building blocks:

- structural and digest validation;
- decoded, physical, and immutable-metadata roots;
- scientific manifest and numerical-policy binding;
- payload verification with an optional physical rehash.

But four limitations are load-bearing:

1. **Authentication:** receipt and publication commit attrs are consistency
   evidence, not cryptographic signatures.  A writer able to rewrite the
   payload and recompute all mutable attrs is outside the current protection.
2. **Archive identity:** the receipt names a Zarr-relative `run_ref`; receipt
   verification does not by itself bind the filesystem/store identity.
3. **Concurrent mutation:** a hash is not an atomic filesystem snapshot.
   Selected arrays must be copied into owned memory and checked, with closing
   selector/metadata checks.
4. **Consolidated metadata:** a fast published reader must require the intended
   consolidated generation and reject stale direct/consolidated state.

Relevant code:

- `src/fisheye/shared/zarr_payload_receipt.py:552-691`
- `src/fisheye/analysis/track_kinematics.py:11903-11913`
- `src/fisheye/shared/archive_identity.py:52-105`
- `src/fisheye/shared/proof_verification.py:1-20,93-183`
- `src/fisheye/shared/metadata_equivalence.py:97-205`
- `src/fisheye/shared/zarr_io.py:20-32`

### 4.3 Honest projection guarantee

A projection reader may prove:

- the selected immutable run and its publication/manifest identity;
- the exact declared projection and canonical paths;
- selected metadata, dtype, shape, alias relationship, and decoded digest;
- selected-array contents copied into owned memory;
- coordinate, temporal, row-identity, and archive bindings;
- unchanged selector, manifest, receipt, and selected metadata at close.

It must explicitly **not** claim that every unselected payload byte was freshly
revalidated during that read.  Mutation of an unselected array is within the
scope of the exhaustive audit, not the projection read.  This difference must
be present in the API, receipt/evidence, documentation, and tests.

---

## 5. Proposed consolidated interface

Conceptually:

```python
load_bound_track_motion_projection(
    root,
    *,
    run_name: str,
    scope: Literal["offline", "live"],
    track_id: int,
    projection: TrackMotionProjection,
) -> BoundTrackMotionProjection
```

Design constraints:

1. `TrackMotionProjection` is a closed, versioned roster maintained by Palette;
   callers cannot pass arbitrary storage paths.
2. `FULL_V1` delegates to or preserves the exhaustive reader contract.
3. All projections use the same selector resolution, position-authority
   resolver, coordinate/temporal authority vocabulary, and publication
   identity.
4. The resolver maps logical fields to canonical physical paths and explicit
   aliases.  It never hashes both an alias and its canonical source merely
   because both names exist.
5. Returned arrays are owned C-contiguous copies and cannot change if the
   backing store changes later.
6. There is no `skip_validation`, `trust_receipt=True`, profile adapter, or
   consumer-specific monkey-patch.

A projection authorization record, if required by the final threat model,
must bind at least:

- exact archive/store identity and run path;
- publication owner/generation and selector identity;
- full manifest digest and validation/integrity receipt digests;
- projection ID/version and exact logical fields;
- canonical path, alias target, dtype, shape, and content digest per field;
- coordinate, temporal, and row-identity authority digests;
- consolidated metadata generation.

**Open disposition for existing publications:** existing validation receipts
must not be silently promoted to reader authority.  Before implementation, the
design must choose and test one of these explicit paths:

- projection evidence minted as part of future full publication validation;
- an immutable, archive-bound successor evidence record minted only after one
  exhaustive validation for existing publications; or
- operation-scoped proof reuse only, leaving independent reads exhaustive.

Whichever path is selected is evidence over the same track publication, not a
new scientific authority.  It must have a real publisher/evidence-writer to
unpatched projection-reader boundary test.

### Proposed read protocol

1. Open the selector-visible immutable publication using the required
   consolidated metadata generation.
2. Resolve the exact run and require complete, selector-eligible,
   coordinate-bound state.
3. Bind archive/store identity and the live source authorities.
4. Validate manifest, publication commit, and integrity/scientific evidence
   structurally.
5. Resolve the named projection to canonical manifest surfaces.
6. Validate selected metadata and aliases.
7. Read every selected canonical array once into owned memory and hash it once
   against the declared decoded digest.
8. Recheck selector state, run attrs, manifest/evidence digests, archive
   identity, and selected metadata after the reads.
9. Return a bound projection whose guarantee and selected field set are
   inspectable.

---

## 6. Required tests and hard gates

### 6.1 Access-count tests

Add `tests/unit/fisheye/test_track_kinematics_projection_io.py` with counting
fake nodes and hash wrappers.  Hard requirements:

- no access to unrequested arrays;
- every selected canonical array receives at most one full payload read and
  one decoded-content hash pass;
- aliases resolve to the declared canonical source without duplicate payload
  reads;
- no second full manifest rebuild occurs in one projection read;
- total decoded bytes are bounded to selected arrays plus declared authority
  arrays, with no more than 10–15% bookkeeping overhead;
- projection values are byte-for-byte equal to the same fields returned by
  the current exhaustive reader.

Keep the existing test asserting that the exhaustive publication reader hashes
every published array.

### 6.2 Tamper and replacement tests

The projection reader must fail closed for:

- selected decoded-array mutation;
- same-size selected chunk replacement;
- selected metadata mutation;
- manifest, validation-receipt, or projection-evidence digest tampering;
- run/receipt material taken from another archive;
- direct metadata changed while consolidated metadata remains stale;
- whole-child replacement at the same path;
- selector changes during the read;
- selected-array replacement between two reads;
- concurrent selected-array mutation before the closing check.

An unselected-array mutation should be an explicit semantics test: the bounded
projection may pass if all of its evidence remains valid, while the exhaustive
reader must reject.  This prevents the bounded reader from falsely claiming an
archive-wide audit.

Returned arrays must be detached copies; mutating the backing Zarr after return
must not alter them.

### 6.3 Publication/evidence tests

Extend the publication tests so projection evidence, if adopted:

- is minted only after the full publication validation and integrity receipt
  succeed;
- binds the exact publication manifest/commit and selected fields;
- prevents selector eligibility when evidence creation fails;
- cannot be substituted across archive identities or consolidated generations.

At least one test must be a real writer/evidence writer to real unpatched
projection reader round trip.  Zarr-heavy integration coverage should run
outside the Codex sandbox per repository policy; deterministic counting and
tamper tests should use in-memory fakes where possible.

---

## 7. Benchmark and telemetry plan

### 7.1 Make telemetry unavoidable

Custom campaign submission must not bypass resource sampling.  Prefer placing
the wrapper/harvest logic at the workflow executor boundary rather than only
inside one shell submission helper.

Every workflow report should link:

- process-tree resource summary;
- two-second sample JSONL;
- captured workflow stdout;
- materializer phase telemetry for each node;
- final LSF `bacct`/accounting evidence after completion.

Add `PhaseTelemetry` or equivalent CPU/I/O evidence around at least:

- `load_track_projection`;
- each swim-bout speed-level pass;
- bout consolidation/publication;
- final validation.

Report both `average_effective_cpu_cores` and efficiency relative to requested
workers and allocated slots.  `/proc` and psutil storage counters are useful
but do not directly measure network transfer, so they must not be labeled as
network-filesystem bytes.

### 7.2 Reproducible reader benchmark

Model the benchmark on:

- `src/fisheye/diagnostics/benchmark_bout_classification_v2_reads.py`
- `src/fisheye/diagnostics/trace_storage_io.py`

For each current recording, run the exhaustive and projection readers in fresh
child processes and record:

- wall and CPU seconds;
- effective cores and slot efficiency;
- decoded bytes and encoded chunk bytes touched;
- canonical array read count;
- hash count and bytes hashed;
- `/proc`/psutil I/O counters;
- peak RSS;
- consolidated metadata mode and generation;
- cache state, host, commit, archive/run identity, and projection ID.

Use repeated trials and report medians.  Wall-clock speedup is diagnostic
evidence, not a brittle CI assertion.  Array/read/hash count limits are the
hard CI gates.

The v003 reload baseline for comparison is:

```text
mean wall:       890.279 s
median wall:     903.748 s
range:           815.263–938.358 s
cumulative wall: 3,561.117 s across four tasks
```

---

## 8. Implementation order

1. **Telemetry hardening, no scientific behavior change.** Make process-tree
   sampling and final LSF accounting part of every workflow execution report;
   add CPU/I/O phases to the swim-bout loader.
2. **Freeze projection specifications.** Derive exact logical field rosters
   from consumers and alias declarations; start with `swim_bouts_v1` and
   `track_identity_v1`.
3. **Land access-count and tamper tests first.** Preserve the exhaustive-reader
   tests unchanged.
4. **Decide projection evidence for existing versus future publications.** Do
   not reinterpret the current validation receipt silently.
5. **Implement one shared projection resolver.** Reuse the established
   selector, consolidated-metadata, archive, coordinate, temporal, and row
   authority checks.
6. **Migrate swim bouts only.** Compare output exactly against the exhaustive
   reader on the four current recordings and collect cold-process benchmark
   evidence.
7. **Migrate remaining consumers one at a time.** Bout kinematics,
   visualization, activity/spatial export, and tail identity each receive a
   named projection and real-reader coverage; remove duplicate local bounded
   readers as they are subsumed.
8. **Right-size cluster resources from evidence.** Only after the read shape is
   fixed should worker, slot, and memory requests be changed per DAG node.

The first implementation should not combine projection work with a broad
rewrite of publication, storage layout, or crop authority.  The architectural
success criterion is subtraction: one shared resolver, one projection
vocabulary, and fewer consumer-specific read surfaces than exist today.

---

## 9. Acceptance criteria

The optimization is ready for production use only when all of the following
are true:

1. `swim_bouts_v1` reads no unrequested motion payload arrays.
2. Each selected canonical array is read and hashed no more than once per
   projection load.
3. The projection performs no second whole-manifest rebuild.
4. Projection output exactly matches the corresponding exhaustive-reader
   fields on the four current recordings.
5. Swim-bout scientific outputs produced from the projection match the
   exhaustive baseline under the declared numerical policy.
6. All selected-data, metadata, selector, archive-substitution, and concurrent
   mutation tests fail closed.
7. The API and evidence state clearly that unselected bytes were not freshly
   audited.
8. The real writer/evidence writer to unpatched projection reader test is in
   required CI.
9. Every benchmark records consolidated mode/generation, cache state, CPU,
   wall, I/O, decoded bytes, array reads, hashes, RSS, commit, and authority
   identity.
10. The campaign report retains process-tree telemetry and final scheduler
    accounting instead of leaving it only in stdout or transient `bjobs`
    output.

No selector activation, publication-policy change, shared-checkout update, or
claim of production readiness is implied by this audit document.
