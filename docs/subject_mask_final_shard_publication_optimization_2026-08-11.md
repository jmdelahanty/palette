# Subject-mask final-shard publication optimization

Date: 2026-08-11

Status: implementation complete through the receipt-composed Phase-C quality
and sampled-contour boundary. Immutable worker QC reads now verify the exact
dense units already resident for computation; the recording finalizer composes
those receipts without another full dense-mask scan. Recording-scale Phase-D
evidence is still required before production promotion.

## Goal

Reduce the multi-hour recording-level raw/refined subject-mask publication
boundary without weakening retry safety, immutable publication, logical array
identity, or selector atomicity.

Clip workers remain private and selector-ineligible. A failed clip or physical
unit must never expose a partial recording authority. The finalizer remains the
only owner of run-group metadata, manifests, consolidation, bundle import, and
selector activation.

## Non-negotiable invariants

- [x] Compute and publication happen on node-local scratch before controlled
  import into the analysis archive.
- [x] Parallel writers may own only complete, non-overlapping physical Zarr
  chunks or shards.
- [x] No worker writes attributes, manifests, completion markers, consolidated
  metadata, or selectors.
- [x] Legacy whole-value SHA-256 manifests remain readable and valid. The new
  payload identity is explicitly versioned, storage-independent, ordered by
  fixed global row units, and bound to exact worker validation receipts.
- [x] A worker exception marks the staged run failed; it cannot publish or
  activate a partial authority.
- [x] Existing validated clip artifacts remain reusable after finalizer failure.
- [x] Dense refined `masks_roi` remains the editable authority. Compact masks,
  contours, and metrics remain derived products.

## Phase A — bounded whole-physical-unit publication

- [x] Add a configurable publication-worker count to the shared subject-mask
  core publisher.
- [x] Keep source reads, logical hashing, and receipt validation in canonical
  first-axis order.
- [x] Pipeline only row bands aligned to the array's outer-shard shape, or to
  its unsharded chunk shape.
- [x] Bound pending writes by the configured worker count so memory use cannot
  grow with recording duration.
- [x] Preserve the serial path as the default until canary evidence promotes a
  parallel setting.
- [x] Record requested/effective parallelism and the ownership policy in
  transport provenance and benchmark results.
- [x] Pass an explicit worker count from the full-duration canary.
- [x] Test byte equality, manifest validity, source-receipt enforcement,
  failure propagation, and non-overlapping physical ownership.

Acceptance gate:

- exact logical and sampled reopened hashes;
- no source-receipt, schema, metadata, or completion regression;
- peak RSS remains inside the publication job budget;
- raw and refined physical-publication wall time improves materially over one
  writer;
- no production selector or registry mutation during the canary.

Local implementation smoke (not promotion evidence): one fresh write per
setting over a 32 MiB random `uint8[128,4,256,256]` payload, with
`(4,1,256,256)` inner chunks and `(16,1,256,256)` outer shards, produced:

| Writers | Seconds | Logical MiB/s | Hash match |
| ---: | ---: | ---: | :---: |
| 1 | 0.206 | 155.2 | yes |
| 2 | 0.151 | 212.4 | yes |
| 4 | 0.132 | 242.5 | yes |

These single-process local numbers only demonstrate that compression/write
overlap is real and bounded. Selection still requires repeated recording-scale
cluster trials with RSS and I/O evidence.

## Phase B — reusable final-layout unit packages

- [x] Derive final array chunk/shard plans before clip jobs are launched.
- [x] Partition final payload packaging by complete final physical ownership,
  not merely by clip row range.
- [x] Assign units crossing clip boundaries to a deterministic boundary owner
  or assembly task.
- [x] Emit immutable unit packages with array path, exact selection, codec and
  storage-plan identity, logical digest, encoded-object digest, source clip
  identities, producer commit, and terminal status.
- [x] Reuse a complete immutable worker package on retry; missing or invalid
  packages fail closed and only the affected worker package must be rebuilt.
- [x] Make the finalizer verify complete, non-overlapping `[0, R)` coverage and
  adopt or copy already encoded final-layout objects without decode/re-encode.
- [x] Keep the assembled candidate selector-ineligible until all units,
  manifests, direct metadata, and consolidated metadata validate.

The legacy manifest-v2/v4 compatibility boundary still requires one canonical
decoded whole-value hash. The new manifest-v3/v5 path replaces only the large
dense payload's whole-value SHA with a storage-independent ordered identity:
global 256-row logical units, each containing its decoded byte count and
SHA-256. Narrow lineage and metric arrays retain conventional whole-value
SHA-256 records.

Workers validate their decoded payload against their existing semantic receipt
during final-layout packaging and emit complete logical units or authenticated
segments for a unit that crosses a worker boundary. The finalizer binds each
package to the exact worker receipt, copies already encoded complete physical
units, and decodes only logical or physical units crossing worker boundaries.
It cannot silently fall back to a full decoded payload scan in composable mode.
Legacy streaming mode remains available for old packages and rollback.

Worker packages contain only the large dense authority payloads:
`mask_probs_roi` and `masks_roi`. Narrow lineage/metric arrays remain cheap
canonical writes. Package objects are selector-ineligible transport evidence;
workers never write run manifests, completion state, consolidation, or
selectors.

Local implementation smoke (not recording-scale promotion evidence): a 32 MiB
binary `uint8[128,4,256,256]` refined payload with `[16,1,256,256]` inner chunks
and `[64,1,256,256]` outer shards produced:

| Operation | Seconds |
|---|---:|
| Upstream package construction | 0.152 |
| Phase A finalizer re-encode | 0.129 |
| Phase B finalizer adoption | 0.035 |

The already-packaged finalizer was 3.71x faster and adopted both complete row
units with zero boundary rebuilds. Package construction is shown separately
because production performs it inside independent clip jobs; adding the two
local times serially is not the recording workflow. The smoke used an
in-memory source and local filesystem, so real compressed-source decoding,
PRFS package transfer, clip boundaries, RSS, and quality publication still
require the recording-scale Phase D gate.

Implementation validation at the Phase-B checkpoint:

- 38 focused core, canary, and atomic recording-bundle tests passed;
- exact decoded equality, complete-unit adoption, cross-worker boundary
  rebuild, missing-worker rejection, encoded-object corruption rejection, and
  immutable package reuse are covered;
- Ruff, Black, Python compilation, and `git diff --check` passed for the changed
  Python surfaces.

### Contour publication boundary

Per-clip refinement defaults to full ragged contours disabled and fixed-count
sampled contours enabled. Phase C now seals each worker's sampled arrays
against that worker's dense-mask semantic receipt and carries the receipt with
the immutable worker bundle. The recording publisher validates exact,
contiguous row coverage and assembles those arrays into the access-aware cache
without a second dense-mask extraction pass. The receipt-composed full-duration
canary requires the four-member bundle-v4 path; bundle v3, the older
three-member form, and dense regeneration remain explicit compatibility/repair
surfaces.

Regeneration is required only after a dense edit, an algorithm/version change,
or stale/failed evidence. The default worker receipt rejects stale contours
and rejects any full ragged `contours` group. Historical full contours remain
readable and migratable as optional cold inspection/export data.

## Phase C — quality and cache reuse

- [x] Bind worker-produced fixed-count sampled contours to exact refined worker
  receipts, algorithm semantics, component registry, row interval, and logical
  unit hashes.
- [x] Assemble complete sampled-contour worker coverage without a second dense
  extraction pass.
- [x] Make the sampled-contour cache a required full-duration canary bundle-v4
  member while preserving explicit legacy bundle and three-member publication.
- [x] Forbid full ragged contours in the new worker default while retaining
  historical readers and migration.
- [x] Validate every dense block consumed by worker QC against the producer's
  ordered unit hashes during the already-required computation, rather than add
  a separate hash-only scan.
- [x] Preserve conventional whole-value SHA publication for the legacy
  monolithic path while giving the partitioned path an explicitly versioned
  composable source reference.
- [x] Parallelize only the row-local quality kernel with bounded workers while
  preserving ordered source reads, digest updates, scratch writes, Zarr writes,
  and publication ownership.
- [x] Compute row-local quality evidence from each terminal refined worker,
  with exact dense-worker receipt and global row/frame-interval binding.
- [x] Freeze the current quality profile as observation-local; it has no
  cross-worker reducer. Any future temporal/global metrics require a separate
  profile and explicit reduction stage.
- [x] Bind quality input to the exact refined manifest, component registry,
  complete dense logical-unit identity, and whole-value identities of every
  narrow source array.
- [x] Assemble the worker-produced quality arrays without recomputing connected
  components, topology, containment, or overlap metrics over the complete
  recording.
- [x] Replace the compatibility whole-value source hash with a composable
  receipt-bound source reference so the final publisher need not perform its
  remaining ordered dense identity-verification scan.
- [ ] Regenerate stale bitpacked/RLE/contour caches only at explicit validation,
  promotion, or maintenance boundaries.

Each quality job owns one terminal refined worker and writes one node-local
partition containing only the 11 row-aligned quality arrays (the final
`frame_row_offsets` index is derived during assembly). Its immutable receipt
binds the exact refined-worker receipt, every ordered dense-mask validation
unit, component and quality contracts, producer commit, frame interval, row
interval, array shapes/dtypes, and fixed-size output logical-unit hashes. The
partition refuses to publish if any dense byte read for QC differs from its
producer receipt. The recording publisher requires gap-free ordered `[0, R)`
coverage and never lets workers write the final Zarr concurrently.

Partition receipt/assembly v2 binds the complete refined producer-evidence
digest. Quality manifest v3 and logical-content v2 carry the exact refined
array declarations rather than inventing a whole-recording dense SHA. Quality
write-receipt v4 records zero finalizer compute and
`receipt_bound_partitions_with_verified_worker_units_v2`. The finalizer still
reads and verifies the narrow lineage arrays, adopts the quality payload, and
writes/consolidates its publication, but it does not read `masks_roi`.

Sampled-contour assembly v2 binds the same refined producer evidence. Cache
manifest v3 and cache-receipt v2 bind the refined composable dense identity and
the exact worker assembly; bundle manifest v4 requires the quality and cache
members to name that same refined authority. Legacy quality manifests,
whole-value source hashes, cache receipts, and bundle versions remain accepted
on their explicit compatibility paths.

Old quality partition receipts do not claim that QC reads were compared with
producer units, so they cannot be upgraded by metadata rewriting. A canary
created before partition receipt v2 must rerun only the quality partitions.
Its sealed inference/refinement final-layout packages remain reusable when
their existing worker receipts validate. Sampled-contour worker payloads also
remain reusable; their recording-level assembly receipt is rebuilt cheaply.

Canary plan v9 requires partitioned QC, selects four compute threads per
partition and ten concurrent partition jobs by default, and makes recording
publication depend on terminal quality coverage. Plan v8 remains loadable and
retains the monolithic compatibility path. A 512-row real refined-mask
benchmark on the workstation produced identical payload digests for every
candidate:

| QC workers | Seconds | Rows/s |
|---:|---:|---:|
| 1 | 9.714 | 52.7 |
| 2 | 5.445 | 94.0 |
| 4 | 3.536 | 144.8 |
| 8 | 3.366 | 152.1 |

Four workers are the provisional knee: 2.75x the single-worker throughput,
while eight workers improve only another 5%. This is implementation evidence,
not the recording-scale promotion gate.

At the measured four-thread rate, one roughly 53,000-row Sleepyfish clip is
about six minutes of QC. Twenty-two clips at concurrency ten require three
waves, so approximately 18-25 minutes is a reasonable hypothesis for the
scientific QC phase, versus roughly six hours for one single-threaded
recording traversal. This is a projection, not a completed cluster benchmark;
scheduling, PRFS transfer, final Zarr writes, and consolidation are additional
publication time. The prior finalizer dense-scan cost is retired for the
receipt-composed path.

The first full-duration partitioned-quality publication measured 2,709.95 s
total, of which 2,229.18 s was the finalizer's redundant ordered dense-source
verification and only 0.189 s was quality-partition adoption. Removing that
scan leaves roughly 481 s of measured non-scan work, so approximately eight
minutes is the current full-duration finalizer hypothesis. This subtraction is
not promotion evidence: the next selector-ineligible canary must measure the
new implementation directly, including I/O, RSS, and consolidation.

## Phase D — benchmark and promotion

- [ ] Compare 1, 2, and 4 publication workers on the same immutable inputs.
- [ ] Record raw/refined/quality phase time, total wall time, CPU utilization,
  read/write bytes, write counts, peak RSS, NUMA placement, and failures.
- [ ] Verify identical logical digests and Crimson-visible declarations.
- [ ] Inject one unit failure and prove retry does not recompute successful
  units or change authority.
- [ ] Run one selector-ineligible full-duration canary.
- [ ] Promote a bounded default only after required CI and canary gates pass;
  retain the serial setting as rollback.

The fresh composable canary completed all 22 inference and 22 refinement
workers from worker commit `e523c816`. Publication then failed closed on two
previously unexercised recording joins: refined workers persisted a virtual
`<collection>` source path despite binding one exact raw-worker sidecar, and
recording-common authority incorrectly included worker-local cache paths,
timestamps, and eye-assignment summaries. Commits `b3c3066c` and `5b2be8df`
respectively fixed those contracts without changing or recomputing the sealed
workers. Publication-only job `153370229` is reusing those immutable bundles
and records its publication commit separately from the worker commit.

That v7 canary intentionally retained the single-worker QC baseline. Its live
row evidence projected a roughly 5-6 hour exhaustive quality phase, which
motivated but does not retroactively alter the v9 partitioned path. A fresh
plan is required to exercise receipt v2/v4 composition; existing artifacts are
never rewritten to claim the stronger evidence. The final canary must report
worker QC, partition adoption, receipt-bound source verification, core
publication, cache publication, consolidation, and total wall time separately.

Implementation validation for the composable checkpoint:

- 64 focused core, quality, cache, bundle, recording-bundle, and canary tests
  passed in one combined outside-sandbox run;
- a fail-on-read source proves complete adopted physical units are published
  without finalizer decoding;
- package schema v2 binds the exact worker array-unit receipt and is mandatory
  in the composable canary;
- quality rejects a changed dense value against the composed identity while
  retaining its bounded block computation;
- the end-to-end recording fixture publishes coordinate-aware core manifest v5,
  carries the quality-derived whole-mask digest into the sampled-contour cache,
  binds the four-member bundle, and opens it through the strict inactive
  coordinate-authority reader;
- legacy v2/v4 publication tests remain green;
- Ruff, Black, Python compilation, and `git diff --check` pass.

Additional partitioned-quality implementation validation:

- 29 partition, quality-publication, canary, and recording-bundle tests passed;
- all 10 recording-bundle publication tests passed, including the long
  coordinate-bound fixture and six independent multi-worker/cache fixtures;
- serial and partitioned QC arrays are exactly equal, including canonical NaN
  handling;
- tampered partition arrays, source receipts, gaps, overlaps, reordered rows,
  duplicate instance keys, and incomplete assemblies fail closed;
- current plans insert a dedicated LSF quality array between refinement and
  publication, while old plans remain readable compatibility inputs;
- static compilation, Ruff, Black, and `git diff --check` pass.

Receipt-composition validation:

- 57 focused quality-schema, partition, cache, extension, bundle, and
  recording-publication tests pass outside the sandbox;
- changing one dense mask byte while retaining the producer receipt fails in
  the QC worker before a partition receipt can be emitted;
- the recording-level coordinate-aware fixture publishes refined composable
  authority, quality manifest v3/write-receipt v4, cache manifest v3/receipt
  v2, and four-member bundle v4 without a finalizer dense read;
- recomputed cache-manifest tampering that separates the cache worker assembly
  from its refined producer evidence fails closed;
- historical whole-hash publication and bundle tests remain green.

## Failure semantics

If a parallel write fails, the destination remains a failed node-local
candidate. Its run never becomes complete, is never copied into an authority,
and cannot change a selector. The validated clip inputs remain intact, so the
publication can be retried. Phase B narrows that retry from the whole finalizer
to the failed physical unit.
