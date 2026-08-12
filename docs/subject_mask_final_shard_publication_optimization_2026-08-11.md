# Subject-mask final-shard publication optimization

Date: 2026-08-11

Status: implementation checkpoint complete for Phases A-B and the Phase-C
publication boundary; bounded row-local quality compute is implemented,
worker-produced quality reuse is deferred, and the recording-scale Phase-D
canary is running

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
without a second dense-mask extraction pass. The full-duration canary requires
this four-member bundle-v3 path; the older three-member and dense-regeneration
paths remain explicit compatibility/repair surfaces.

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
- [x] Make the sampled-contour cache a required full-duration canary bundle-v3
  member while preserving explicit legacy three-member publication.
- [x] Forbid full ragged contours in the new worker default while retaining
  historical readers and migration.
- [x] Validate the refined composable identity during the already-required
  bounded quality computation, rather than add a separate hash-only scan.
- [x] Emit the conventional whole refined-mask SHA from that quality pass so
  the existing quality manifest remains honest and compatible.
- [x] Parallelize only the row-local quality kernel with bounded workers while
  preserving ordered source reads, digest updates, scratch writes, Zarr writes,
  and publication ownership.
- [ ] Compute row-local quality evidence with the same final physical-unit
  ownership where scientifically valid, eliminating the quality computation's
  own full dense traversal.
- [ ] Define explicit boundary/global reducers for non-row-local metrics.
- [x] Bind quality input to the exact refined manifest, component registry,
  complete dense logical-unit identity, and whole-value identities of every
  narrow source array.
- [ ] Assemble quality arrays without a second full dense-mask scan.
- [ ] Regenerate stale bitpacked/RLE/contour caches only at explicit validation,
  promotion, or maintenance boundaries.

The unchecked quality-local items are an explicit follow-on optimization. They
are not required to remove the redundant core-finalizer scan: quality still
has to read refined masks to compute scientific QC, and the current checkpoint
validates the composable identity during that same bounded traversal. A future
worker-local quality design may reuse more computation, but it must first
define scientifically correct reducers for metrics that cross row or worker
boundaries.

The bounded compute path uses write-receipt v2 and records requested/effective
workers plus its execution policy. Canary plan v8 selects four workers by
default; plan v7 and older publications remain readable and execute the prior
single-worker path. A 512-row real refined-mask benchmark on the workstation
produced identical payload digests for every candidate:

| QC workers | Seconds | Rows/s |
|---:|---:|---:|
| 1 | 9.714 | 52.7 |
| 2 | 5.445 | 94.0 |
| 4 | 3.536 | 144.8 |
| 8 | 3.366 | 152.1 |

Four workers are the provisional knee: 2.75x the single-worker throughput,
while eight workers improve only another 5%. This is implementation evidence,
not the recording-scale promotion gate.

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

That v7 canary intentionally retains the single-worker QC baseline. Its live
row evidence projected a roughly 5-6 hour exhaustive quality phase, which
motivated but does not retroactively alter the v8 bounded-compute path. The
final canary must report separate core publication and quality compute phases:
the former no longer decodes every complete dense unit, while the latter still
reads refined masks because it computes scientific QC.

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

## Failure semantics

If a parallel write fails, the destination remains a failed node-local
candidate. Its run never becomes complete, is never copied into an authority,
and cannot change a selector. The validated clip inputs remain intact, so the
publication can be retried. Phase B narrows that retry from the whole finalizer
to the failed physical unit.
