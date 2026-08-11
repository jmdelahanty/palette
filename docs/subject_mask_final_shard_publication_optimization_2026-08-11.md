# Subject-mask final-shard publication optimization

Date: 2026-08-11

Status: implementation checklist; Phase A is the current checkpoint

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
- [x] The canonical logical SHA-256 order and source-validation receipt checks
  remain unchanged.
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

- [ ] Derive final array chunk/shard plans before clip jobs are launched.
- [ ] Partition work by final physical ownership, not merely by clip row range.
- [ ] Assign units crossing clip boundaries to a deterministic boundary owner
  or assembly task.
- [ ] Emit immutable unit packages with array path, exact selection, codec and
  storage-plan identity, logical digest, encoded-object digest, source clip
  identities, producer commit, and terminal status.
- [ ] Retry only missing or failed units.
- [ ] Make the finalizer verify complete, non-overlapping `[0, R)` coverage and
  adopt or copy already encoded final-layout objects without decode/re-encode.
- [ ] Keep the assembled candidate selector-ineligible until all units,
  manifests, direct metadata, and consolidated metadata validate.

This phase is what eliminates recording-scale raw/refined rematerialization.
Phase A improves today's required rematerialization but does not claim to
remove it.

## Phase C — quality and cache reuse

- [ ] Compute row-local quality evidence with the same final physical-unit
  ownership where scientifically valid.
- [ ] Define explicit boundary/global reducers for non-row-local metrics.
- [ ] Bind quality unit receipts to the exact refined dense-mask digest.
- [ ] Assemble quality arrays without a second full dense-mask scan.
- [ ] Regenerate stale bitpacked/RLE/contour caches only at explicit validation,
  promotion, or maintenance boundaries.

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

## Failure semantics

If a parallel write fails, the destination remains a failed node-local
candidate. Its run never becomes complete, is never copied into an authority,
and cannot change a selector. The validated clip inputs remain intact, so the
publication can be retried. Phase B narrows that retry from the whole finalizer
to the failed physical unit.
