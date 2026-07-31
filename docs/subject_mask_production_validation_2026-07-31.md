# Subject-Mask Production Validation

Date: 2026-07-31

Status: maintained raw inference, refined finalization, whole-recording, and
clipped-recording DAGs now produce proof-bound inactive bundle candidates;
the 22,926-row paired validation-mode gate passes, reference-full remains the
API default, the isolated full-duration driver passes real-input preflight,
and no selector, registry authority, or full-duration job has been activated

## Decision

Full-duration subject-mask publication must not repeat the small-canary
validation oracle over the complete decoded surface before and after writing.
For Cam2010095, approximately 1,169,010 rows of four-component
`uint8[512,512]` refined masks represent about 1.1 TiB of logical pixels.  The
reference publisher's repeated semantic scans and hashes would process several
TiB and duplicate work already performed by inference and refinement.

Every row must still be validated.  Production obtains that guarantee from
bounded worker evidence accumulated while values are already resident, not
from repeatedly reopening the completed full snapshot.

## Explicit Modes

`SubjectMaskCoreValidationMode.REFERENCE_FULL` remains the API default.  It:

1. validates the complete source schema and crop lineage;
2. recomputes every derived metric from the authoritative pixel surface;
3. writes the immutable byte-planned layout;
4. hashes every completed array;
5. reopens the store, repeats full semantic validation, and rehashes it.

This is the reference oracle for small fixtures, new schema/profile canaries,
and deliberate offline audits.

`SubjectMaskCoreValidationMode.PRODUCTION_STREAMING` is explicit opt-in.  It:

1. requires a versioned, digest-bound source-validation receipt;
2. requires complete, ordered, non-overlapping semantic row coverage;
3. checks source run, manifest, schema, dimensions, components, threshold,
   exact path inventory, shape, dtype, and logical-array digests;
4. hashes the exact bytes during the one read already required to write each
   complete outer shard or unsharded chunk;
5. fails and marks the run failed if streamed bytes differ from the receipt;
6. verifies direct/consolidated metadata equivalence; and
7. reopens exact array metadata plus bounded first and last physical row bands
   of each array instead of rescanning the full payload.

The run manifest is version 2 and records the validation mode, source-receipt
digest and sidecar binding, hash timing, physical write counts, and bounded
reopen samples. The potentially large unit inventory lives in the strict
`source_validation_receipt.json` sidecar rather than inflating consolidated
Zarr metadata; the manifest binds its exact canonical bytes.

## Source-Validation Receipt

The v1 receipt is `palette.subject_mask.source_validation_receipt`.  It binds:

- exact source run path and canonical source-manifest digest;
- raw or refined logical schema identity;
- complete dimensions and ordered component registry;
- raw probability threshold when applicable;
- the closed publication-array inventory;
- exact shape, dtype, and C-order logical SHA-256 for each array; and
- ordered semantic-validation units covering `[0, n_rois)` exactly once.

Each unit is valid only under
`palette.subject_mask.source_semantics@1` and carries its own evidence digest.
Missing rows, overlapping rows, stale/recomputed outer digests, unknown fields,
wrong validators, changed manifests, and changed array bytes fail closed.

`build_reference_subject_mask_validation_receipt` exists only to create oracle
receipts for small fixtures and equivalence tests.  Full-duration production
must construct the same final receipt incrementally from inference/refinement
workers and an ordered publication coordinator; it must not call the reference
builder over the completed full surface.

## 22,926-Row Paired Canary

The current implementation passed a real local-scratch paired gate on 2026-07-31
using the completed crop-v2 cache-pipeline handoff at:

`/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/subject_mask_storage/integration/20260128_cropv2_subject_mask_cache_pipeline_20260731_v2`

The driver staged the 73 MiB compressed raw/refined fixture and the 3.7 MiB
crop-v2 archive to `/tmp`, then processed 23,287 acquisition frames, 22,926
mask observations, 18.03 GB of logical raw arrays, and 24.04 GB of logical
refined arrays. It created fresh current-code `reference_full_v1` and
`production_streaming_v1` publications from the same staged source.

| Surface | Receipt scan | Reference-full publication | Production-streaming publication | Streaming/reference |
| --- | ---: | ---: | ---: | ---: |
| Raw, 3 components | 135.45 s | 322.51 s | 41.79 s | 0.130 (7.72x faster) |
| Refined, 4 components | 163.17 s | 382.44 s | 55.09 s | 0.144 (6.94x faster) |

The receipt scans are an integration-fixture cost only. In production, the
same evidence is accumulated while inference/refinement values are resident;
it is not a separate completed-store scan. Even if that fixture-only scan is
charged to streaming, receipt plus publication took 177.24 s raw and 218.26 s
refined, versus 322.51 s and 382.44 s reference-full.

Both pairs had exact equality for:

- logical-content documents and every C-order array SHA-256;
- logical schema and byte-derived physical storage plan;
- every array's direct physical metadata declaration;
- normalized direct/consolidated metadata;
- frame offsets, crop lineage, shapes, dtypes, and component ordering; and
- persisted bounded-reopen validation and receipt-sidecar binding.

All four generated stores remained benchmark-only, selector-ineligible, and
unregistered, with no parent selector attributes and no production-state
changes. Raw and refined logical-content digests remained, respectively,
`e0292091f7bc9ee2b1942b750404abf1da6da683c5dee848b7c5d400bef4408b`
and
`7a10d0b1a5446f578fcda02b9d3460a7ad95074d64d28ec6ead45f2435a301e4`.

The complete result payload digest is
`5b89af1a5f34cdb022ccf05cca1e785187f22599ba7b80433d63c7a485474f20`;
the strict JSON file SHA-256 is
`46516c4cf38ec926ddce6e33e2102666c14a908c69eb08a9b2dfcc69f57cbec9`.
The run correctly records `worktree_dirty=true` and base commit
`a9e47db16ede1f8a96a30e905c852983b2514ebe`; it is strong development
evidence, not immutable revision-bound promotion evidence. The reduced record
is retained at
`docs/diagnostics/subject_mask_validation_modes_2026-07-31/summary.json`.

Peak RSS uses cumulative same-process `ru_maxrss`, so it proves the reference
path can expand memory substantially but is not a fair isolated mode
comparison. Raw reference-full raised the process high-water mark by 2.23 GB;
the later streaming phases raised the already-high mark by only 11.6 MB raw
and 19.9 MB refined. A fresh-process full-duration canary remains responsible
for the promotion-quality RSS comparison.

## Production Lifecycle

### ROI-cache topology

Cache boundaries follow real recording structure; they are not invented for
scheduler convenience.

- An unclipped recording has one contiguous flat `uint8[N,H,W]` ROI cache and
  one streaming raw-mask inference attempt over that cache.  The cache is
  copied to node-local scratch once and authenticated by SHA-256 during that
  same sequential copy.  Publication may seal bounded row/physical-shard
  receipts, but those receipt units are not clips and do not create additional
  cache identities.
- A clipped recording collection has one flat cache and one terminal inference
  receipt per real clip.  Scheduler jobs may bundle cache construction, but
  every clip retains its own source identity, cache digest, attempt lineage,
  and terminal result.
- Future parallelism for an unclipped recording must stage the single cache
  once per host and share it read-only.  It must not create synthetic clips or
  duplicate the cache merely to obtain workers.

Logical safety is independent of this topology: every writer still owns whole,
non-overlapping physical output units, and the recording-level coordinator
proves complete ordered row coverage before publication.

### Raw inference

- Validate exact output shape, dtype, probability encoding, component order,
  row identity, and crop lineage while each inference unit is resident.
- Compute canonical probability maxima and derived binary metrics in that pass.
- Emit terminal success/failure evidence for each owned row interval.
- Accumulate ordered logical hashes while sealing the recording-level source.

### Refined finalization and editing

- Validate newly computed or edited rows/components while resident.
- Preserve immutable receipts for unchanged inherited units.
- Mark dependent metrics and caches stale for interactive edits.
- At compaction, prove complete non-overlapping coverage and construct a new
  immutable validation receipt.
- Do not audit every unchanged pixel during an interactive edit.

### Immutable publication

- Require the sealed source receipt.
- Rematerialize complete output physical units through the byte planner.
- Compare publication-stream hashes with receipt hashes.
- Publish no selector until metadata, receipt binding, completion, and bounded
  reopen checks pass.

### Recording-level atomic bundle

Publication creates and validates three immutable recording-level runs before
selection:

1. `subject_mask_runs/<raw>` contains access-aware raw probability authority;
2. `refined_subject_masks_runs/<refined>` contains dense editable authority;
3. `subject_mask_quality_runs/<quality>` contains the separately derived
   scientific quality surface.

The runs are imported from node-local outputs, remain individually
selector-ineligible, and are cross-bound by exact row identity, frame offsets,
crop geometry, component availability, source-manifest digests, and quality
source hashes.  `subject_mask_bundle_runs/<bundle>` is completed only after all
three direct and consolidated metadata surfaces reopen and validate.  A later
explicit activation changes one root `subject_mask_authority` envelope; it does
not advance three independent family selectors.

Immutable target names are preflighted before the first import.  An interrupted
pre-commit activation leaves no usable authority: the next lock holder repairs
readiness from the last committed authority, clears the stale lease,
reconsolidates metadata, and repeats full bundle validation before attempting a
new commit.

### Whole and clipped recording adoption

- Whole-video workflows stage the single recording cache to node-local scratch,
  produce one proof-bearing raw shard and refined draft, and invoke the common
  recording bundle publisher.
- Clipped workflows preserve one attempt and semantic receipt per real clip.
  The importer proves exact, non-overlapping crop-row coverage, treats
  `available_channels[C]` as a collection-global declaration, assembles one
  recording-level refined draft, and publishes the raw clip shards through the
  same common bundle publisher.
- Validation jobs depend on completed bundle publication and validate the
  bundle candidate.  Registry finalization ignores running, failed, malformed,
  and inactive candidates.  Bundle members become readable through the modern
  registry path only when they match the exact committed root authority.
- Both paths still default to inactive publication.  This code does not enable
  production selection merely because a candidate validates.

### Scientific quality

`subject_mask_quality_runs` remains a separate scientific computation.  It may
intentionally traverse every row once.  Publication validation proves internal
and lineage correctness; it does not replace containment, overlap, temporal, or
anatomical quality assessment.

## Full-Duration Driver Preflight

`fisheye.cluster.subject_masks.full_duration_canary` now owns one immutable,
benchmark-only execution boundary for both recording layouts. A clipped
recording uses only the real materialized clip index; an unclipped recording
uses one whole-video window. It never invents scheduler clips. The driver:

1. requires a clean, commit-pinned Palette checkout for reproducible runs;
2. verifies the model SHA-256 and current crop/refined-keypoint manifests;
3. freezes clip-index, clip-manifest, video-file, model, and reference-tree
   identities;
4. proves every frame and every crop row is owned exactly once;
5. stages pixels, geometry, keypoints, and the model to node-local scratch;
6. publishes each worker result as an atomic immutable bundle only after its
   completion and proof envelopes reopen;
7. runs GPU inference, CPU refinement, and recording publication as three
   separate dependency-barriered LSF stages; and
8. emits only an inactive raw/refined/quality bundle below
   `.palette_benchmarks`, with no registry or authority mutation.

The real Cam2010095 development preflight covered 22 maintained 54,000-frame
clips, 1,188,000 acquisition frames, and 1,169,010 crop/keypoint rows. Its plan
digest was
`6820c7d7e7f4020d6457837b2ac4cf0a50ab88f18ed414989c1c3d3bf7275d4d`.
The dry-run topology contained 22 GPU inference tasks (maximum concurrency
four), 22 CPU refinement tasks (maximum concurrency four), and one final
recording publisher, with all-success barriers between stages.

The preflight also exposed one historical input-contract transition. The
full-duration refined-keypoint snapshot used source-bindings v1, which lacked
the now-required inline skeleton semantics. The new contract republisher
accepts only that exact v1-to-v2 transition, preserves every logical-array
hash plus lineage/snapshot identity, writes a new selector-ineligible
access-aware companion, and rejects any additional source-manifest defect. On
the real 1,169,010-row snapshot it preserved logical-content digest
`8c71d9e85f796263ffc23df9ffde447230a7cddddbda8e98df75653604d6f721`,
wrote 23 sharded arrays as 27 payload objects, and completed in 36.74 seconds.
That `/tmp` run explicitly recorded a dirty development checkout and is
preflight evidence only; the cluster canary must use a fresh companion and
plan produced by the final clean deployed commit.

## Implementation Checklist

- [x] Preserve reference-full as the default.
- [x] Add explicit production-streaming mode.
- [x] Freeze the incremental source-validation receipt schema.
- [x] Enforce exact contiguous semantic row coverage.
- [x] Bind source manifest, schema, dimensions, components, threshold, arrays,
      shapes, dtypes, and logical hashes.
- [x] Compute publication hashes during required output writes.
- [x] Fail and mark the candidate failed when source bytes differ from receipt.
- [x] Replace the production full reopen with exact metadata plus bounded
      first/last physical-row-band checks.
- [x] Keep the complete unit receipt in a digest-bound strict JSON sidecar
      rather than inline consolidated metadata.
- [x] Keep candidates selector-ineligible.
- [x] Add small raw/refined mode-equivalence and adversarial tests.
- [x] Emit raw worker semantic receipts during maintained inference.
- [x] Emit refined worker semantic receipts during maintained finalization.
- [x] Aggregate logical hashes during the ordered recording-level merge rather
      than rereading the completed source.
- [ ] Bind manual-edit/compaction receipts for changed and inherited units.
- [x] Compare reference-full and production-streaming manifests and logical
      hashes on the completed 22,926-row canary.
- [x] Implement the isolated clipped/whole-video full-duration driver, exact
      reference/clip preflight, node-local staging, atomic worker bundles, and
      dependency-barriered LSF plan.
- [x] Distinguish keyed delta packages from proven complete frame-window
      partitions. Complete partitions now require exact crop-offset coverage,
      digest-bound collection/clip identities, and independent finalizer
      validation; the ordinary work-package default remains delta-only.
- [ ] Benchmark full-duration phase time, decoded bytes, peak RSS, and bounded
      reopen reads.
- [ ] Run one selector-ineligible full-duration production-streaming canary.
- [ ] Obtain Palette and Crimson correctness/performance review.
- [ ] Activate a versioned profile only after those gates pass.

## Safety Boundary

The implementation has passed 444 maintained changed-surface tests with one
environment-dependent skip, including
real-Zarr single-source whole-recording publication and a two-clip proof import
that flows through raw/refined/quality publication into one inactive bundle.
Those fixtures prove lifecycle behavior and exact decoded content on small
surfaces; they are not full-duration performance or promotion evidence.

The first isolated 54,000-frame inference preflight completed successfully and
published a terminal, selector-ineligible worker bundle. It then exposed that
the inference writer had labeled every work package as a delta even when the
driver had supplied the exact complete crop-row partition for one authenticated
clip. The refinement finalizer correctly failed closed. The writer and
finalizer now share the explicit complete-partition proof above. A fresh
single-clip inference/refinement preflight is still required before launching
all 22 windows; no production selector, registry authority, or archive has been
changed.

The shared inference hardware/runtime contract is documented in
`docs/inference_accelerator_provenance_2026-07-31.md`. Subject-mask publication
receipts retain the upstream stage and run provenance before scratch cleanup.
