# Publication receipt and hashing lifecycle — 2026-08-29

Status: **design decision and optimization contract**. This governs new work;
existing publication families may require a compatibility migration before
their redundant payload scans can be removed.

Legacy behavior was observed against Palette commit `bf058521`
(`subject-shape-unbound-receipt-20260827-bf058521`). The receipt-v2
implementation checkpoint is `e35714b0`.

## Decision

When a trusted receipt covers the final immutable payload, normal publication
must not decode or hash that payload again.

Payload identity is established while data is written or transformed.
Publication verifies the resulting receipt chain, metadata, authority,
ownership, completion, and atomic placement. A complete decoded rehash remains
an explicit audit or recovery operation, not a routine planning, publication,
or consumption step.

This is not a reduction in validation strength. It moves the integrity proof to
the point where the data is already in memory and makes the proof reusable at
later boundaries.

## Required lifecycle

### 1. Planning

Planning is read-only and cheap. It resolves supported producer profiles and
already-published input receipts. It may validate metadata and receipt seals,
but it does not read large payload arrays, create output receipts, or predict
digests for data that has not been written.

### 2. Writing and transformation

The process that creates or transforms payload bytes records their identity as
part of the same attempt. Each write unit must bind at least:

- the publication attempt and owned row/chunk interval;
- array path, shape, dtype, physical chunking, and codec declaration;
- row identity and relevant scientific/coordinate authority;
- the exact transformation and implementation/profile version; and
- a digest of the final bytes or logical values governed by that receipt.

Parallel writers may issue receipts only for whole, non-overlapping physical
chunks they own. Logical row slices that share a physical Zarr chunk are not
independent write units.

### 3. Finalization

Finalization proves that the expected write-unit receipts are complete,
non-overlapping, ordered, and bound to one attempt. It deterministically seals
them into a publication receipt.

A flat SHA-256 of a complete array cannot be reconstructed from independent
SHA-256 chunk digests. Parallel output therefore needs a versioned composable
identity, such as a chunk/Merkle manifest, or a single ordered writer that
computes the required flat digest while writing. The receipt schema must state
which identity it uses; consumers must never silently treat one as the other.

### 4. Atomic publication

Routine publication validates:

- receipt schema and seal;
- completeness and exact write-unit coverage;
- attempt, producer, source-authority, and row-domain bindings;
- array topology and metadata declarations;
- expected publication path and ownership;
- completion and selector policy; and
- the final consolidated metadata generation.

An atomic rename on the same filesystem does not change payload identity and
therefore does not require a payload rehash. A copy or cross-storage transfer
must produce a transfer receipt bound to the source publication receipt. If the
transport cannot prove the transfer, a destination-side verification scan is
required once.

### 5. Consumption

Normal consumers validate the sealed publication receipt and the metadata
generation required by the publication contract. They must not routinely
recompute full payload digests already proven by that receipt.

### 6. Deep audit

Full decoded equality checks, re-derivation, and payload rehashes remain
available as explicit operations. They are required when:

- the writer did not issue a compatible receipt;
- any payload or covered metadata changed after receipt creation;
- publication performed an unreceipted transformation;
- a transfer lacks sufficient integrity evidence;
- the receipt is absent, incomplete, invalid, or uses an unsupported profile;
- recovery is being attempted; or
- an operator explicitly requests a deep audit.

These conditions fail closed. They must not trigger an unreported expensive
fallback during ordinary planning or publication.

## Baseline defect demonstrated by the subject-shape canary

The commit-`bf058521` subject-shape writer does not emit reusable evidence for
its final parallel output. Finalization consequently rereads completed arrays in
`build_subject_shape_unbound_payload_scan_receipt` to manufacture flat payload
hashes. Other admission paths have also reloaded payloads to reproduce claims
already sealed by upstream receipts.

That scan is compatibility work caused by a missing write-time receipt; it is
not subject-shape computation and is not the intended steady-state publication
design.

The durable optimization is to make the already-required storage transform
issue the write-unit receipt while it has source and destination values in
memory, then teach publication and consumers to accept that versioned
composable identity.

## Subject-shape v2 implementation checkpoint

Status: **implemented and locally validated in
`agent/palette/subject-shape-plan-receipts-20260829` at `e35714b0`; pushed to
draft PR #73. Required CI passed at that runtime commit and the
selector-ineligible performance canary completed successfully. The canary also
exposed a separate pre-existing nondeterministic ellipse-fit edge case, so the
canary remains ineligible and the branch is not merged or production-active**.

The projected access-aware subject-shape path now implements the first durable
receipt profile:

`verified_staged_transfer_plus_binding_append_receipt_v2`.

The proof chain is:

1. The parallel compute writer closes its scratch tree with a sealed,
   metadata-only deferred-transform receipt. It does not invent or predict any
   payload digest. That receipt can be consumed only by the named access-aware
   storage profile under exclusive scratch ownership.
2. Access-aware storage conversion writes complete physical outer units,
   immediately reads each unit back, and emits the closed decoded copy report
   and flat per-array content digests from that same traversal. Exact decoded
   source-to-destination equality is checked before the receipt closes.
3. Only after every copy/readback succeeds does conversion consume the deferred
   receipt and stamp the ordinary sealed unbound manifest. A missing, changed,
   leftover, or profile-mismatched deferred receipt fails closed.
4. The shared atomic publisher passes its already-verified physical-copy
   receipt into the existing post-rename callback. This is one shared
   publication interface; no subject-shape-specific transfer adapter was
   added.
5. Final-path binding verifies that the numeric projection is already in
   source-camera coordinates and therefore does not rewrite any staged array.
6. Binding creates and reads back only the six appended arrays:
   `instance_key`, `source_crop_row_ids`,
   `source_acquisition_frame_index`, `component_centroid_xy`,
   `component_centroid_valid`, and `body_frame/axis_valid`.
7. The staged and appended decoded receipts must partition the exact live
   array inventory. Missing, overlapping, or unexpected arrays fail closed.
8. The final target receives one physical-payload hash and immutable-metadata
   receipt. The scientific validation receipt binds that final integrity root,
   the coordinate manifest, the transfer evidence, and the decoded receipt
   composition.

The old `single_locked_bound_payload_receipt_v1` grammar remains readable and
continues to be emitted for paths that genuinely transform numeric geometry at
final binding or do not supply the v2 transfer evidence. Its exported constant
retains its historical meaning; v2 has a distinct explicit constant and
profile marker.

This removes both complete decoded rescans from the maintained projected
access-aware path: the post-compute unbound scan and the post-binding scan. It
deliberately does **not** remove the final physical-file hash yet: that scan
covers the low-single-digit-gigabyte compressed tree, not the hundreds of
gigabytes delivered by decoded array traversal. Reusing physical leaves across
transport is a smaller follow-up and requires the atomic transport to expose
compatible per-file leaves, not merely an aggregate copy digest.

Local evidence so far:

- 62 complete affected atomic-publisher and subject-shape tests passed in
  425.92 seconds;
- the final instrumented access-aware regression passed in 189.94 seconds;
- that regression compares every v2 declared final array digest with the live
  final array, guarding against reuse of a staged digest after a binding write;
  and
- that regression makes both
  `build_subject_shape_unbound_payload_scan_receipt` and
  `_scan_subject_shape_bound_payload` raise if the v2 path attempts either
  complete decoded rescan.

### Completed legacy baseline

LSF job `153770521` ran the immutable commit-`bf058521` implementation against
Cam2010094 and completed successfully on 2026-08-29 at 22:17:31 UTC. The run
`subject_shape_fused_projection_canary_20260829_bf058521_cam2010094` is
complete, selector-ineligible, bound under `bound_canonical_v2`, and sealed
with the historical `single_locked_bound_payload_receipt_v1` profile. Registry
writes were disabled.

| Phase | Wall time | Delivered read characters |
|---|---:|---:|
| plan | 1,403.552 s (23m24s) | 37.31 GB |
| scientific compute, including the old post-compute receipt scan | 6,764.819 s (1h52m45s) | 396.67 GB |
| access-aware storage conversion | 361.063 s (6m01s) | 2.78 GB |
| atomic publication | 9,826.970 s (2h43m47s) | 826.43 GB |
| cleanup | 22.871 s | negligible |

Within atomic publication, `post_rename_binding` alone took 9,748.514 seconds
(2h42m29s) and delivered 818.42 GB of reads. The whole job took 18,390.019
seconds (5h06m30s), delivered 1,460,813,022,742 read characters, and averaged
2.507 effective CPU cores across 32 allocated slots: 7.835% slot efficiency.
The OS reported only 6,487,134,208 physical storage-read bytes, so delivered
read characters primarily measure repeated decode/cache traversal rather than
network or physical-disk traffic.

A metadata-file census of the baseline target found 106 arrays totaling 10.266
GiB of logical values. Its final physical-integrity receipt covers
1,439,980,758 bytes (about 1.34 GiB). Therefore the 1.461 TB of delivered read
characters was about 133 times the complete logical payload and about 1,014
times the final compressed tree. Delivered read characters include page-cache
traffic and are not physical storage bytes, but the ratios still expose
repeated sharded decode traversal rather than one sequential verification pass.

### Completed receipt-v2 canary

LSF job `153770860` ran commit `e35714b0` against the same Cam2010094 inputs,
on the same host and 32-slot resource request. It started immediately after the
baseline at 22:17:31 UTC, completed with exit code zero at 23:03:58 UTC, and did
not overlap the baseline. The run
`subject_shape_receipt_v2_canary_20260829_e35714b0_cam2010094` remains
selector-ineligible, registry writes were disabled, and the consolidated root
contains the completed `bound_canonical_v2` publication.

| Phase | Legacy `bf058521` | Receipt v2 `e35714b0` | Speedup |
|---|---:|---:|---:|
| plan | 1,403.552 s | 23.080 s | 60.81x |
| scientific-compute envelope | 6,764.819 s | 995.189 s | 6.80x |
| access-aware storage conversion | 361.063 s | 348.977 s | 1.03x |
| atomic publication | 9,826.970 s | 1,299.618 s | 7.56x |
| post-rename binding | 9,748.514 s | 1,222.107 s | 7.98x |
| total scheduler-observed job | 18,390.019 s | 2,786.824 s | 6.60x |

Total wall time fell from 5h06m30s to 46m27s, an 84.85% reduction. Delivered
read characters fell from 1,460,813,022,742 to 475,337,630,238, a 67.46%
reduction. Average effective CPU use increased from 2.507 to 9.950 cores and
32-slot efficiency rose from 7.835% to 31.095%, approximately 3.97 times the
baseline efficiency.

The receipt chain closed as designed:

- the final profile is
  `verified_staged_transfer_plus_binding_append_receipt_v2`;
- all 100 storage-transform array digests equal the corresponding 100 carried
  final digests;
- the six declared binding arrays are the only appended paths;
- the final receipt records
  `verified_staged_transfer_plus_final_binding_readback_v2` as its digest
  source;
- the physical-copy receipt covers 1,365,323,664 bytes in 1,168 files and was
  verified by an `rsync --checksum` dry run; and
- the published consolidated metadata records completion, selector
  ineligibility, canonical binding, and the v2 profile.

There is no hidden legacy decoded-payload scan. The remaining 1,222.107-second
binding interval is dominated by `authority_stamping`: 912.950 seconds and
43.008 GB of delivered read characters. The final `physical_payload_hash` took
only 2.912 seconds and delivered 1.441 GB. The authority interval wrote only
90.876 MB, which points to large-metadata read/modify/write amplification, not a
second numeric transformation. Admission revalidation (113.308 seconds),
coordinate-source binding/projection (111.018 seconds), and identity-array
append (69.914 seconds) are the other material subphases. Reducing repeated
authority-metadata serialization is the next publication optimization; it is
not evidence for removing the receipt checks.

### Scientific parity finding

The receipt-v2 canary has the same 106-array topology as the baseline. Ninety-
eight array digests are byte-identical. Eight differ, all downstream of
`cv2.fitEllipse`:

| Component/output | Differing rows or elements |
|---|---:|
| left-eye ellipse parameters | 31 rows |
| left-eye ellipse success | 29 elements |
| left-eye body-axis angle | 30 elements |
| right-eye ellipse parameters | 87 rows |
| right-eye ellipse success | 75 elements |
| right-eye body-axis angle | 85 elements |
| swim-bladder ellipse parameters | 1,651 rows |
| swim-bladder ellipse success | 1,249 elements |

The source bindings, worker count, chunk size, native-thread limit, and mask
publication digests are identical. A bounded replay against the immutable
source masks found that every affected mask contains only 4--12 foreground
pixels. More decisively, two consecutive calls to the current ellipse fitter on
the same in-memory mask agreed for only 19/31 left-eye rows, 53/87 right-eye
rows, and 754/1,651 swim-bladder rows. Replay variably matched the baseline,
the optimized run, or neither. This is numerical instability in OpenCV's
ellipse fit on degenerate tiny contours, not a transfer-receipt or storage-copy
failure.

Do not choose one canary's unstable values as canonical. The scientific
hardening follow-up is to define a deterministic degenerate-mask admission
predicate before `cv2.fitEllipse`, emit an explicit component failure reason,
and test repeated-run identity. This matches the existing policy that a
component-level geometry failure is recorded without invalidating an otherwise
valid publication. Until that policy lands, this canary remains
selector-ineligible even though the receipt-v2 performance and composition
checks passed.

## Compatibility and migration

Existing consumers may require a historical flat content digest. During a
bounded migration, a publisher may carry both:

- the existing flat digest, computed by the one unavoidable compatibility
  scan; and
- the new composable write-time receipt used by updated publication and
  consumers.

Dual identity is transitional. Each publication profile adopting it needs a
written removal condition for the flat rescan and a real
writer -> finalizer -> publisher -> unpatched consumer boundary test.

Receipt profiles remain stage-specific. Subject masks, subject shape, and
keypoints can share low-level digest, chunk-ownership, and atomic-publication
machinery, but their receipts must declare their distinct scientific and
authority claims.

## Instrumentation requirements

Publication timing must separate:

- transformation and payload writing;
- write-unit receipt creation;
- receipt aggregation and sealing;
- final decoded scan, if compatibility still requires one;
- physical transfer or rename;
- metadata and authority stamping;
- metadata consolidation; and
- selector/registry finalization.

Record bytes read and written, CPU time, wall time, worker count, requested and
effective chunk ownership, receipt profile, and whether a deep scan ran. A
multi-hour scan must never be reported merely as "planning" or "publication."

## Acceptance checklist

- [ ] The writer emits sealed receipts for the exact final write units.
- [ ] Parallel write units align to whole, non-overlapping physical chunks.
- [ ] Finalization detects missing, duplicate, overlapping, reordered, or
      foreign-attempt receipts without opening payload arrays.
- [ ] Routine publication performs zero decoded payload reads after a valid
      final receipt exists.
- [ ] Same-filesystem atomic rename performs no payload rehash.
- [ ] Transfer publication either validates a bound transfer receipt or runs
      exactly one explicitly reported destination verification scan.
- [ ] Normal consumers validate receipts and published metadata generation
      without reproducing payload hashes.
- [ ] Explicit deep audit recomputes the payload and detects tampering.
- [ ] Receipt or metadata tampering fails closed.
- [ ] Counting-store and phase telemetry prove that redundant reads are gone.
- [ ] A real writer -> finalizer -> publisher -> strict unpatched consumer test
      guards each supported receipt profile.
