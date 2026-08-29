# Publication receipt and hashing lifecycle — 2026-08-29

Status: **design decision and optimization contract**. This governs new work;
existing publication families may require a compatibility migration before
their redundant payload scans can be removed.

Observed against Palette commit:
`bf058521` (`subject-shape-unbound-receipt-20260827-bf058521`).

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
`agent/palette/subject-shape-plan-receipts-20260829`; this document is part of
the implementation checkpoint, which is not yet pushed, CI-green, merged, or
production-active**.

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
- the final access-aware regression passed in 181.34 seconds; and
- that regression makes both
  `build_subject_shape_unbound_payload_scan_receipt` and
  `_scan_subject_shape_bound_payload` raise if the v2 path attempts either
  complete decoded rescan.

The commit-`bf058521` canary cannot benefit from code written after submission.
Its resource trace changed character at approximately 19:27:23 UTC, after
8,176.78 seconds and 626.40 GB of cumulative delivered reads: thread count and
writes began rising, which is consistent with leaving the serial unbound scan
and entering parallel storage conversion. By 19:33:23 UTC conversion had added
only about 2.73 GB of reads and 1.35 GB of writes. At 19:33:53 UTC a short
multi-process copy/physical phase began, followed by another read-heavy,
write-flat phase consistent with the old post-binding decoded scan. These phase
labels are inferences from resource transitions because the old process buffers
its stage log. The job remains diagnostic baseline evidence, not a benchmark of
this v2 implementation.

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
