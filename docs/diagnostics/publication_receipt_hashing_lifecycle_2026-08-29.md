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

## Current defect demonstrated by the subject-shape canary

The current subject-shape writer does not emit composable digests for its final
parallel output. Finalization consequently rereads the completed arrays in
`build_subject_shape_unbound_payload_scan_receipt` to manufacture flat payload
hashes. Other admission paths have also reloaded payloads to reproduce claims
already sealed by upstream receipts.

That scan is compatibility work caused by a missing write-time receipt; it is
not subject-shape computation and is not the intended steady-state publication
design.

The immediate optimization may eliminate duplicate admission reads while
retaining one final scan. The durable optimization is to emit write-unit
receipts during the transformation, seal those receipts without rereading the
payload, and teach the publication/consumer contracts to accept the versioned
composable identity.

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
