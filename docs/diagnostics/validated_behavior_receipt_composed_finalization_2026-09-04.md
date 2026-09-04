# Validated-behavior receipt-composed finalization — 2026-09-04

Status: **approved implementation contract; implementation pending**.

This document is a profile-specific application of
[`publication_receipt_hashing_lifecycle_2026-08-29.md`](publication_receipt_hashing_lifecycle_2026-08-29.md)
to the existing validated-behavior cohort exporter. It is an implementation
annex to
[`validated_behavior_cohort_export_implementation_design_2026-08-31.md`](../validated_behavior_cohort_export_implementation_design_2026-08-31.md),
not a new publication architecture or a competing plan of record.

The logical publication surface remains `validated_behavior/v1`. The existing
planner, shard writer, finalizer, manifest-selected lazy reader, table profiles,
and CLI remain the public interfaces. This change versions and strengthens the
evidence passed between those components so normal finalization can compose
proofs instead of reconstructing them from payloads.

## Decision

New validated-behavior exports will use one receipt-composed finalization path:

```text
sealed plan and bundle authority
  -> recording-owned Parquet shard plus semantic receipt
  -> one destination-side transfer verification per copied part
  -> composed generation validation receipt
  -> receipt-only pre-rename gate
  -> same-filesystem generation rename
  -> manifest-last compare-and-swap
  -> receipt-mode reopen
```

Primary-key, row-owner, and foreign-key validation happens once per recording
shard, where the work is parallel. Its result is sealed against the exact
Parquet part hashes. Cohort finalization proves global closure by composing the
complete ordered shard-receipt roster. It does not decode those columns again.

A complete physical and decoded revalidation remains available only through
the existing explicit full-validation mode. Missing, unsupported, stale, or
incomplete receipt evidence blocks normal finalization. It must never trigger
an unreported exhaustive fallback.

## Observed production baseline

The audit used the completed four-camera full-frame-rate publication:

- Palette commit:
  `34cddfdfb3976b5bf91f71feddc3c8164017e937`
- operation:
  `/groups/johnson/johnsonlab/jeremy/operations/sleepyfish_validated_core_behavior_full_rate_20260904_v002`
- export run:
  `sleepyfish-validated-core-behavior-full-rate-20260904-v002`
- generation:
  `0759489a0f494fb5a5e065f83f884829`
- manifest record SHA-256:
  `1db73e5d5e8ccdd4e1fd01bb6f05eb8af6fa36c206ca66095408b6623f77d325`
- shard count: 4
- Parquet part count: 32
- Parquet bytes: 5,860,597,608
- table rows: 149,468,783

LSF job `153975834` used approximately 2,258 seconds of wall time, 2,305
CPU-seconds, and 9,958 MB maximum RSS. The timestamps separate the run into:

| Phase | Elapsed |
|---|---:|
| Source-shard receipt and payload revalidation | 604.9 s |
| Copy 5.86 GB into hidden staging | 12.2 s |
| First staged global hash, owner, key, and FK validation | 825.8 s |
| Repeated pre-rename hash, owner, key, and FK validation | 809.3 s |
| Atomic generation/manifest commit | about 0.02 s |
| Post-commit receipt-mode reopen | 6.4 s |

The payload is dominated by `tail_trace_samples`:

| Table | Rows | Bytes |
|---|---:|---:|
| `tail_trace_samples` | 114,699,250 | 4,006,005,882 |
| `eye_trace_samples` | 11,750,416 | 806,178,131 |
| `subject_body_frame_samples` | 11,469,925 | 574,393,092 |
| `kinematics_samples` | 11,469,925 | 469,874,297 |
| Other four tables | 79,267 | 4,146,206 |

Across the finalizer, the current implementation performs approximately 448
million owner-row visits, 448 million primary-key-row visits, and 299 million
foreign-key-row visits. It hashes every Parquet part three times and copies it
once. The all-at-once owner-column read also explains the nearly 10 GB RSS
peak. This is evidence reconstruction, not scientific computation.

## Evidence already present

Every current shard receipt is self-digested and binds:

- the exact export plan, membership, bundle set, member, capability matrix,
  software authority, and safety policy;
- the complete deterministic eight-table part roster and zero-row reasons;
- each part path, byte size, row count, full file SHA-256, Arrow schema
  identity, primary-key declaration, and primary-key bounds; and
- the `complete_validated` lifecycle state and the v1 validation policy.

The sealed plan and bundle set additionally bind the selected five-grain
scientific authorities. The finalizer never opens source Zarr data. Therefore
source-Zarr receipt improvements are not a prerequisite for this finalizer
change.

The existing v1 shard receipt does not explicitly serialize:

- one result record per primary-key validation;
- one exact row-owner result bound to the part hash;
- one result record per declared foreign-key edge;
- a composability declaration proving the recording-owned partition rule; or
- a destination transfer receipt for the copied part.

Those omissions are why a v1 receipt must not be silently reinterpreted as the
new no-decode proof.

## Global proof composition

`validate_table_specs()` already requires every table primary key to begin with
`(export_run_id, recording_id)` and every foreign key to carry the same
recording prefix on both sides. The current publication policy also requires
exactly one part per member and table.

For primary keys, global uniqueness follows when all of the following are
sealed and validated:

1. each part contains only its declared recording owner;
2. each part's primary key satisfies its declared exact validation policy;
3. each part's key count equals its row count;
4. the plan's `(export_run_id, recording_id)` owners are unique; and
5. the part roster contains exactly one part for every member and table.

No keys from two recording-owned parts can collide because their required key
prefixes differ.

For foreign keys, global closure follows when each shard proves every declared
recording-scoped local relation is closed against the corresponding target
part in that same shard. A local row cannot legally reference another member's
target because both sides include the recording-owner prefix.

This is global validation across all parts by exact proof composition. It does
not weaken the original requirement that validation extend beyond isolated
shards.

## Versioned evidence contracts

The implementation must evolve the existing contract family rather than add
an adapter or parallel publisher.

### Plan evidence profile

The current plan schema must grow an exact, self-digested evidence profile that
declares at least:

- required shard-receipt schema and version;
- required shard semantic-validation policy;
- required transfer-verification policy;
- required generation-composition policy;
- whether normal finalization permits payload decoding (`false`); and
- the explicit full-audit mode that may reconstruct payload evidence.

A planner from an older commit must reject this plan version. A current
finalizer must reject a plan that does not declare a supported evidence
profile. This makes accidental execution through the old path mechanically
impossible.

### Shard semantic receipt

For every part, the current part binding remains required and a v2 semantic
result binds the exact `file_sha256` and records:

- observed row count and Arrow/footer contract status;
- exact owner `(export_run_id, recording_id)` and owner-validation status;
- primary-key fields, validation mode, count, bounds, and completion status;
- required-field/null validation status;
- every declared foreign-key local/target table and field roster;
- the local and target part hashes used for each FK proof;
- local row count, target distinct-key count, unmatched count, and closure
  status; and
- the proof implementation/method identifier.

The receipt is an attestation by the pinned trusted writer over exact
content-addressed files. A second key-sequence digest is not required for the
current one-part-per-recording layout because the full file digest already
binds the bytes and the receipt binds validation of those bytes. If a future
layout permits multiple parts for one recording, it must add adjacent
first/last-key ordering proofs or an exact external key index before using the
receipt-composed path.

The first implementation may retain one exhaustive post-write semantic check
inside each shard job. That work is recording-local, parallel, and performed
once. A later optimization may generate the same evidence while Arrow batches
are resident, following the subject-mask write-unit receipt pattern. It must
not be required to unblock finalizer composition.

### Transfer receipt

For each source part, finalization must:

1. reject symlinks and non-regular source or destination paths;
2. copy to the unique hidden staging generation;
3. hash the destination exactly once;
4. require destination size and SHA-256 to equal the sealed source-part
   receipt; and
5. seal source shard receipt, source part binding, destination relative path,
   observed size/hash, transfer method, and staging-attempt identity into a
   transfer record.

The destination verification is the one legitimate payload scan in normal
finalization. A pure same-filesystem rename would not need it; this workflow
performs a copy and therefore does.

### Generation validation receipt

The v2 generation receipt composes:

- the exact plan/membership/bundle/profile identities;
- the ordered shard-receipt roster and its digest;
- the ordered transfer-receipt roster and its digest;
- the exact destination part inventory and its digest;
- summed row counts and part counts by table;
- complete composed owner/primary-key/FK proof status;
- the staging-attempt identity and mutation-exclusion policy;
- software authority and validation timestamp; and
- unchanged selector-ineligible safety policy.

It must distinguish `complete_receipt_composed_v2` from a fresh decoded scan.
It may not report the existing generic value `complete` in a way that conceals
which proof path ran.

## Finalization algorithm

Normal finalization must perform these steps in order:

1. Read and validate the plan, membership, and bundle set once into an
   operation-local immutable context.
2. Validate the exact v2 shard receipt roster without opening Parquet payload
   columns or rehashing source parts.
3. Copy each part into a fresh unpredictable staging generation and perform
   one destination hash comparison against the source receipt.
4. Copy and validate the small shard receipts.
5. Compose global row counts, owner closure, primary-key closure, and FK
   closure from the exact shard proofs.
6. Verify destination Parquet footer/schema metadata and the exact closed file
   inventory without decoded column traversal.
7. Seal the transfer and generation validation receipts.
8. Freeze the staged part files against cooperative accidental writes and
   record the staging ownership/mutation-exclusion policy.
9. In the pre-rename callback, revalidate only small receipt seals, plan/input
   bindings, exact path/stat inventory, staging ownership, and expected
   manifest identity. Do not rehash or decode a part.
10. Preserve the existing atomic staging-to-generation rename and manifest-last
    compare-and-swap.
11. Keep the existing receipt-mode post-commit reopen. It costs about six
    seconds and verifies the consumer boundary that colleagues will use.

The generation and manifest remain selector-ineligible and do not update the
registry or mutate source Zarr arrays.

## Compatibility and failure behavior

- Existing v1 export manifests, validation receipts, shard receipts, and
  generations remain readable and auditable without mutation.
- New plans and shard outputs use the versioned evidence profile. The normal
  current finalizer accepts only that complete profile.
- A v1 shard, absent FK proof, missing transfer evidence, unknown proof method,
  mismatched digest, stale plan, or changed inventory returns a typed blocking
  result before manifest publication.
- Normal finalization never falls back to an exhaustive scan.
- The existing explicit `validate --full-part-hashes` path remains the deep
  audit for published generations. It rehashes payloads and validates decoded
  owners, keys, and FKs.
- Historical v1 publication through old shard evidence is not required for
  normal production. If an operator later needs it, it must be an explicitly
  selected recovery/deep-audit operation, never an implicit branch.
- No existing publication, source shard, selector, or registry row is rewritten
  or deleted.

## Threat model and immutability boundary

The normal path protects against accidental source mutation, short/torn copy,
wrong-file copy, stale receipt, missing/extra file, cooperative concurrent
writer, and manifest race. Destination hashing catches source mutation during
copy because the copied bytes must match the previously sealed source digest.

Receipt mode, like the existing Palette receipt system, assumes trusted
producer code and trusted storage ownership. Self-digested SHA-256 records do
not authenticate an adversary who can rewrite both payload and receipts.
Generation files currently use group-writable modes, so the implementation
must add an explicit cooperative immutability/freeze policy before relying on
receipt-only reopen. That policy and its tests are part of this change, not an
optional cleanup.

Power-loss durability and cryptographic issuer authentication are separate
system-wide concerns. This change must not claim to add them silently.

## Full-audit implementation hygiene

The explicit deep audit remains exact, but it should also be bounded and avoid
known needless work:

- fuse owner validation into the primary-key traversal because owner fields are
  the mandatory first two key fields;
- stream batches rather than calling `ParquetFile.read(...).to_pylist()` over
  an entire large part;
- use the structural recording-prefix proof for FKs that contain no fields
  beyond `(export_run_id, recording_id)`;
- retain exact relation scans for FKs with additional fields; and
- record full-audit telemetry without putting runtime data into scientific
  identity.

These changes reduce memory and audit cost but do not substitute for the v2
normal path.

## Upstream source-receipt follow-up

The five-grain bundle has complete scientific authority, but uniform physical
payload-receipt coverage is not yet complete:

- subject-shape and tail publications are receipt-backed;
- kinematics retains detailed per-surface digests but still performs some
  authority-only reads during shard extraction;
- eye trace has no complete dense source-payload receipt;
- canonical bouts reconstruct selected event/frame-axis digests; and
- the tail-to-track identity index is reconstructed during extraction.

These costs occur in the parallel shard writer, not the cohort finalizer. They
are explicitly deferred so this change remains a narrow repair of the
37-minute serial fan-in.

## Acceptance gates

The implementation is accepted only when all of the following hold:

- real writer -> v2 shard receipt -> receipt-composed publisher -> unpatched
  lazy reader succeeds;
- every existing v1 publication fixture still opens through the compatibility
  reader;
- normal publication performs zero decoded Parquet owner/PK/FK scans;
- every destination part is physically hashed exactly once after copy;
- missing or invalid v2 evidence blocks without invoking the full validator;
- source, staging, receipt, plan, membership, bundle, and manifest mutations
  each fail closed at their contracted boundary;
- published row counts, part hashes, table contracts, authority bindings, and
  safety flags match a full audit;
- the deep-audit reader independently validates the v2 publication;
- peak RSS is bounded well below the observed 9,958 MB;
- the four-camera 5.86 GB canary finalizes in at most five minutes, with a
  design target of one to three minutes; and
- every required CI check completes successfully before merge or production
  use.

Wall-clock assertions do not belong in ordinary CI. Unit tests count payload
hashes, decoded column reads, receipt validations, and fallback invocations.
The production-sized canary records wall/CPU/RSS/I/O telemetry separately.

## Implementation checklist

### Phase 0 — Contract and baseline

- [x] Record the exact production generation, sizes, counts, and phase timing.
- [x] Inventory the current shard and upstream evidence.
- [x] Define normal receipt composition, explicit deep audit, and fail-closed
      compatibility.
- [ ] Add cross-links from the governing lifecycle and cohort design docs.

### Phase 1 — Execution evidence versions

- [ ] Version the plan evidence profile so old execution code rejects it.
- [ ] Version the existing shard receipt family with explicit owner, PK, FK,
      and composability results.
- [ ] Version the generation validation receipt and outer export envelope as
      needed for honest reader dispatch.
- [ ] Keep logical table contracts, profile IDs, and the
      `validated_behavior/v1` storage surface unchanged.
- [ ] Add v1/v2 reader dispatch and byte-for-byte v1 compatibility tests.

### Phase 2 — Shard proof production

- [ ] Produce one exact semantic proof per part and FK edge.
- [ ] Bind every proof to exact source part hashes and table-spec identities.
- [ ] Ensure the writer performs at most one post-write semantic validation.
- [ ] Seal only after all tables and FK targets are present.
- [ ] Reject missing/extra proof records and unknown proof methods.

### Phase 3 — Receipt-bound transfer and composition

- [ ] Cache one validated plan/membership/bundle context per finalizer.
- [ ] Copy each part and verify the destination digest once.
- [ ] Emit an exact transfer receipt for all 32 part copies.
- [ ] Compose row counts, owner closure, PK uniqueness, and FK closure without
      payload decoding.
- [ ] Freeze and validate staging ownership before commit.
- [ ] Replace repeated `_global_validate_generation()` calls in normal
      finalization with receipt/inventory validation.
- [ ] Preserve atomic generation rename, manifest-last CAS, and receipt-mode
      post-commit reopen.

### Phase 4 — Deep audit and tests

- [ ] Make full owner validation bounded and fuse it with PK traversal.
- [ ] Add the recording-prefix-only FK structural fast path.
- [ ] Add hash/read/fallback call-count tests.
- [ ] Add adversarial source, copy, staging, receipt, plan, and manifest tests.
- [ ] Add writer-to-unpatched-reader and v1 archive compatibility boundaries.
- [ ] Run focused tests outside the Codex sandbox with `scripts/py`.

### Phase 5 — Evidence and integration

- [ ] Run a small deterministic receipt-composition smoke.
- [ ] Run the four-camera production-sized selector-ineligible canary from a
      clean commit-pinned deployment.
- [ ] Compare the canary with manifest counts and an explicit full audit.
- [ ] Record phase telemetry and verify the five-minute acceptance gate.
- [ ] Require all CI checks to pass before merge or production use.

## Code integration map

Primary implementation locations:

- `src/fisheye/analytics_exports/validated_behavior_cohort.py`
  - plan, shard, transfer, generation receipt validation;
  - shard proof production;
  - proof composition;
  - v1/v2 manifest reading;
  - deep-audit traversal.
- `src/fisheye/analytics_exports/publication.py`
  - shared immutable generation commit and staging ownership boundary.
- `src/fisheye/utils/materialize_validated_behavior_cohort_export.py`
  - unchanged public commands with current-plan enforcement and concise
    process telemetry.
- `scripts/submit_validated_behavior_cohort_export_bsub.sh`
  - retain the post-commit receipt-mode validation while the implementation is
    canaried.
- `tests/unit/fisheye/test_validated_behavior_cohort_export.py`
  - receipt versions, proof composition, mutation, no-decode, compatibility,
    and real writer/publisher/reader coverage.

Reusable precedents rather than new adapters:

- `subject_mask_final_layout_units.py` for writer-unit proof composition;
- `subject_shape_storage.py` for staged-transfer plus appended-proof
  composition;
- `zarr_payload_receipt.py` for receipt-only normal validation and explicit
  deep audit; and
- `commit_validated_immutable_generation()` for the existing atomic visibility
  boundary.
