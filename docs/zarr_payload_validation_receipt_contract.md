# Zarr payload integrity and validation receipts

Status: active pilot contract
Schema versions: `palette.zarr_payload_integrity_receipt` v1 and
`palette.zarr_payload_validation_receipt` v1
Initial adopter: guarded track-kinematics materialization

## Purpose

Large immutable analysis runs should not need to decode and scientifically
recompute every derived array at each publication checkpoint. They do need to
retain evidence that:

1. the exact logical payload was copied successfully;
2. the installed physical payload has not changed;
3. one identified scientific validator accepted that exact payload; and
4. completion and selector activation refer to that same validation.

These receipts provide those bindings without replacing Palette's coordinate,
lineage, ownership, completion, selector-generation, or rollback contracts.

## Evidence boundaries

Palette treats four claims separately:

| Claim | Authority |
|---|---|
| Construction | Existing stage provenance, parameters, source refs, code identity, and staging manifest |
| Integrity | `palette.zarr_payload_integrity_receipt` |
| Scientific validation | `palette.zarr_payload_validation_receipt` plus the stage scientific manifest |
| Publication | Atomic publisher owner, completion state, selector lease/generation, and final eligibility write |

An integrity receipt does not claim that values are scientifically correct.
A validation receipt does not grant selector eligibility. SHA-256 supplies
content binding, not authenticated authorship.

## Integrity receipt

The integrity receipt has three independent Merkle-style roots.

### Decoded logical root

The sharded-copy writer already reads each complete non-overlapping output
shard after writing it and records:

- array path;
- declared dtype and shape;
- row interval for row-sharded arrays;
- decoded byte count; and
- SHA-256 over contiguous C-order decoded bytes.

Leaves are grouped by array, ordered canonically, hashed into array roots, and
then hashed into a run decoded-payload root. The receipt requires a closed
array inventory and gap-free, overlap-free coverage of every row-sharded
array.

### Immutable Zarr metadata root

Each `zarr.json` document is canonicalized after removing only its
`attributes` object. The remaining node type, shape, data type, chunk grid,
chunk-key encoding, codecs, fill value, dimension names, and Zarr-format
metadata form an immutable metadata root. Verification rereads this root even
at lightweight publication checkpoints, preventing existing payload bytes from
being reinterpreted through changed array metadata.

### Physical payload root

Every immutable physical payload file beneath the run is represented by:

- relative path;
- file size; and
- SHA-256 of its bytes.

The canonically ordered file records form the physical-payload root. Hashing
may be parallelized because each worker reads a distinct immutable file.

All files named `zarr.json` are excluded from this physical-byte root because
coordinate binding, lifecycle transitions, and publisher attributes
necessarily change them. Their non-attribute contents remain protected by the
immutable metadata root. The owning stage validates attribute semantics
separately.

## Validation receipt

The validation receipt binds:

- the integrity-receipt digest;
- decoded, immutable-metadata, and physical-payload roots;
- scientific-manifest schema and digest;
- validator schema and version;
- numerical-policy record and digest; and
- a literal successful result.

For track motion, the validator remains the exhaustive full-motion manifest
builder. It checks controlled inventories, input lineage, row and time domains,
numeric invariants, physical scaling, summaries, semantic attributes, and
aliases. The receipt merely permits later guarded publication checkpoints to
refer to that completed work.

## Guarded track-kinematics flow

The v3 materializer/publisher performs:

1. exhaustive local scientific and decoded-copy validation;
2. atomic-publisher physical copy verification;
3. a physical payload root at the hidden run's renamed canonical path;
4. final-path coordinate binding;
5. a second physical-root check proving binding was metadata-only;
6. one exhaustive full-motion manifest build after completion;
7. persistence of the validation receipt;
8. lightweight completion and pointer checks against the receipt; and
9. a fresh complete physical-payload rehash immediately before the literal
   `stage_selector_eligible=true` commit.

The public `load_bound_track_motion_run` reader remains exhaustive and rebuilds
the live scientific manifest. Receipt-mode verification is private to the
guarded publisher and cannot mint normal reader authority.

## Mutation behavior

Publication fails closed when any of these occurs:

- a decoded shard is missing, duplicated, overlapping, or changed;
- a physical payload file is added, removed, resized, renamed, or changed;
- a receipt, manifest, validator identity, numerical policy, owner, or run path
  differs;
- coordinate binding changes a physical payload file;
- completion or selector state differs from the expected phase; or
- an activation callback observes a replacement run or lost selector lease.

Metadata-only changes are not accepted merely because `zarr.json` is excluded
from the physical root. Stage structural validation, manifest/commit binding,
publisher metadata comparison, owner checks, pointer checks, and selector
lease checks remain required.

## Audit and compatibility policy

- Existing direct writers retain exhaustive verification by default.
- Existing public readers retain exhaustive verification by default.
- Receipt mode is opt-in and requires both the exact receipt and canonical run
  filesystem path.
- An exhaustive audit can always use the normal public loader.
- A future shared-publisher adoption must preserve stage-specific metadata and
  scientific validation rather than treating the generic integrity receipt as
  sufficient on its own.

## Canary acceptance

Before production rollout, rerun the same long Sleepyfish recording and verify:

- identical scientific manifest digest for an equivalent run;
- receipt roots and validator identity are persisted;
- public exhaustive loading succeeds;
- hostile payload and metadata mutation tests fail closed;
- selector and rollback tests remain green; and
- completion and activation time decrease without moving cost into an
  unmeasured phase.
