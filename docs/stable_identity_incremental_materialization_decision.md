# Stable Identity And Incremental Materialization Decision

<!-- decision-meta
status: accepted-design
created: 2026-07-10
last_updated: 2026-07-18
owner: jeremy
scope: observation identity, downstream invalidation, copy-forward materialization,
  and asynchronous reconciliation
depends_on: docs/instance_track_subject_identity_contract.md,
  docs/refined_detect_row_identity_contract.md,
  docs/mutable_review_runs_contract.md
refines-staging-in: docs/sleap_palette_storage_assessment.md,
  docs/manual_add_row_propagation_design.md
-->

## Decision Summary

Palette will keep recording Zarrs as the canonical provenance unit and will use
stable sparse observation identity to connect detection, crop, keypoint, mask,
arena-assignment, and tracking artifacts.

A changed observation rowset must eventually produce complete downstream runs,
but **complete output does not mean full recomputation**. The default
materialization policy for large recordings is:

1. create a new downstream run bound to the exact new source rowset;
2. reuse or copy forward payload rows whose identity, source content, model,
   method, and relevant configuration are provably compatible;
3. compute model predictions only for new or invalidated observation keys;
4. preserve reviewed human-authored values for unchanged logical rows;
5. rebuild inexpensive run-wide indexes, summaries, and derived caches as
   necessary;
6. validate complete coverage and exact keyed lineage before publication;
7. publish the new run atomically, leaving the old run immutable.

Full model inference across every observation in a recording is an exceptional
fallback, not the normal response to adding, deleting, or editing one
observation. It is required only when compatibility cannot be proven or when a
change genuinely invalidates the whole stage, such as a new model checkpoint,
incompatible preprocessing contract, or changed source pixels.

Where earlier staging documents recommend "full downstream regeneration" or
"whole-run regeneration," interpret that as producing a complete replacement
artifact. This document controls the computation policy: compatible unchanged
rows are reused, and ordinary localized edits do not authorize all-row model
inference.

Palette will not initially make an intentionally incomplete derived run the
active canonical run. Pending work belongs to an explicit materialization plan
or staging run. The active run changes only after the replacement artifact is
complete and validated.

## Relationship To Live-Edit Staging

This decision governs downstream reconciliation and publication after an
authoring change has been durably committed. It does not replace the browser
checkpoint, edit-commit, conflict, or apply boundaries in
`docs/mutable_review_runs_contract.md`.

The two contracts compose as follows:

```text
Crimson checkpoint or autosave
  -> durable session overlay, not canonical and not materialization input

explicit commit/apply
  -> validated authoring revision with stable logical row identity

incremental materialization
  -> copy compatible rows, compute invalidated rows, validate complete output

atomic publication
  -> select the new complete versioned run
```

Unapplied checkpoints must never trigger downstream materialization. Browser or
application exit is not a reliable commit boundary. Exit may request an apply
or enqueue already committed work as a convenience, but durability must come
from an explicit server-owned checkpoint or commit transaction.

The materializer must bind to one exact committed authoring revision and a
closed range of edit events. Edits committed after that boundary belong to a
later materialization. This keeps session recovery, multi-user conflict
handling, and downstream provenance independent of job timing.

## Why This Decision Exists

Palette recordings contain tens of thousands of frames. Re-running crop,
keypoint inference, subject-mask inference, refinement, and finalization across
an entire recording because one observation was added would be unnecessarily
expensive. It would also risk overwriting reviewed corrections and would make
interactive curation operationally impractical.

At the same time, simply appending a row independently to every existing run is
unsafe. Current payload families contain row-aligned dense arrays, ragged
indexes, metrics, provenance, and derived caches. A partially extended run can
look valid to a reader that only checks one array. Publication must therefore
remain atomic and fail closed.

The decision combines the useful parts of both approaches:

- stable, instance-oriented sparse identity;
- efficient dense/chunked Zarr payloads;
- delta computation for expensive inference;
- complete, immutable published artifacts;
- exact source binding and reproducible provenance.

## Terms

### Complete published run

A run in which every required source observation has an explicit, valid
materialization state and every required array/index satisfies the stage
contract. Depending on the stage, a row may contain a successful payload, a
contract-defined failed result, or another explicit terminal state. It must not
be silently absent.

### Full recomputation

Running a stage's expensive computation again for every source observation,
even when most inputs and outputs are unchanged. This is not the default.

### Copy-forward delta materialization

Building a new complete run by copying verified compatible rows from an exact
prior run and computing only new or invalidated rows. This is the default.

"Copy-forward" describes logical reuse. It does not require duplicating every
unchanged payload byte when the storage layer can safely reference immutable
base chunks, shards, or rows.

### Partial active materialization

Publishing a canonical run while some source observations are still pending and
do not yet have complete downstream payloads. This remains deferred until all
readers, validators, review surfaces, and dispatchers understand the pending
state.

### Reconciler

A process that converts explicit stale or pending state into an exact-run
materialization plan, submits the required work, validates completion, and
publishes the replacement run. It may be a periodic command or cluster workflow;
it does not need to be a continuously running daemon.

## Identity Boundaries

These namespaces must remain separate:

| Identifier | Meaning | Stability rule |
| --- | --- | --- |
| `instance_key` | One observation originating at detection or manual addition | Mint once at origin and copy thereafter. A bbox edit to the same logical observation must not change it. |
| `refined_row_id` | One logical curated row inside a refined-detect run | Preserve across ordinary edits and physical rewrites; never reuse after deletion. |
| `track_id` | One run-local temporal trajectory | Scoped to one exact tracking run and may change when association is revised. |
| `subject_id` | Optional known biological identity | Assigned through reviewed biological metadata, not inferred silently from tracking. |

Globally addressable forms remain:

```text
observation       (recording_id, instance_key)
curated row       (recording_id, refined_detect_run, refined_row_id)
trajectory        (recording_id, tracking_run, track_id)
biological animal subject_id
```

An `instance_key` is not an appearance fingerprint and is not a promise that
two observations in different frames depict the same animal. Appearance or
re-identification features may later provide evidence for track or subject
assignment, but they do not replace observation identity.

## Manual-Origin Key Preservation

The stable-key contract applies to manually added rows as strongly as it applies
to detector-origin rows.

- A new manual observation mints one namespaced `instance_key`.
- Ordinary bbox, class, note, status, or review edits to that surviving logical
  row preserve its existing `instance_key`.
- Physical row reordering preserves the key.
- Copying the observation downstream preserves the key verbatim.
- Deleting an observation tombstones its logical identity; its key is not reused.
- Splitting one logical observation into multiple observations tombstones the
  old observation and mints new keys for the new observations.
- Merging multiple logical observations tombstones the inputs and mints a new
  key for the merged observation.

The curation writer preserves keys for surviving `refined_row_id` values and
uses a monotonic, non-reusing row-ID allocator. New manual-origin keys include
their assigned row ID in the minting namespace, so delete-and-recreate, split,
and merge operations cannot reuse a retired observation key. Supplying a
negative row ID explicitly requests a fresh identity; explicitly reusing a
retired non-negative ID fails closed.

## Canonical Storage And Authority

Palette will borrow SLEAP's useful separation between sparse instance identity
and dense payload storage without adopting a mutable multi-recording project
file as the source of truth.

```text
recording Zarr
  detect/refined detect       sparse observation identity and authoring
  crop/keypoint/mask runs     dense or ragged materialized payloads
  arena_assignment_runs       spatial assignment by observation key
  tracking_runs               temporal assignment by observation key
  analysis runs               derived products bound to exact sources

registry                      rebuildable discovery/freshness projection
collection manifest           exact cross-recording project/cohort selection
exports                       rebuildable interchange/analytics products
```

Raw model outputs remain immutable. Refined surfaces remain the reviewed
authoring layer. Derived runs remain replaceable products bound to exact source
runs, revisions, model/config fingerprints, and rowset fingerprints.

## Complete Output Without Full Inference

Suppose an existing source rowset contains keys `A, B, C, D` and an operator
adds key `E`.

The target keypoint materialization is:

```text
new complete keypoint run
  A   copied from compatible prior keypoint run
  B   copied from compatible prior keypoint run
  C   copied from compatible prior keypoint run
  D   copied from compatible prior keypoint run
  E   newly cropped and predicted
```

The output covers the complete new keyset, but only `E` requires expensive
model inference.

If `C` also received a bbox edit:

```text
A, B, D   copy forward
C, E      regenerate crop and geometry-dependent predictions
```

If `B` was deleted, it is omitted from the new run. The old run remains intact
as provenance; it is not resized or rewritten.

## Logical Completeness Without Physical Duplication

A complete run means that every target observation resolves to a valid payload;
it does not require each run to own a second physical copy of every unchanged
dense crop or mask.

The implementation may use one of these physical strategies:

1. **Physical row copy:** write unchanged rows into new arrays. This is the
   simplest compatibility path and may be adequate for compact keypoint tables.
2. **Aligned chunk or shard reuse:** reuse immutable storage objects when all
   rows in the physical chunk are compatible and the backend supports safe
   sharing or copy-on-write behavior. Byte-level reuse additionally requires
   identical array metadata, dtype, dimension order, shape interpretation,
   chunk grid, codec pipeline and parameters, fill values, and target
   row-to-chunk placement. If any of those conditions cannot be proven, decode
   and copy logical rows or use an explicit base-plus-delta resolver instead.
3. **Immutable base plus delta:** publish a complete logical run whose manifest
   maps unchanged keys to an exact immutable base run and changed keys to new
   delta payloads. A resolver presents the result as one complete keyed rowset.

The third strategy is not the same as partial materialization. Every target key
already has a terminal payload; some payloads are resolved from an immutable
base rather than duplicated into the new run.

Composite runs require explicit contracts:

- exact base-run and base-row references;
- acyclic reference graphs with a bounded resolution depth;
- base-run retention while references exist;
- keyset and source-signature validation across base and delta;
- resolver support in every canonical consumer;
- export/materialization tooling for consumers that require standalone arrays;
- registry and inventory support that does not mistake referenced payloads for
  missing data.

Filesystem hard links must not be used as an informal substitute for this
contract. They can couple supposedly immutable runs to later mutation and make
retention or garbage collection unsafe. Storage-level sharing is allowed only
when immutability and lifecycle semantics are explicit and validated.

The first implementation may use physical copying to establish semantics, but
crop and dense-mask prototypes must measure bytes read/written and storage
amplification. If copying unchanged payloads remains expensive, immutable
base-plus-delta composition should be implemented before active partial-row
publication.

## Row Reuse Compatibility

Matching `instance_key` values are necessary but not sufficient to copy a
payload row. Reuse must fail closed unless every input that affects that stage
is compatible.

At minimum, a reusable row must match on:

- `instance_key`;
- source recording and exact source-run family;
- source row content or relevant row/component revision;
- bbox/crop geometry and pixel-coordinate contract for geometry-dependent
  stages;
- source video/pixel identity when pixels are consumed;
- model artifact fingerprint;
- inference method and method version;
- configuration hash and relevant normalized parameters;
- component/skeleton/schema version;
- any upstream payload identity used by the stage, such as the exact keypoint
  row used to assign left/right eyes.

The rowset fingerprint is a membership and revision gate. It is deliberately
order independent and does not, by itself, prove that a key is still aligned to
the same bbox, pixels, or upstream component payload. Incremental reuse therefore
also needs a per-row source signature or an equivalent trustworthy revision
contract.

The implementation should define one shared reuse decision record rather than
allowing each writer to invent incompatible checks. A target record should be
able to explain:

```text
instance_key
action = copy | compute | omit | preserve_manual
reason_code
source_run, when copied
source_row_index, when copied
source_row_signature
target_row_index
```

### Per-row source signature and revision contract

The canonical comparison primitive is implemented in
`fisheye.shared.row_source_signature`. A materializer computes
`source_row_signature` as a `uint8[N, 32]` SHA-256 array, in bounded row
batches, and stores the full versioned signature specification in attrs.
Signatures are addressed and compared by `instance_key`; physical row order is
not part of their meaning.

Version 1 hashes:

```text
schema/version and signature-spec digest
instance_key
stage-relevant fixed-width numeric content components
stage-relevant nonnegative row-revision components
```

The signature-spec digest covers the stage name, component names, component
basis (`content` or `revision`), normalized dtype/trailing shape, the named
producer authority for every revision component, and a strict JSON
compatibility context. The context is where a caller binds global inputs
such as the source-pixel fingerprint, coordinate contract, skeleton/component
schema, preprocessing version, or another upstream signature contract. A
context change invalidates the complete signature set. Model artifact,
configuration, method, and exact source/reuse-run references must also remain
explicit in the materialization plan even when duplicated in this context.

Content components are canonicalized to little-endian C-order bytes. Float NaN
and signed-zero representations are normalized. Revision components are
canonical nonnegative `uint64` values so an `int32` versus `int64` storage
choice cannot create false invalidation. Strings, objects, and other ambiguous
variable-width values are not accepted as row components; callers must encode
them through a versioned enum or a separately defined digest.

A row revision may substitute for source content only when its named authority
guarantees that every accepted change relevant to the consuming stage
increments that revision atomically with the edit. Refined subject-mask
component `row_revision` values satisfy this model for dense-authority edits.
Where no such producer guarantee exists, the actual relevant content must be
hashed. A run-wide `edit_revision` remains a publication/concurrency gate; it
does not replace per-row signatures because it cannot identify which rows are
reusable.

The digest intentionally does not infer the relevant inputs for a stage. The
stage adapter declares them. Initial adapters should use:

| Target stage | Required per-row basis | Required global compatibility context |
| --- | --- | --- |
| Crop | frame index and bbox/crop geometry content | source video/pixel identity and coordinate/crop contract |
| Raw keypoints | exact crop-row signature | pose model, preprocessing, ROI shape, and skeleton schema |
| Raw subject masks | exact crop-row signature | mask model, preprocessing, ROI shape, and component schema |
| Refined keypoints | raw-keypoint signature plus accepted manual row revision/content | refinement method/config and skeleton schema |
| Refined subject masks | raw-mask signature plus per-component accepted dense row revisions | refinement/finalization method/config and component schema |

The batch API produces the same signatures whether called for the whole rowset
or one storage shard at a time. Materializers therefore must not load an entire
recording merely to compute signatures. They should read, compare, and write
one bounded source/output shard at a time while retaining a compact keyed plan.

## Change And Invalidation Matrix

| Change | Expensive computation required | Complete replacement behavior |
| --- | --- | --- |
| Add one observation | New/affected rows only | Copy compatible rows; compute the addition; rebuild indexes/summaries. |
| Delete one observation | Usually none for unchanged rows | Omit the deleted key; copy survivors; rebuild indexes/summaries. |
| Reorder physical rows | None | Keyed reorder/copy only; rowset fingerprint remains membership-equivalent. |
| Edit bbox or crop geometry | Edited observation and geometry-dependent descendants | Preserve identity; regenerate crop, keypoints, masks, and affected metrics for that key. |
| Edit review note or non-computational status | None unless policy changes usability | Copy payload; update authoring metadata and dependent summaries. |
| Manually edit keypoints | No keypoint inference for the accepted edit | Preserve reviewed keypoints; invalidate only consumers that use them. |
| Manually edit dense subject mask | No mask inference for the accepted edit | Preserve authoritative dense edit; rebuild derived mask caches/metrics as needed. |
| Change arena definitions | No vision inference | Recompute spatial assignment; tracking/kinematics may need new runs. |
| Change tracking method or parameters | No detection/keypoint/mask inference | Re-associate the complete observation sequence in a new tracking run. |
| Change model checkpoint or incompatible inference configuration | All rows claimed by that model stage | Full stage inference is expected; downstream stages are reconsidered separately. |
| Change source pixels or incompatible pixel contract | All pixel-dependent rows | Full affected-stage recomputation is expected. |
| Change schema with a lossless migration | Prefer transformation/copy | Validate migrated complete run; do not invoke models without need. |

## Stage-Specific Policy

### Detection and refined detection

Raw detector outputs are not rerun or rewritten merely because an operator adds
or edits a curated observation. The refined authoring surface creates a new
revision or replacement run containing the complete curated rowset.

Surviving rows retain `refined_row_id` and `instance_key`. New rows receive new
identities. Deleted rows are tombstoned or omitted according to the refined
identity contract without allowing identifier reuse.

### Crop

Crop payloads may be copied only when the observation key, bbox geometry, source
pixels, crop parameters, padding behavior, and pixel contract are compatible.
New or geometry-changed observations regenerate their crop rows.

The output crop run covers the complete target observation rowset and records
which rows were copied versus computed.

A committed detection edit records invalidation but does not, by itself,
require eager crop-pixel persistence. When a crop-dependent materialization is
requested, one crop-delta preparation step durably writes and validates the
affected ROI pixels before keypoint and subject-mask inference fan out. Those
branches may then run concurrently against the same exact pixels and retry
independently. Preview-only crops may remain transient; pixels cited by a
completed downstream run must come from an exact durable crop run. See
`docs/composite_crop_storage_contract.md` for lifecycle and retention details.

### Keypoints

Raw keypoint predictions are copied for compatible unchanged crops and computed
only for new or invalidated crops. A model/config change invalidates all rows for
that keypoint prediction stage.

Reviewed keypoint corrections are authoring data, not disposable model output.
They are preserved by `instance_key` and stable refined-row identity when their
source geometry remains compatible. New or changed predictions may require
review, but unchanged accepted rows must not be reset to unreviewed solely
because another observation was added.

Run-wide success statistics and indexes may be recalculated across all rows.
That inexpensive scan does not justify repeating GPU inference.

### Subject masks

For modern refined subject masks, dense `masks_roi` remains the authoritative
pixel surface. Compatible unchanged dense rows, including reviewed manual edits,
are copied forward. New or invalidated rows run inference/refinement as needed.

Bitpacked masks, RLE, contours, and metrics are derived products. They may be
copied when independently proven compatible, recomputed only for changed rows,
or rebuilt run-wide when their global index format makes that simpler. An O(N)
deterministic cache/index rebuild is acceptable when it avoids O(N) model
inference and preserves correctness.

Finalization validates a complete dense authority before refreshing or
publishing derived caches. It must not overwrite accepted dense edits with new
model output.

### Arena assignment and tracking

Spatial arena assignment is cheap enough to recompute over the complete rowset,
although keyed delta reuse is allowed. Recomputing arena assignment does not
require rerunning vision models.

A real multi-subject tracker may need to process a complete temporal sequence
after an observation change because one association can affect later track
assignments. That is a full tracking pass, not a full detection/keypoint/mask
inference pass. Tracking remains a separate derived layer over existing
observations and features.

### Track kinematics and later analysis

Analyses should invalidate at the narrowest trustworthy scope. Adding or
changing one observation may require rebuilding the affected track, temporal
window, or run-wide index. It does not automatically invalidate independent
tracks or upstream vision payloads.

Where a current writer cannot safely rebuild a subset, it may rebuild the
complete analysis run from already materialized upstream arrays. This is still
different from repeating upstream model inference.

## Materialization Workflow

Each incremental replacement follows the same state machine.

### 1. Resolve exact inputs

Resolve concrete source run names and revisions before planning. Do not leave
`latest` to be resolved by a later worker.

### 2. Snapshot source identity

Record the target source rowset path, row count, edit revision, sorted-key
digest, and rowset fingerprint. Reject duplicate or missing modern keys.

### 3. Select an exact reuse candidate

Choose a complete prior run with compatible model, method, schema, configuration,
and upstream lineage. Never copy from a merely convenient latest run.

### 4. Build the keyed delta plan

Classify every target key as copy, compute, or preserve-manual. Classify keys
that exist only in the prior run as omitted. Record reason codes and expected
counts.

### 5. Write a new staging run

Allocate the complete target arrays in a new run or temporary group. Do not
resize the active run in place. Copy and compute work must obey physical Zarr
chunk ownership rules; parallel workers may not write overlapping physical
chunks.

### 6. Rebuild required indexes and derived caches

Rebuild frame indexes, ragged offsets, summaries, metrics, contours, RLE, or
other derived structures according to the stage contract. Prefer delta work,
but permit deterministic full scans when they are materially cheaper than model
inference and simplify correctness.

### 7. Re-read and validate

Before publication:

- re-read the current source fingerprint and revision;
- verify it has not changed during processing;
- verify exact keyset equality and uniqueness;
- verify every required array covers the complete target rowset;
- verify copied-row compatibility and computed-row completion;
- verify no stale or pending row is presented as a valid empty payload;
- verify provenance and materialization counts;
- run stage-specific pixel, geometry, component, and review-state checks.

### 8. Publish atomically

Mark the new run complete and update the authoritative/latest pointer only after
validation. Failure leaves the previous complete run selected and preserves the
failed staging run for diagnosis according to normal failed-run policy.

Publication must be conditional on the exact source revision and fingerprint
captured by the plan. After validation, the publisher must re-read that source
state and use a compare-and-swap-equivalent pointer update that includes the
expected prior selection or publication generation. If either the source state
or expected publication generation changed, publication fails closed and the
new staging run remains unselected. This prevents an older or slower job from
replacing output produced for a newer committed edit revision.

The recording Zarr is the authoritative publication surface; the registry is a
rebuildable projection. Publication must not rely on a cross-system transaction
between Zarr and the registry. Registry refresh follows successful authoritative
publication and must be safe to retry.

## Provenance Requirements

Every copy-forward materialization should record, at minimum:

- exact target source run/path/revision/fingerprint;
- exact reuse source run;
- model artifact fingerprint, method version, and configuration hash;
- total target rows;
- copied row count;
- newly computed row count;
- preserved manual row count;
- omitted/deleted prior row count;
- invalidation reason counts;
- materialization planner/schema version;
- whether any run-wide caches or indexes were rebuilt;
- validation result and publication timestamp.

Per-row origin should be queryable through an enum/code array or a compact keyed
table when that information is needed for audit and debugging. The provenance
must make it possible to answer why a row was copied or recomputed.

## Stale State And Reconciliation

An observation-rowset change immediately marks dependent runs stale; it does
not immediately make an incomplete replacement authoritative.

The stale event should include:

- exact old and new source references;
- added, deleted, and changed `instance_key` values or a durable keyed delta
  artifact when the list is large;
- source revision and rowset fingerprints;
- invalidation reasons;
- affected downstream stage families.

The reconciler consumes that state, creates the delta plan, submits exact-run
work, and publishes only after validation. Its first implementation may process
one recording and one stage family at a time. Cluster parallelism is an
execution detail and must preserve non-overlapping Zarr chunk ownership.

Pending states should be visible to operators and registry/status views, but an
old complete run should be labeled stale rather than silently disappearing
while its replacement is being built.

## Relationship To Partial Per-Row Materialization

Copy-forward delta materialization should be implemented before allowing active
partially materialized runs. It captures most of the computational savings:
only new or invalidated observations require expensive inference.

Active partial materialization would additionally require all consumers to
understand states such as:

```text
materialized
pending_generation
failed
rejected
stale
```

It would also require safe append/index updates, review UI support, registry
projection, retry semantics, training-export policy, and reader behavior for
missing payloads. That complexity is justified only if atomically publishing a
complete copy-forward replacement remains too slow or too storage-intensive in
practice.

## Rollout Plan: Phase 0 Prerequisite Plus Six Implementation Phases

### Phase 0: close identity correctness gaps

- [x] Preserve existing manual-origin `instance_key` values across ordinary edits.
- [x] Add regression tests for bbox edits, reorder, delete, split, and merge.
- [x] Reject one-sided `instance_key` loss instead of silently falling back to
  positional lineage comparison; label two-keyless compatibility as
  `legacy_positional`.
- [x] Make refined-detect, refined-keypoint, and refined-subject-mask validation
  enforce explicit modern key presence and uniqueness while reporting historical
  keyless runs as legacy compatibility.
- [x] Make clipped production validation compare exact ordered keys for cache
  row indexes and raw mask shards as well as merged crops, keypoints, and refined
  masks.
- [x] Define and implement the versioned, bounded-batch per-row source
  signature/revision contract used for reuse.

### Phase 1: shared planner and crop proof

- [x] Implement a shared keyed delta planner.
- [x] Materialize a complete crop run by copying unchanged crops and computing only
  new/geometry-changed rows.
- [x] Record row-level action/reason provenance.
- [x] Prove atomic failure behavior and exact-rowset validation.

#### Phase 1 implementation evidence

Phase 1 is implemented by:

- `fisheye.shared.keyed_delta`, the stage-neutral NumPy planner;
- `fisheye.tracking.incremental_crop`, the first complete-run adapter;
- `fisheye.utils.materialize_incremental_crop`, the dry-run-first operator CLI;
- `tools/benchmark_incremental_crop.py`, the deterministic I/O benchmark.

The Phase-1 physical follow-up is also implemented for crops as an
unselected-by-default canary. `fisheye.shared.composite_crop` defines a
depth-one immutable base-plus-delta schema and resolver, and
`materialize_composite_incremental_crop_run` writes only computed ROI rows.
The complete schema, reader boundary, selection behavior, retention guard, and
future compaction rule are in `docs/composite_crop_storage_contract.md`.

The planner does not construct a Python dictionary containing one tuple per
observation. It keeps compact NumPy key/signature/action arrays and matches
keys with stable sort plus `searchsorted`. Its persisted
`materialization_plan` is a columnar Zarr v3 group with 131,072 requested outer
rows per indexed shard. The row columns are:

```text
instance_key
target_row_index
source_row_index
action_codes
reason_codes
omitted_instance_key
omitted_source_row_index
omitted_reason_codes
```

Version 1 action codes are `copy=0`, `compute=1`, and
`preserve_manual=2`. Reason codes distinguish unchanged, added,
source-changed, signature-spec-changed, no-reuse-source, and
preserved-manual rows. Deleted source-only keys are recorded in the omitted
columns. The writer validates action/reason consistency, unique target and
omitted keys, one-to-one source references, and complete target-row coverage
before storing the plan.

The crop adapter requires modern unique `instance_key` values and computes the
row source signature from the full normalized bbox, frame index, source pixel
fingerprint, source rowset family, ROI size, frame shape, center-rounding rule,
padding rule, and pixel representation. Optional clipped/refined lineage
columns are copied from the target source rowset and revalidated before
publication. A historical crop without the Phase 1 signature contract cannot
silently become a reuse source; the first replacement computes every row and
then becomes eligible as a base.

Dense ROI pixels are streamed one complete logical output chunk at a time. The
single driver owns every output chunk, group, attr, completion marker, and
pointer update. Compact identity/geometry/signature/plan arrays may remain in
memory; source frames and the full dense ROI payload do not. Every output ROI
chunk is read back and compared before publication. Immediately before
selection, the writer recomputes the source row signatures and rowset
fingerprint, rechecks optional lineage, and checks the expected prior crop
publication generation. It then marks the run complete and changes
`latest`, `latest_complete`, `latest_materialized`, and `latest_any` in one
parent-attrs update. Source mutation, frame-read failure, or a newer competing
publication leaves the staging run failed and does not overwrite the prior or
newer selected run.

This publication rule depends on serialized ownership of the `crop_runs`
parent. The generation comparison detects a changed parent before publication,
but a generic POSIX Zarr attrs write is not a multi-process compare-and-swap.
The production reconciler must therefore ensure only one publisher owns a
recording/stage parent at a time. Phase 1 does not claim lock-free concurrent
publication.

The CLI is read-only unless `--apply` is supplied:

```bash
scripts/py -m fisheye.utils.materialize_incremental_crop \
  /path/to/recording_analysis.zarr \
  --source-rowset-path refined_detect_runs/<exact-run> \
  --source-pixel-fingerprint <stable-pixel-fingerprint> \
  --roi-size 512 512 \
  --base-crop-run <exact-phase1-base> \
  --output-run <new-run>
```

Production `--apply` execution belongs in an LSF compute job, not on a login
node. The first adapter is deliberately narrow: Zarr v3,
`raw_video/images_full` grayscale `uint8` frames, a directly materialized
modern source row group, and a Phase 1 base. External-video decode, clipped
proxy resolution, and reuse from unsigned legacy crops remain later adapters;
they fail closed here instead of weakening reuse checks.

The retained reproducible benchmark command is:

```bash
scripts/py tools/benchmark_incremental_crop.py \
  --rows 8192 --roi-size 64 --frame-size 128 --delta-fraction 0.01
```

On the 2026-07-18 workstation run, the delta contained 8,028 copied rows, 82
geometry-changed rows, 82 additions, and 82 omissions. It reduced computed
rows from 8,192 to 164 and source-frame bytes read from 4.00 MiB to 1.38 MiB.
It still wrote and validated the complete 32.0 MiB dense target and read 31.36
MiB from the base. Consequently, the final synthetic no-decode pass took 1.29 s
for the delta versus 1.02 s for full computation. That is an expected and
useful result: Phase 1 proves localized expensive computation and complete
atomic output, but physical row copying still performs O(N) dense I/O. Real
video decode/model work can make that worthwhile; if dense copying dominates,
the next physical optimization is the already-decided immutable base-plus-delta
resolver or provable aligned object reuse, not weakening identity validation.

The first base-plus-delta benchmark used the same 8,192-row, 64x64 ROI fixture.
It wrote and read back 164 delta rows (0.64 MiB), read no base pixels during
publication, and completed in 0.68 s versus 1.63 s for the standalone
copy-forward delta. Logical pixels matched the standalone output exactly. The
composite group occupied 0.37 MiB in this highly compressible fixture versus
0.98 MiB for the standalone replacement. This is synthetic local-hot-cache
evidence, not a PRFS durability claim.

Composition adds a read-side cost: three repeated complete reads were about
0.26 s composite versus 0.095 s standalone, and 256 random rows were about
0.27 s versus 0.069 s. The resolver bounds intermediate dense reads to 64 MiB,
but it still performs mapping, base/delta source reads, and output scatter.
Consequently, composite storage is appropriate for edit deltas and canaries,
not yet the unconditional selected production default. The next canary must
measure real PRFS-to-compute inference access and the future standalone
compaction path before broader promotion.

### Phase 2: keypoint copy-forward

- Copy compatible raw predictions and preserved reviewed rows by key.
- Infer only new or invalidated crops.
- Rebuild run-wide metrics/indexes without repeating inference.
- Verify that adding one observation schedules one observation for inference.

### Phase 3: subject-mask copy-forward and finalization

- Copy compatible authoritative dense rows and preserved manual edits.
- Infer/refine only new or invalidated rows.
- Rebuild or refresh derived caches explicitly.
- Validate complete dense authority before publication.

### Phase 4: reconciler and status projection

- Turn source deltas and stale state into exact-run materialization jobs.
- Expose pending/rebuilding/stale/failed state in registry and operator tools.
- Retry safely without changing the selected complete run prematurely.

### Phase 5: tracking and downstream analyses

- Rerun association over the necessary temporal domain without rerunning vision
  inference.
- Rebuild affected tracks/analyses at the narrowest trustworthy scope.
- Preserve exact observation and tracking lineage in outputs.

### Phase 6: evaluate active partial materialization

- Measure copy-forward latency, storage amplification, and cluster cost.
- Add active per-row pending semantics only if complete atomic replacement is
  still operationally inadequate.

The real four-well canary should exercise phases 0-3 before a true interacting-
subject tracker is promoted. The canary should include at least one added or
geometry-edited observation so it tests delta materialization rather than only
the unchanged happy path.

## Acceptance Tests

The implementation is not complete until it proves these cases:

1. Adding one observation to a large source rowset invokes expensive inference
   for only that observation while publishing a complete replacement run.
2. Editing one bbox preserves its existing `instance_key` and recomputes only
   geometry-dependent descendants for that key.
3. Reordering source rows performs keyed realignment without model inference.
4. Deleting one observation omits it without recomputing unchanged survivors.
5. Existing reviewed keypoint and dense mask edits survive an unrelated add.
6. A model or incompatible configuration change prevents row reuse and triggers
   the expected full stage inference.
7. A source change during processing fails publication and leaves the prior
   complete run selected.
8. Duplicate keys, missing modern keys, keyset mismatch, or invalid reuse
   provenance fail closed.
9. Copied and newly computed rows are indistinguishable to ordinary consumers
   except through audit provenance.
10. Run-wide indexes, RLE offsets, summaries, and metrics match a clean reference
    materialization.
11. Parallel execution writes only whole, non-overlapping physical Zarr chunks.
12. Registry/status surfaces distinguish stale selected output from a replacement
    currently being built.

Performance acceptance should report model-inference rows, copied rows, bytes
read/written, wall time, and peak memory. A correct implementation that silently
falls back to all-row inference does not satisfy this decision for ordinary
row additions or edits.

## Non-Goals

- Replacing recording Zarrs with a mutable multi-recording project file.
- Rerunning raw detection after a manual refined-detect edit.
- Retraining a model because one recording observation changed.
- Overwriting immutable raw model outputs or reviewed authoring data.
- Treating `instance_key` as cross-frame or cross-recording animal identity.
- Making a partially complete run authoritative in the initial implementation.
- Optimizing away inexpensive deterministic scans at the cost of correctness.
- Automatically inferring or overwriting biological `subject_id` from appearance.

## Open Implementation Decisions

The architecture is decided, but these physical details still need focused
prototypes:

1. Whether large unchanged dense payloads should be copied row-wise, by aligned
   chunk, or by a storage-level reference/copy mechanism.
2. Which derived caches are cheaper and safer to rebuild run-wide versus copy
   forward by row.
3. Storage-retention policy for superseded complete runs created by frequent
   review edits.
4. Whether the first reconciler stores keyed deltas in Zarr, registry tables, or
   immutable sidecar manifests.
5. Stage-specific thresholds for choosing delta materialization versus a full
   deterministic transformation when no model inference is involved.

These choices must not weaken the core rule: ordinary localized observation
changes compute localized expensive work and publish a complete, exact-source,
validated replacement artifact.

## References

- `docs/instance_track_subject_identity_contract.md`
- `docs/refined_detect_row_identity_contract.md`
- `docs/sleap_palette_storage_assessment.md`
- `docs/manual_add_row_propagation_design.md`
- `docs/realtime_sparse_row_index_contract.md`
- `docs/mutable_review_runs_contract.md`
- `docs/refined_detect_sparse_instances_schema.md`
- `docs/multi_subject_tracking_phase5_plan.md`
- `docs/dask_zarr_write_safety.md`
