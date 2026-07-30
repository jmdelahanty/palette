# Crimson Inference, Commit, And Palette Compaction Decision

<!-- decision-meta
status: accepted-design
created: 2026-07-29
last_updated: 2026-07-29
owner: jeremy
scope: interactive detection/keypoint authoring, pluggable inference,
  append-only commits, successor compaction, and publication ownership
depends_on: docs/stable_identity_incremental_materialization_decision.md,
  docs/crop_geometry_storage_contract_v1.md,
  docs/keypoint_storage_contract_v2.md,
  docs/mutable_review_runs_contract.md
-->

## Decision Summary

Palette will first implement the reference successor DAG for refined
detections, geometry-only crops, and raw keypoints. The DAG will use a
producer-neutral inference boundary so a later Crimson model runtime can
produce the same terminal inference evidence as a Palette cluster worker.

Crimson may own interactive draft state, model loading, preview inference, and
commit initiation. Palette remains the sole owner of canonical compaction,
Zarr physical layout, manifests, consolidated metadata, publication gates, and
selector activation.

The lifecycle is:

```text
Crimson or Palette draft/edit
  -> append-only committed edit and optional inference evidence
  -> Palette freezes one exact event generation
  -> Palette compacts complete immutable successor snapshots
  -> validation and consumer gates
  -> separate selector promotion
```

Names such as `D2`, `C2`, `Kraw2`, and `Kref2` are explanatory generation
labels only. Persisted artifacts remain versioned runs under their existing
families and bind exact parent run IDs and manifest digests.

## Responsibility Boundary

| Concern | Owner |
| --- | --- |
| Interactive selection, editing, preview, undo, and local draft recovery | Crimson |
| Optional loaded-model inference for immediate preview | Crimson |
| Cluster or workstation batch inference | Palette inference provider |
| Append-only edit/inference transaction submission | Crimson or Palette client |
| Event validation, concurrency checks, identity allocation, and generation freeze | Palette |
| Detection, crop, raw/refined-keypoint compaction | Palette |
| Chunking, sharding, codecs, manifests, consolidation, and receipts | Palette shared storage modules |
| Selector changes and rollback | Palette promotion operation |
| Exact schema reading and fail-closed selection | Crimson consumer |

Crimson must not become a second independent canonical Zarr writer. Duplicating
the compactor and storage-policy implementation in C++ would allow identity,
schema, codec, and selector behavior to drift between applications.

## Successor DAG

The first DAG edge is:

```text
validated refined-detection successor
  -> complete geometry-only crop successor and reconciliation plan
  -> terminal inference for added or invalidated crop rows
  -> complete raw-keypoint successor
  -> optional quality, refined-keypoint, and body-frame successors
```

Crop reconciliation is keyed by `instance_key`:

- unchanged detection and crop geometry: reuse the compatible parent pose row;
- new detection: materialize its crop pixels and run inference;
- changed bbox or crop geometry: rerun inference because ROI coordinates
  changed;
- deleted detection: omit the crop and keypoint rows from the successor;
- changed model, preprocessing, skeleton, or pixel authority: fail the
  incremental v1 path and require an explicit compatible migration or full
  recomputation.

The public crop and keypoint runs remain complete immutable snapshots. Sparse
computation is an implementation optimization, not a partial public schema.

## Pluggable Inference Boundary

The DAG will depend on an inference-provider interface rather than a specific
YOLO, GPU, process, or host implementation. Two provider classes are intended:

1. `PaletteBatchInferenceProvider`: materializes bounded crop work packages on
   node-local scratch and runs the configured model.
2. `ValidatedExternalInferenceProvider`: accepts terminal evidence produced by
   a future Crimson model runtime and applies the same validation gate.

A conceptual request binds:

```text
instance_key
target crop row/signature
source pixel-authority digest
model/schema/checkpoint digest
preprocessing digest
skeleton digest
```

A terminal result supplies exact typed ROI-local keypoints, per-keypoint
confidence, pose confidence, pose bbox, and success state for the same key.
The persisted request/result envelope will be frozen before external Crimson
writes are enabled.

Palette validates that every added or invalidated key appears exactly once,
that no unrelated key is supplied, and that every result binds the target crop
signature. A result computed from an older bbox or crop is stale and fails
closed even if its `instance_key` is unchanged.

## Bounded Inference

“Bounded” constrains working memory and GPU occupancy; it does not impose a
maximum recording size or total number of observations.

- Only rows classified as added or invalidated enter expensive inference.
- Crop pixels are decoded from the source-video authority or a validated
  durable flat ROI cache.
- Work is staged to node-local scratch when practical.
- Pixel payloads are not added to the analysis Zarr.
- Rows are processed in bounded batches selected from measured RAM/GPU limits.
- Zarr output uses whole, non-overlapping physical-unit ownership.
- Complete indexes, row order, signatures, and manifests are rebuilt for the
  successor.

The initial in-memory successor preparer is reference logic. The production DAG
must use bounded writers so recording-scale publication does not retain duplicate
full input and output tables unnecessarily.

## Draft, Commit, And Compaction

These states are distinct:

### Draft

Mutable Crimson session state used for responsive editing, preview inference,
undo, and crash recovery. It is not canonical and cannot trigger downstream
publication until explicitly committed.

### Commit/event

A small immutable append-only transaction bound to an exact base snapshot and
expected previous revision. It records edit identity, actor, reason, time,
operation, and any model inference evidence. Committing does not resize an
existing canonical array or change a selector.

### Compaction

Palette validates a closed event prefix and creates new complete immutable
successor runs. It reuses compatible rows, computes invalidated rows, rebuilds
CSR indexes, validates manifests and metadata, and writes a digest-bound
receipt. Old runs remain immutable.

### Promotion

A separate fail-closed operation activates only a fully validated set of
compatible successors. Compaction success alone does not authorize selection.

## Detection, Crop, And Keypoint Semantics

A manual detection addition first creates a detection observation and stable
`instance_key`. Its crop geometry is derived; there is no independent manual
crop-row append. The corresponding raw-keypoint row is then one of:

- successful model inference;
- explicit terminal inference failure with `pose_success=false` and NaN pose
  payloads; or
- pending work, which blocks publication of the complete successor.

A later manual landmark correction is a refined-keypoint edit keyed by the same
`instance_key` and skeleton label. It does not create another detection.

Crop geometry normally has no independent edit delta because it is
deterministically derived from the refined detection, crop policy, and pixel
authority. Changing the crop policy is an explicit new crop lineage/profile and
invalidates dependent pose results as declared by that contract.

## Identity And Concurrency

Every transaction binds the exact base snapshot ID and manifest digest. Palette
must reject or explicitly rebase stale commits; it must never apply them
silently to a newer selected snapshot.

`instance_key` remains observation/edit-lineage identity, not subject or track
identity. New manual observations require the contract allocator and retired
keys are never reused. Concurrent Crimson clients require server-coordinated
row-ID/key allocation or a reservation protocol before multi-writer commit is
enabled.

Each event also carries a unique idempotency identity so retrying a network
submission cannot apply the same edit twice.

## Publication Atomicity

Atomicity is a workflow guarantee: readers cannot select a new crop run with an
old incompatible keypoint run, or a new keypoint run whose required inference
is incomplete. It does not require multiple Zarr groups to appear through one
portable filesystem syscall.

Candidates may be staged and imported separately while selector-ineligible.
One final receipt binds their run IDs and manifest digests. Promotion either
updates the complete compatible authority set under the versioned selection
contract or changes nothing.

Failed candidates remain unselected and may be retained as explicit diagnostic
artifacts. A live Crimson session continues reading its previously bound
immutable snapshot until it deliberately reloads the promoted successor.

## Rejected Alternatives

### Crimson writes canonical Zarr snapshots directly

Rejected. It duplicates Palette storage policy and publication logic and makes
cross-language schema drift likely.

### Append rows independently to existing detection, crop, and keypoint arrays

Rejected. Array lengths, ordering, CSR offsets, row signatures, derived caches,
and physical chunks can become observably inconsistent.

### Rerun inference for the whole recording after every edit

Rejected as the default. Complete successor publication is required, but
compatible unchanged rows should be reused.

### Treat an absent inference row as a failed inference

Rejected. Absence means pending or interrupted work. A terminal failure must be
explicit and carries the contract-defined failed payload.

## Implementation Checklist

### Palette reference workflow

- [x] Implement exact crop successor reconciliation and selector-ineligible
      publication.
- [x] Implement complete raw-keypoint successor preparation and
      selector-ineligible publication.
- [x] Test additions, crop changes, removals, success, terminal failure, and
      missing-result rejection.
- [x] Implement the same terminal-evidence boundary for bounded clip inference:
      exact instance keys, crop signatures, coordinate/model/preprocessing and
      input-package bindings, immutable source-array hashes, and explicit
      success/failure rows.
- [x] Add a selector-ineligible recording finalizer that rematerializes raw,
      quality, refined, and body-frame snapshots through shared byte planners
      and binds them with the crop snapshot in one receipt.
- [ ] Add a bounded inference-provider protocol and Palette batch provider.
- [ ] Add bounded physical-unit output writers for recording-scale successors.
- [x] Add an opt-in DAG fragment that validates per-clip terminal sidecars and
      publishes a direct-path selector-ineligible integration bundle.
- [ ] Add the production archive-import transaction after the clipped campaign
      emits strict recording-level detection/refined/crop authorities.
- [ ] Benchmark inference, compaction, publication, peak RSS, and local/shared
      transfer phases.

### Commit and compaction

- [ ] Freeze the cross-application edit/inference transaction schema.
- [ ] Add idempotent submission and stale-base conflict behavior.
- [ ] Add server-coordinated manual observation identity allocation for
      concurrent editors.
- [ ] Implement keyed refined-keypoint deltas and immutable compaction.
- [ ] Regenerate quality, body-frame, and other declared downstream successors.
- [ ] Add a versioned all-or-nothing authority-set promotion operation.

### Crimson integration

- [ ] Load an approved pose model and preprocessing contract.
- [ ] Produce preview results through the frozen inference-result schema.
- [ ] Keep drafts locally recoverable and visibly non-canonical.
- [ ] Submit explicit commits bound to the selected base snapshot.
- [ ] Display pending, rejected, compacting, completed, and promoted states.
- [ ] Reload only a fully promoted compatible successor set.

## Consequences

This design adds an explicit transaction and compaction boundary, but it keeps
one canonical storage owner, permits responsive Crimson inference, avoids
whole-recording recomputation for local edits, and preserves complete immutable
archives for downstream consumers and training exports.
