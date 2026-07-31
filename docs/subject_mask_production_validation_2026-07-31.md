# Subject-Mask Production Validation

Date: 2026-07-31

Status: shared receipt and publisher boundary implemented; reference-full remains
the default; no production writer, selector, registry, or full-duration job has
been activated

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

## Production Lifecycle

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

### Scientific quality

`subject_mask_quality_runs` remains a separate scientific computation.  It may
intentionally traverse every row once.  Publication validation proves internal
and lineage correctness; it does not replace containment, overlap, temporal, or
anatomical quality assessment.

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
- [ ] Emit raw worker semantic receipts during maintained inference.
- [ ] Emit refined worker semantic receipts during maintained finalization.
- [ ] Aggregate logical hashes during the ordered recording-level merge rather
      than rereading the completed source.
- [ ] Bind manual-edit/compaction receipts for changed and inherited units.
- [ ] Compare reference-full and production-streaming manifests and logical
      hashes on the completed 22,926-row canary.
- [ ] Benchmark phase time, decoded bytes, peak RSS, and bounded reopen reads.
- [ ] Run one selector-ineligible full-duration production-streaming canary.
- [ ] Obtain Palette and Crimson correctness/performance review.
- [ ] Activate a versioned profile only after those gates pass.

## Safety Boundary

The current cluster job is pinned to its deployed Palette commit and continues
to use reference-full validation.  This implementation does not alter that
process.  It also does not authorize a full-duration run yet: maintained raw
inference and refined finalization must first emit the frozen receipt evidence,
and the two validation modes must agree on the completed 22,926-row fixture.

The shared inference hardware/runtime contract is documented in
`docs/inference_accelerator_provenance_2026-07-31.md`. Subject-mask publication
receipts retain the upstream stage and run provenance before scratch cleanup.
