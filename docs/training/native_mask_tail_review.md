# Native mask-derived training annotations

`fisheye.training.native_mask_tail_review` creates new 19-point annotation
versions within an existing native `_training.zarr`. It reuses the recovered
mask workflow's geometry, payload builder, atomic publisher, and crop-only
editor. It requires exact, complete mask and keypoint run names with recorded
manual approval for training. Every declared mask channel must be approved and
available; body and swim-bladder channels and three named head landmarks are
required. It does not activate a selector or modify source labels.

```bash
scripts/py -m fisheye.training.native_mask_tail_review RECORDING_training.zarr \
  --mask-run REVIEWED_DENSE_MASK_RUN \
  --keypoint-run REVIEWED_KEYPOINT_RUN \
  --version tail19_review_v001 --report receipt.json
# Add --apply after checking the plan; use a fresh receipt path.
```

This is an explicit scientific/identity extension:

- The existing recovered v1 contract, digest grammar, coordinate labels, and
  default recipe are preserved. Native snapshots use
  `palette.training.native_mask_tail_review.v1` and `native_training_roi_xy`.
- Native mask and pose rows join through explicit, unique `source_crop_row_ids`,
  with matching crop frame indices. Row order need not match. Images remain an
  exact pixel copy. Frame indices retain the source crop's domain; original
  crop row IDs and available sensor origins remain separate lineage arrays.
- Existing head, snout, and fin labels map by **name**, preserving finite values.
  Origin code 4 means an existing source keypoint; it does not invent a manual
  edit history. Missing snouts use the mask estimate, and missing fins remain
  empty. The entire original keypoint row and its ordered labels remain bound.
- All 11 tail stations, including tail tip, use the unchanged mask-derived
  arc-length recipe. The earlier source tail tip/mid-tail are retained as
  lineage, rather than mixed into the new sampling sequence. Snout diagnostics
  describe the mask estimator even when an existing snout was retained.
- All generated rows begin ineligible for training and pending manual review.
  Complete saves require every landmark finite and inside the crop and retain
  the existing head geometry QC. Failed derivations remain targeted tasks.

Each snapshot contains a materialized crop, an immutable dense mask source,
an editable dense mask copy, an immutable keypoint seed, and editable keypoints.
The original archive can retain its other active schemas. Source array digests,
run attributes, reviewed state, and producer provenance are sealed in the new
version. Source identity is rechecked before each atomic child publication and
afterward. Consolidation is the final visibility step. Existing versions are
refused, including incomplete ones; retries use a fresh version and preserve
failed unselected children for diagnosis.

The native adapter bounds its selected source payload to 512 MiB and publishes
with one writer. It does not resolve sensor-coordinate authority or promote
these annotation snapshots into analysis. Registry activation and training
export remain separate steps. Correcting masks requires a new derivation
version before further manual label work; existing edits are never overwritten.

The browser shares the crop-only save/checkpoint path with recovered labels,
including per-landmark manual origin tracking and all-visible row checks. The
internal historical `recovered_roi_only` flag denotes that compatibility path;
it does not relabel native source frame identities as recovered sample rows.
