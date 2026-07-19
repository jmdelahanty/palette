# Keyed Downstream Compaction Contract

Status: implemented for raw keypoints and raw subject masks in Zarr v3.

## Purpose

A crop edit, added detection, or deleted detection changes the target crop
rowset without making every prior downstream result obsolete. Keyed compaction
creates a new complete immutable raw run while rerunning inference only for
rows whose crop source changed.

`instance_key` answers which observation a row represents. The crop
`source_row_signature` separately answers whether the pixels and geometry used
by inference are unchanged. Reuse requires both to match under the same signed
specification.

## Inputs

Each compaction binds:

- one complete target `crop_runs/<run>` with unique `uint64 instance_key` and
  signed source rows, or one verified auxiliary clipped-collection proxy with
  an exact historical-manifest identity bridge;
- one complete immutable base raw run;
- zero or more complete work-package inference runs carrying
  `incremental_materialization_role=delta_replacement_rows`;
- one new output run name.

The replacement keyset must equal exactly the rows classified `added`,
`source_changed`, or `signature_spec_changed`. Missing replacements, extra
reruns of reusable rows, duplicate keys across replacement runs, model/schema
differences, and stale crop bindings fail closed.

Model identity is not inferred from a pathname alone. Keypoint compaction
requires one `keypoint_model` SHA-256 artifact fingerprint and mask compaction
requires one `subject_mask_unet_checkpoint` SHA-256 fingerprint. Base
collection keypoint runs resolve this identity through their bound source
shards. Keypoint inference parameters must also agree exactly across the base
sources and replacements.

Package-backed inference outputs also persist the exact selected
`source_row_signature` values and their signature specification. Compaction
compares those stored values with the live target crop before accepting a
replacement. The temporary pixel package may therefore be cleaned after its
inference consumers and compaction finish without erasing the proof of which
crop generation produced the predictions.

The resulting source map is computed with NumPy sort/search operations rather
than a Python tuple/dictionary identity map. It therefore remains compact at
recording scale.

### Historical clipped proxy bootstrap

Clipped collection proxy crops are intentionally marked
`palette_run_completion_status=auxiliary`; they must never become ordinary
crop selectors. They are nevertheless eligible immutable rowsets for keyed
compaction only when `proxy_crop_complete=true`,
`stage_selector_eligible=false`, `crop_storage_mode=geometry_only`, their
versioned proxy schema is recognized, and their finalized collection identity
is present.

Historical proxies predate `source_row_signature`. Their compatibility
signatures are derived under
`palette.legacy_proxy_crop_signature_bootstrap` version 1 from stable
`instance_key`, canonical frame index, ROI origin, clip index, clip-local frame
index, recording identity, finalized collection identity, frame/ROI shape,
and crop semantics. Frame plus ROI origin and fixed ROI shape fully determine
the crop pixels; normalized bounding boxes are deliberately not required.

When a historical mask proxy is also keyless, keys may be supplied only by a
complete refined collection manifest that names the exact raw shard paths,
binds the same target crop, covers every target row exactly once, and matches
at least four stable lineage columns for every shard. Physical
`detection_indices` are excluded because they legitimately reset inside each
clip shard. This bridge is labeled
`derived_from_exact_refined_collection_manifest_v1`; there is no positional
fallback.

Package-backed keypoint and mask inference persists the selected bootstrapped
signatures in its replacement run. Subsequent compaction therefore validates
the actual historical crop rows used by inference, not merely their keys.

## Keypoint snapshots

`fisheye.utils.compact_keypoint_deltas` writes a standalone complete
`keypoints_runs/<run>` in exact target-crop order.

- Prediction columns are copied in bounded batches from either the base or the
  appropriate replacement run.
- Lineage columns come from the target crop, not from stale base positions.
- `source_crop_row_ids` is the dense target range and `instance_key` must equal
  the target key vector exactly.
- `frame_counts`, `n_rois`, `n_keypoints`, and `heading_usable` are recomputed.
- The ordinary canonical 1,024/16,384-row inner grids and 131,072-row indexed
  shard policy are retained.
- Each output outer shard is written serially as one owned unit and read back
  before completion.

Keypoint coordinates are small enough that physical standalone publication is
the simpler compatibility boundary. The implementation still does not load
the full coordinate run into memory.

Dry-run:

```bash
scripts/py -m fisheye.utils.compact_keypoint_deltas \
  /path/to/recording_analysis.zarr \
  --base-run <complete-keypoint-run> \
  --target-crop-run <complete-target-crop> \
  --replacement-run <delta-keypoint-shard> \
  --output-run <new-keypoint-snapshot> \
  --dry-run --json
```

Omit `--dry-run` only in an LSF compute job to write and promote the snapshot.

## Subject-mask snapshots

Decoded probability masks are too large to copy merely to obtain a new run
name. `fisheye.utils.compact_subject_mask_deltas` therefore publishes a
depth-one complete logical `subject_mask_runs/<run>`. Schema version 2 permits
one or more immutable physical raw bases:

```text
target row
  -> source_codes[row]
       0: immutable base mask row
       1: local delta probability row
  -> source_run_indices[row]
       >=0: index into composite_base_subject_mask_run_paths
       -1: delta payload
  -> source_row_indices[row]
```

The new run physically owns:

- its complete target lineage and `instance_key` vector;
- the complete source mapping;
- replacement-only `mask_probs_roi_delta`;
- physically compacted row metrics;
- complete semantic, crop, run, and model provenance.

It deliberately does not expose a partial top-level `mask_probs_roi` array.
`CompositeSubjectMaskArray` resolves the base and delta as one read-only
four-dimensional probability surface. The unified subject-mask loader and the
smart subject-mask finalizer use that resolver. The whole-recording validator
also samples through it.

Every physical base must be a standalone probability run; composite-on-
composite chains are rejected. A completed refined collection run may act as
the authoritative manifest for its explicitly declared
`source_subject_mask_shard_run_paths`, without becoming a probability base
itself. The output still has reference depth one because it points directly to
the raw shard arrays. Base deletion or overwrite is blocked across both
`subject_mask_runs` and `subject_mask_shard_runs` while a composite dependent
exists. Standalone export/materialization remains the compatibility path for
external readers that cannot resolve the composite schema.

Dry-run:

```bash
scripts/py -m fisheye.utils.compact_subject_mask_deltas \
  /path/to/recording_analysis.zarr \
  --base-run <complete-raw-run-or-refined-collection-manifest> \
  --target-crop-run <complete-target-crop> \
  --replacement-run <delta-mask-shard> \
  --output-run <new-mask-snapshot> \
  --dry-run --json
```

## Publication and failure behavior

Both writers create a new run and never overwrite an existing snapshot. They:

1. capture the input key/signature state and prior selection pointers;
2. write and read back the complete physical or logical output;
3. validate exact target coverage and source compatibility;
4. re-read input identity immediately before publication;
5. fail if an input or the parent selection changed;
6. mark the new run complete and promote `latest`/`latest_complete` only after
   every check passes.

A failed run is retained with failure metadata for diagnosis. The previous
complete pointer remains selected.

## Later review deltas

Automated inference replacement shards and human review deltas are different
inputs but converge on the same versioning rule. Review/edit partitions remain
sparse and keyed by `instance_key`. A later compaction/finalizer job takes a
fixed delta generation, applies it shard-by-shard to the chosen immutable base,
publishes a new immutable snapshot, and only then advances the canonical
pointer. Edits arriving during compaction belong to the next generation.

Raw mask probability compaction should continue to use the depth-one resolver.
Editable refined `masks_roi` remains a dense authority under the subject-mask
direction contract; its sparse review deltas are compacted into a new dense
reviewed snapshot rather than turning the active authoring surface into a deep
reference chain.
