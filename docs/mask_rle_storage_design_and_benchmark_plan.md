# Compact Mask Storage and RLE Benchmark Plan

## Context

Palette mask runs currently favor dense numeric Zarr arrays because they are
easy to inspect, chunk, train from, render, and edit. The dominant arrays are:

- `eye_masks_runs/<run>/masks_roi`: `(N, C, H, W) uint8`
- `refined_eye_masks_runs/<run>/masks_roi`: `(N, C, H, W) uint8`
- `subject_mask_runs/<run>/masks_roi`: `(N, C, H, W) uint8`
- `subject_mask_runs/<run>/mask_probs_roi`: `(N, C, H, W) float16/float32/uint8`
- `refined_subject_masks_runs/<run>/masks_roi`: `(N, C, H, W) uint8`

This is a good compatibility representation, but it scales poorly when we copy
or migrate mask runs. A `512 x 512` binary channel is 262,144 bytes before
compression, and multi-channel mask runs multiply that by row count and channel
count. Probability arrays are larger and are not good RLE candidates.

## Storage Principle

Training, Crimson, and review tools need dense tensors at the point of use.
They do not necessarily require dense masks as the canonical on-disk encoding.

The target architecture should separate:

- **Logical mask surface**: a row/channel mask with shape `(H, W)`.
- **Physical storage encoding**: dense `uint8`, bit-packed, RLE, contour, or
  reference/alias.
- **Materialization API**: a reader that returns dense masks for training,
  rendering, editing, and metrics regardless of physical encoding.

## Implementation Status

Implemented surfaces:

- `fisheye.shared.mask_rle` encodes/decodes exact binary COCO-style RLE with
  typed NumPy arrays.
- `fisheye.shared.mask_store.open_mask_store(...)` materializes dense masks from
  dense `masks_roi`, compact fixed-size `mask_bitpacked`, or compact component
  RLE.
- `MaskStore.storage_surface` and `MaskStore.storage_path` report the selected
  physical backing surface (`masks_roi`, `mask_bitpacked`, or `mask_rle`) so
  exporters, auditors, and provenance writers do not infer this independently
  from encoding names.
- `fisheye.shared.mask_store.write_bitpacked_mask_store_from_dense(...)` streams
  dense masks row chunks into `mask_bitpacked/masks_packed` with
  `bitpacked_binary_v1` metadata.
- `fisheye.shared.mask_store.write_component_rle_mask_store_from_dense(...)`
  streams dense masks row chunks into `mask_rle/components/...` without loading
  the full dense tensor into memory.
- `fisheye.refinement.finalize_subject_masks --mask-storage dense_and_bitpacked`
  writes the historical dense `masks_roi` plus an additive compact
  `mask_bitpacked/masks_packed` mirror.
- `fisheye.refinement.finalize_subject_masks --mask-storage bitpacked_v1`
  writes the same compact bitpacked store and removes the dense `masks_roi`
  compatibility cache after finalization/postcompute completes.
- `fisheye.refinement.finalize_subject_masks --mask-storage dense_and_rle`
  writes the historical dense `masks_roi` plus an additive compact
  `mask_rle/components/...` mirror.
- `fisheye.refinement.finalize_subject_masks --mask-storage rle_v1`
  writes the same compact `mask_rle/components/...` store and removes the dense
  `masks_roi` compatibility cache after finalization/postcompute completes.
- `fisheye.utils.materialize_refined_subject_mask_store` can dry-run,
  materialize, refresh, or delete the dense `masks_roi` compatibility cache for
  a compact refined-subject run, and can regenerate compact `mask_rle` from the
  current dense cache with `--refresh-rle`.
- The same materializer can regenerate or scoped-refresh compact bitpacked
  storage from dense:
  `--refresh-bitpacked --components eye_left --rows 42 --apply` rewrites only
  the selected fixed-size packed row/channel cells.
- The materializer can refresh compact RLE by component:
  `--refresh-rle --components eye_left --apply` rewrites only the selected
  `mask_rle/components/<component>/` groups and preserves unrelated component
  groups. Dense `masks_roi` remains the authoritative review/edit write surface;
  direct in-place RLE painting is intentionally deferred.
- Palette-owned refined-subject review/writeback paths now refresh
  `mask_bitpacked` immediately for touched dense rows/components when the
  bitpacked mirror exists. They continue to mark `mask_rle` stale because RLE
  refresh may shift variable-length offsets for later rows and remains an
  explicit maintenance step.
- `fisheye.diagnostics.benchmark_mask_rle_storage` uses the same shared writer
  so benchmark layout and production layout stay aligned.
- Stage-array validation accepts refined-subject-mask runs with either dense
  `masks_roi`, a valid compact `mask_bitpacked` store, or a valid compact
  `mask_rle` store.
- `fisheye.shared.refined_subject_masks_io.load_refined_subject_masks_run_tables(...)`
  reads dense or compact stores through `MaskStore` and fails early when an
  advertised compact store is stale/unreadable instead of silently returning a
  metadata-only table.
- `docs/examples/read_subject_masks_from_example_recording.py` demonstrates
  the storage-agnostic refined read path through `MaskStore` instead of direct
  `masks_roi` indexing.
- `subject_mask_performance` registry rows expose compact storage fields:
  `mask_storage_encoding`, `mask_store_encodings_json`,
  `masks_roi_materialized`, `mask_rle_materialized`, `mask_rle_schema_id`,
  `mask_rle_encoding`, and `mask_rle_layout`.

Current compatibility rule: `dense_uint8` remains the default.
`dense_and_bitpacked` and `dense_and_rle` are additive shadow/audit modes.
`bitpacked_v1` and `rle_v1` are available for compact-only experiments and
consumers that are already audited through the `MaskStore` materialization
boundary.

Validation rule: production batch runs use invariant validation for compact
stores. This validates the typed compact store without decoding the entire dense
logical surface. Full dense round-trip validation remains available for targeted
audits and canaries.

## Next Direction: Three-Tier Mask Storage

The current RLE work clarified an important lifecycle distinction: the best
storage representation depends on whether masks are still being edited.

Recommended direction:

- `dense_uint8`: use for training artifacts, active painting/review sessions,
  GPU/model input, and debugging. This remains the easiest and safest write
  surface because row/channel edits rewrite ordinary fixed-size Zarr chunks.
- `bitpacked_binary_v1`: add as the compact editable/publish surface for
  produced analysis masks that may still need review. It stores one bit per
  binary pixel, so the physical payload is fixed-size and roughly `8x` smaller
  than dense `uint8` before compression. Unlike COCO RLE, editing one row does
  not shift offsets for later rows.
- `component_rle_v1`: keep as the compact final/archive/export surface for
  stable masks where smallest footprint and read-mostly access matter more than
  random write locality.

The reason not to use RLE as the live editing intermediate is structural, not
just implementation maturity. A component RLE store uses one variable-length
`counts` array plus `indptr`. If row 1 changes length, offsets for later rows in
that component can shift. Palette currently avoids unsafe in-place mutation by
refreshing the affected component from dense. That preserves correctness, but it
is not the right shape for frequent painting.

Bitpacking is the proposed middle tier because it is still chunked/fixed-size:
changing frame 1, component `eye_left`, rewrites only the containing packed
physical chunk rather than re-encoding all later rows in the component.

Proposed lifecycle:

1. Cluster finalization writes a local dense working surface while computing
   geometry and contours.
2. Publication writes `bitpacked_binary_v1` to PRFS/NRS for reviewable analysis
   products, with optional dense materialization for immediate consumers.
3. Review/painting materializes or edits dense chunks, then updates the
   bitpacked mirror at chunk/row scope.
4. Final accepted runs may additionally publish `component_rle_v1` or convert to
   `rle_v1` when the run is stable.
5. Training/export artifacts remain dense `uint8`, because training code should
   not pay decode/unpack costs repeatedly once a dataset is frozen for training.

Implemented bitpacked slice:

- Defined `bitpacked_binary_v1` attrs: source logical shape `(N,C,H,W)`,
  `packed_axis="width"`, `packed_bitorder`, `packed_width_bytes`, component
  labels, value semantics, and chunk policy.
- Added `MaskStore` read support that returns dense `uint8` from bitpacked storage.
- Added writer/materializer helpers that pack/unpack row chunks with
  `np.packbits`/`np.unpackbits`.
- Added finalizer `--mask-storage` choices for `dense_and_bitpacked`,
  `bitpacked_v1`, and `dense_bitpacked_and_rle`.
- Added stage validation support for compact-only bitpacked refined-subject
  mask runs.
- Added batch-wrapper and LSF submitter support for bitpacked storage modes.
- Added Palette-owned refined-subject review/edit writeback support that
  refreshes only affected bitpacked row/channel cells after dense edits.

Remaining implementation checklist:

- Benchmark real refined subject-mask runs across dense-compressed,
  bitpacked-compressed, and component RLE before making bitpacked a production
  default.

Local benchmark command:

```bash
scripts/py -m fisheye.diagnostics.benchmark_mask_storage_encodings \
  /path/to/analysis.zarr \
  --family refined_subject_masks_runs \
  --run <run_name-or-latest> \
  --sample-rows 128 \
  --json-report /tmp/mask_storage_encoding_benchmark.json \
  --markdown-report /tmp/mask_storage_encoding_benchmark.md
```

This diagnostic writes temporary sampled stores for `dense_uint8`,
`bitpacked_binary_v1_probe`, and `component_rle_v1`, then reports logical bytes,
stored bytes, and encode/write/decode rates. Its bitpacked probe should remain
aligned with the production `bitpacked_binary_v1` contract.

Initial local benchmark results from 2026-06-20:

| recording | rows | dense stored | bitpacked stored | RLE stored | bitpacked encode | RLE encode | bitpacked decode | RLE decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| January refined subject-mask sample | 128 | 94.0 KiB | 58.5 KiB | 56.5 KiB | 3238 rows/s | 206 rows/s | 9076 rows/s | 732 rows/s |
| GoodCopBadCop refined subject-mask sample | 128 | 94.3 KiB | 55.4 KiB | 56.9 KiB | 2644 rows/s | 196 rows/s | 7774 rows/s | 725 rows/s |
| GoodCopBadCop refined subject-mask sample | 512 | 374.6 KiB | 212.3 KiB | 179.6 KiB | 3068 rows/s | 204 rows/s | 8615 rows/s | 738 rows/s |

Interpretation:

- Zarr compression already makes dense sampled masks surprisingly small for
  these sparse binary masks.
- Bitpacking still reduces stored bytes by about `1.6-1.8x` versus compressed
  dense in these samples and is much faster to encode/decode than RLE.
- RLE can be smaller than bitpacking on larger sparse samples, but its encode
  and decode rates are roughly an order of magnitude lower.
- This supports bitpacking as the reviewable/publishable middle tier and RLE as
  a stable/archive tier, not as the live editing tier.

## Is Zarr a Poor Fit for RLE?

Not necessarily. Zarr is strongest for typed arrays, and RLE can be represented
as typed arrays instead of JSON-per-row blobs.

A Zarr-native COCO-style RLE layout can use typed payload arrays plus offsets.
Prefer one compact group per semantic component:

```text
mask_rle/components/<component>/counts      uint32  (total_run_count_for_component,)
mask_rle/components/<component>/indptr      int64   (N + 1,)
mask_rle/components/<component>/present     bool    (N,)
mask_rle/components/<component>/area_px     int32   (N,)
mask_rle/components/<component>/bbox_xyxy   int32   (N, 4)
mask_rle attrs: shape=[H, W], encoding, value_semantics
```

This follows the same storage pattern already used by eye contour arrays:

```text
contours_left          float32 (n_points, 2)
contour_left_ptr       int64   (N,)
contour_left_len       int32   (N,)
```

Avoid JSON-per-row for production-scale masks. JSON is useful for manifests and
small metadata, but it is a poor physical layout for millions of variable-size
mask rows.

## Candidate Encodings

### Dense `uint8`

Current compatibility default.

Pros:

- Simple and fast for random row access.
- Works with existing Zarr chunking, Dask, NumPy, PyTorch, and Crimson readers.
- Easy to edit in place.

Cons:

- Expensive for mostly empty binary masks.
- Expensive to duplicate during crop-representation migrations.
- Multi-channel binary masks duplicate background pixels.

### Bit-packed Binary

Stores 1 bit per pixel instead of 1 byte per pixel.

Pros:

- Predictable 8x raw reduction for binary masks.
- Still array-oriented.

Cons:

- Requires unpacking before training/rendering.
- Less compression-friendly if masks have simple geometry compared with RLE.
- More awkward for partial row writes.

### COCO-Style RLE

Run-length encoding of binary masks flattened in COCO's Fortran/column-major
order. Runs alternate background and foreground counts.

Pros:

- Excellent for sparse foreground masks inside large ROIs.
- Exact binary representation.
- Handles holes and disconnected components.
- Well-known in segmentation tooling.

Cons:

- Poor random pixel access.
- Usually decode whole row/channel before use.
- Requires careful ordering/version documentation.
- Not appropriate for probability/logit arrays.

### Contours / Polygons

Store outlines and rasterize at load time.

Pros:

- Very compact for smooth masks.
- Already partly present for eye masks.

Cons:

- Can lose pixel-exact detail unless encoded carefully.
- Ambiguous for holes, multiple components, and rasterization policy.
- Less ideal as the only authoritative edited-mask representation.

### Label Map

Store one dense `(N, H, W)` integer class-ID map instead of `(N, C, H, W)`
binary channels.

Pros:

- Much smaller for mutually exclusive semantic segmentation.
- Easy to train from after one-hot expansion.

Cons:

- Not valid for overlapping multilabel masks.
- Palette subject masks are currently multilabel with `available_channels`.
- A scalar class-ID map silently drops information when two semantic structures
  are intentionally true at the same pixel.

### Paintera / Connectomics Label Maps

Paintera is a useful comparison point, but its core storage model solves a
different problem. In connectomics, label data is usually an instance
partition: each voxel belongs to one neuron/object label or background. Painting
a label into a voxel replaces the previous label. That makes a scalar label map
natural, and Paintera can scale by:

- painting into a temporary dense viewer-aligned canvas,
- tracking affected label-space blocks,
- committing only changed blocks to N5/Zarr/HDF5,
- maintaining optional label-to-block and unique-label metadata for mesh/update
  locality, and
- using label-multiset downsampled levels for multiscale visualization.

Palette's subject and eye masks are not a global instance partition. They are
semantic multilabel channels:

- `subject_body`, `eye_left`, `eye_right`, `eyes_union`, and `swim_bladder`
  are channel identities, not mutually exclusive scalar class IDs.
- `subject_body` can legitimately overlap internal anatomical structures such
  as eyes or swim bladder, depending on the component definition.
- `eyes_union` is explicitly derived as the union of left/right eye channels in
  training/export paths.
- `eye_left` and `eye_right` are intended as separate anatomical components,
  and refinement often splits or assigns them to separate components, but the
  storage contract still represents them as independent binary channels rather
  than a hard class-ID map.

Therefore the main Paintera lesson for Palette is not "store all masks as label
maps." The transferable pattern is:

- keep interactive editing dense and local,
- track changed rows/chunks/blocks,
- persist compact changed regions or compact binary encodings,
- materialize dense masks at the read boundary for Crimson, training, and
  metrics.

Scalar label maps may still be appropriate for a narrow subset of Palette
surfaces that are genuinely mutually exclusive, for example an ROI-local eye
instance map used as an intermediate assignment surface. They should not become
the canonical storage for overlapping semantic mask families without an
explicit lossy projection policy.

### Reference / Alias Runs

For migrations where ROI geometry is unchanged, create a new run that points to
the old mask arrays and records an identity transform instead of copying dense
masks.

Pros:

- Biggest win for crop pixel-contract migrations.
- Avoids duplicating masks when labels are unchanged.

Cons:

- Readers must follow references.
- Editing must decide whether to copy-on-write or edit the source.
- Requires clear provenance and lifecycle rules.

## Current Palette Mask Semantics Audit

This pass checks the current code/docs surface for masks such as `eye_left` and
`eye_right`.

### Subject-Mask Runs

Current subject-mask runs advertise multilabel semantics:

- `src/fisheye/segmentation/infer_unet_subject_masks.py` writes
  `output_semantics = "multilabel"`,
  `overlap_policy = "independent_sigmoid"`, and
  `probability_semantics = "sigmoid_multilabel_logits"`.
- `src/fisheye/segmentation/subject_segmentation.py` uses the same
  `overlap_policy = "independent_sigmoid"` convention for traditional
  subject-mask output.
- `src/fisheye/utils/export_subject_mask_training_zarr.py` defines:
  - `subject_v1_union = ("subject_body", "eyes_union", "swim_bladder")`
  - `subject_v1_lr = ("subject_body", "eye_left", "eye_right",
    "swim_bladder")`
- `src/fisheye/training/zarr_subject_mask_dataset.py` loads
  `target_valid_channels` and trains only valid supervised channels through
  `MaskedBCEDiceCriterion`, rather than using a softmax/one-hot class target.

Implication:

- A scalar label map is not a drop-in replacement for subject-mask runs.
- `available_channels` and `target_valid_channels` are part of the multilabel
  supervision contract and must be preserved by any compact encoding.

### Eye-Mask Runs

Current eye-mask runs have two supported label modes:

- `lr`: two channels, usually `eye_left` and `eye_right`
- `union`: one channel, an eye-objectness union

Relevant behavior:

- `src/fisheye/shared/mask_source.py` normalizes eye-mask sources into either
  `binary_identity = "lr"` or `binary_identity = "union"` based on channel
  count.
- `src/fisheye/utils/export_eye_mask_training_zarr.py` can export `label_mode`
  `lr` or `union`; union mode collapses one or two source channels into a
  single target channel.
- `src/fisheye/segmentation/train_unet_eye_masks.py` uses sigmoid outputs and
  optionally applies an LR channel-overlap penalty. The overlap penalty is a
  training regularizer, not a storage-level invariant.
- `src/fisheye/refinement/refine_eye_masks.py` converts source union or pair
  masks into canonical `eye_left` and `eye_right` outputs using keypoints,
  heading, centroid assignment, cleanup, and ellipse fitting. It records
  `source_eye_labels`, canonical `eye_labels = ["eye_left", "eye_right"]`,
  `mask_binary_identity`, and `mask_probability_identity`.

Implication:

- `eye_left` and `eye_right` are semantic/anatomical channel names.
- They are expected to represent separate eye components after refinement, but
  they are still stored as independent channels.
- A label-map representation could be useful as an intermediate left/right
  assignment surface, but the current training/export/refinement APIs expect
  dense binary channels.

### Refined Subject-Mask Runs

The canonical refined eye-capable schema is moving toward:

```text
label_schema_id = "subject_v1_lr"
mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]
```

The docs and code treat this as a refined component tensor, not a scalar class
map:

- `docs/eye_subject_mask_unification_design.md` states that refined
  eye-capable work should target `refined_subject_masks_runs` with explicit
  `eye_left` and `eye_right` components.
- `src/fisheye/shared/refined_subject_masks_io.py` exposes run-level
  `masks_roi`, `available_channels`, and per-component logical tables.
- `src/fisheye/utils/merge_subject_mask_runs.py` merges body and eye sources
  into a canonical `subject_v1_lr` tensor and preserves `eye_left` and
  `eye_right` as separate source channels.
- `src/fisheye/analysis/subject_shape_runs.py` consumes refined subject masks
  by component name and computes downstream geometry for body, swim bladder,
  and both eyes.

Implication:

- Compact storage should preserve the component-channel model and make dense
  materialization cheap for component readers.
- If a future writer chooses a scalar label-map cache for mutually exclusive
  eye-only rows, it should be recorded as a derived/intermediate encoding, not
  as the general refined subject-mask authority.

## Codebase Components That Would Need To Change

### Shared Mask API

Current partial seam:

- `src/fisheye/shared/mask_source.py`

Needed change:

- Introduce a `MaskStore` or `MaskSurface` abstraction with methods like:
  - `shape`
  - `channels`
  - `read(row_indices, channels=None) -> dense uint8/float array`
  - `write(row_indices, dense_masks, channels=None)`
  - `encoding`
  - `storage_bytes_estimate`
- Make it support dense, RLE, bit-packed, and reference-backed masks.
- Keep dense arrays as a compatibility backend.

### Segmentation Writers

Current dense/probability writers include:

- `src/fisheye/segmentation/infer_unet_eye_masks.py`
- `src/fisheye/segmentation/infer_unet_subject_masks.py`
- `src/fisheye/segmentation/eye_segmentation.py`
- `src/fisheye/segmentation/eye_segmentation_yolo.py`
- `src/fisheye/segmentation/subject_segmentation.py`
- `src/fisheye/segmentation/swim_bladder_segmentation.py`
- `src/fisheye/utils/backfill_subject_mask_runs.py`

Needed change:

- Add a write policy:
  - `dense_uint8`
  - `rle_binary`
  - `dense_plus_rle`
  - `probability_first`
- Keep `mask_probs_roi` separate from binary mask storage. RLE should only
  encode thresholded binary masks.

### Refinement and Review Tools

Dense assumptions are present in:

- `src/fisheye/tune/refined_subject_mask_review.py`
- `src/fisheye/refinement/finalize_subject_masks.py`
- `src/fisheye/refinement/refine_subject_masks.py`
- `src/fisheye/refinement/refine_eye_masks.py`
- `src/fisheye/visualization/visualize_eye_mask_patches.py`
- `src/fisheye/tune/eye_mask_review.py`
- `src/fisheye/tune/eye_mask_tuner.py`

Needed change:

- Decode compact masks to dense rows for display/editing.
- Encode edited dense rows back to the selected storage policy on save.
- Define copy-on-write behavior for reference-backed masks.
- Ensure component metrics, contours, area, bbox, and review statuses are
  refreshed after an encoded edit.

### Crimson Integration

Crimson can still render/edit dense masks. The storage backend only changes the
read/write boundary.

Needed change:

- Palette/Crimson readers should request dense row/channel masks from a mask
  materializer.
- Crimson edits dense masks in memory.
- Save path encodes the edited dense mask into the run's storage encoding.
- If a run is reference-backed, saving should either fail read-only or fork a
  dense/RLE copy-on-write run.

### Training and Export

Dense assumptions are present in:

- `src/fisheye/training/zarr_eye_mask_dataset.py`
- `src/fisheye/training/zarr_subject_mask_dataset.py`
- `src/fisheye/utils/export_eye_mask_training_zarr.py`
- `src/fisheye/utils/export_subject_mask_training_zarr.py`

Needed change:

- Training exporters and promotion tools should consume the mask API for
  analysis-source reads, not direct `refined_subject_masks_runs/*/masks_roi`
  arrays.
- Training artifacts themselves remain dense-only for this migration:
  `subject_mask_runs/<run>/masks_roi` is the model-input contract and
  `subject_mask_runs/<run>/mask_rle` is rejected by validators/loaders.
- Exporters can read compact analysis sources, but must materialize dense
  exported artifacts until there is an explicit future training-artifact
  contract for compact masks.
- The trainer still receives dense tensors.

### Diagnostics, Profiles, and Metrics

Dense scans are present in profile/audit utilities such as:

- `src/fisheye/utils/eye_mask_profile.py`
- `src/fisheye/utils/audit_subject_mask_training_sources.py`
- `src/fisheye/diagnostics/check_eye_masks.py`
- `src/fisheye/diagnostics/check_mask_components.py`

Needed change:

- Metrics should compute from `MaskStore.read_dense(...)` or from precomputed compact
  metrics such as area/bbox when available.
- Storage benchmarks should report both physical Zarr size and logical decoded
  mask shape.

## Implemented RLE Schema

For each mask run, support either dense arrays or a compact component-group
`mask_rle/` group.

```text
<mask_run>/
  attrs:
    mask_storage_encoding = "dense_uint8+component_rle_v1"
    mask_store_encodings = ["dense_uint8", "component_rle_v1"]
    masks_roi_materialized = true
    mask_rle_materialized = true

  mask_rle/
    attrs:
      schema_id = "palette_mask_rle_binary_v1"
      mask_encoding = "coco_rle_fortran_v1"
      mask_value_semantics = "binary_0_1"
      layout = "component_groups"
      encoded_shape_hw = [H, W]
      component_names = [...]

    components/
      00_subject_body/
        counts                  (total_count_values_for_component,) uint32
        indptr                  (N + 1,) int64
        present                 (N,) bool
        area_px                 (N,) int32
        bbox_xyxy               (N, 4) int32
      01_eye_left/
        ...
```

Compatibility rules:

- If `masks_roi` exists, legacy readers may use it.
- If `masks_roi` is absent and `mask_rle/` exists, modern readers must
  materialize dense masks on demand.
- If `mask_rle_stale == true`, `open_mask_store(..., prefer="rle")` must fail
  unless the caller explicitly passes `allow_stale_rle=True` for diagnostics.
  Ordinary consumers should prefer dense `masks_roi` or refresh compact storage
  with `materialize_refined_subject_mask_store --refresh-rle --apply`.
- Writers must record whether `masks_roi` is materialized using
  `masks_roi_materialized`, `mask_store_encodings`, and
  `mask_storage_encoding`.
- Probability arrays remain dense/quantized arrays and are not represented as
  binary RLE.

## Recommended Migration Target

Use whole-ROI typed-array RLE as the first compact canonical encoding.

This is the best fit for Palette's current Zarr model because it keeps the
logical mask surface exact while preserving typed, chunked arrays. It avoids
JSON-per-row blobs, avoids lossy contour-only storage, and avoids making every
consumer learn a tight-bbox ragged representation before the shared mask API
exists.

The recommended first target is:

- **Canonical compact surface**: `mask_rle/components/<component>/` with exact
  whole-ROI binary masks.
- **Compatibility cache**: optional dense `masks_roi` materialized from
  `mask_rle/`.
- **Future optimization**: optional `tight_bbox_rle_v2` after readers and
  writers are already using `MaskStore`.

Bit-packed masks are the fallback if RLE does not beat dense Zarr compression
enough in benchmarks. Contours remain derived geometry/QC outputs, not the only
authoritative mask surface.

### Why Not Tight-BBox RLE First?

Tight bbox plus RLE is probably the smallest final representation for sparse
ROI masks, but it adds two extra sources of complexity:

- each row/channel has its own encoded image shape and origin;
- materializers must paste decoded local masks back into full ROI coordinates
  before Crimson, training, metrics, and edit tools see them.

Those are solvable, but they are not the right first migration step. Whole-ROI
RLE gives an exact compact encoding while preserving the existing `(H, W)` mask
contract. Add bbox metadata in v1 for metrics and future migration planning,
then only promote bbox-local RLE after the API boundary is stable.

## Chunked Storage Contract For Compact Masks

Compact masks must still follow Palette's Zarr write-safety rule: no two
workers may write different logical slices inside the same physical Zarr chunk.

Recommended layout:

```text
<mask_run>/
  attrs:
    mask_storage_encoding = "dense_uint8+component_rle_v1"
    mask_store_encodings = ["dense_uint8", "component_rle_v1"]
    mask_rle_materialized = true

  mask_rle/
    attrs:
      schema_id = "palette_mask_rle_binary_v1"
      encoded_shape_hw = [H, W]
      layout = "component_groups"
      component_names = ["subject_body", "eye_left", ...]

    components/
      00_subject_body/
        attrs:
          component_name = "subject_body"
          component_index = 0
        counts        uint32 (total_count_values_for_component,)
        indptr        int64  (N + 1,)
        present       bool   (N,)
        area_px       int32  (N,)
        bbox_xyxy     int32  (N, 4)

      01_eye_left/
        counts
        indptr
        present
        area_px
        bbox_xyxy
```

Chunking policy:

- Keep dense compatibility cache chunks as `(storage_row_chunk, 1, H, W)` using
  `refined_subject_mask_storage_chunks(...)`.
- The historical dense cache row chunk is small for interactive one-ROI review
  access. Large cluster publication should use explicit refined dense row chunk
  `256`, the current production candidate. A `512` row chunk reduces PRFS file
  count further but was slower overall in the 2026-06-20 GoodCopBadCop benchmark,
  and it increases the logical amount decompressed for random single-ROI reads.
- Chunk per-component row metadata arrays on the existing metric row grid, for
  example `(refined_subject_mask_metric_row_chunk(N),)` for `present` and
  `(refined_subject_mask_metric_row_chunk(N), 4)` for `bbox_xyxy`.
- Chunk per-component `indptr` on the same row grid plus one boundary entry;
  the driver should normally write `indptr`, not workers.
- Chunk each component's `counts` as a 1D payload array targeting large
  sequential chunks, such as 1-16 MiB of `uint32` payload, after benchmarking.
- Record requested and effective worker row chunk sizes in run attrs whenever
  parallel writers are used.
- Parallel finalizer workers must own whole physical Zarr chunks for every array
  they write. When a benchmark run increases dense mask row chunks, worker row
  chunks must be rounded to the metric/dense chunk grid; otherwise partial writes
  to compressed chunks reintroduce read-modify-write overhead and stale-overwrite
  risk.

Ragged payload writes need special care. Do not let multiple workers append to
one shared component `counts` array. Use one of these safe patterns:

- **Two-pass final array**: compute per-row/channel encoded lengths, have the
  driver prefix-sum and create per-component `indptr`, then workers write only
  their assigned non-overlapping `counts[start:end]` ranges per component.
- **Shard then reduce**: each worker writes a private temporary shard with local
  counts and metadata; the driver validates, concatenates, writes final arrays,
  and removes shards.
- **Serialized writer**: one writer appends all encoded payloads. This is
  simplest but should be a fallback for small runs or debugging.

The shard-then-reduce path is the safest first implementation because it avoids
concurrent ragged appends and matches the current row-sharded finalizer design.
Each worker should own a row shard and encode all components for that shard.
Component groups are separated in the final layout, but component-sharded
execution is not the preferred first implementation because it rereads the same
rows and complicates row-level provenance/QC joins.

## Storage Benchmark Plan

### Goals

Measure how much storage RLE would save compared with current dense Zarr mask
arrays, without changing canonical data.

Report:

- Dense logical bytes: `N * C * H * W * dtype_size`
- Dense physical bytes on disk for each current array
- RLE logical bytes: `counts.nbytes + indptr.nbytes + metadata arrays`
- RLE physical bytes if written to temporary Zarr with proposed chunks/codecs
- Compression ratio by family/run/channel
- Encode/decode timing
- Random row decode timing
- Batch decode timing for training-like access

### Scope

Run on representative approved training Zarrs:

- a small eye-mask-heavy training Zarr
- a subject-mask training Zarr
- one sleepyfish recording
- one sickyfish recording
- one older feeding training Zarr

Include families:

- `eye_masks_runs`
- `refined_eye_masks_runs`
- `subject_mask_runs`
- `refined_subject_masks_runs`

Exclude or separately report:

- `mask_probs_roi`, because probability maps are not binary masks and should not
  be evaluated as RLE candidates except after thresholding.

### Benchmark Utility

Add a read-only diagnostic:

```bash
scripts/py -m fisheye.diagnostics.benchmark_mask_rle_storage \
  <archive>.zarr \
  --families eye_masks_runs refined_eye_masks_runs subject_mask_runs refined_subject_masks_runs \
  --runs latest \
  --sample-rows all \
  --tmp-root /tmp/palette_mask_rle_benchmark \
  --json-report /tmp/mask_rle_storage_<label>.json \
  --markdown-report /tmp/mask_rle_storage_<label>.md
```

Suggested options:

- `--sample-rows all|N|fraction`
- `--channels all|0,1,...`
- `--write-temp-zarr`
- `--codec zstd|blosc-zstd|none`
- `--encode-workers 1|N`
- `--decode-benchmark-rows 1024`
- `--delete-temp`

### Output Schema

```json
{
  "archive": "...",
  "family": "refined_subject_masks_runs",
  "run": "...",
  "source_array": "masks_roi",
  "shape": [231, 1, 512, 512],
  "dtype": "uint8",
  "dense_logical_bytes": 60555264,
  "dense_physical_bytes": 1234567,
  "rle_counts_count": 12345,
  "rle_logical_bytes": 67890,
  "rle_physical_bytes": 45678,
  "dense_to_rle_logical_ratio": 891.8,
  "dense_to_rle_physical_ratio": 27.0,
  "encode_seconds": 0.12,
  "decode_rows_per_second": 5000.0,
  "notes": []
}
```

### Validation Gates

For every sampled row/channel:

- Decode RLE and compare to source dense mask exactly.
- Preserve shape and channel order.
- Preserve empty masks.
- Preserve multi-component masks.
- Confirm `area_px` equals decoded foreground count.

### Decision Criteria

RLE is worth implementing beyond benchmarking if:

- Physical storage savings are consistently meaningful after Zarr compression.
- Decode speed is acceptable for Crimson row-level access and training batch
  materialization.
- The savings remain meaningful for refined subject masks and eye masks, not
  only for unusually sparse examples.
- The implementation can be hidden behind a mask-reader API so existing tools
  are migrated incrementally.

If savings are modest after Zarr compression, prioritize reference/alias runs
for migration workflows and keep dense masks as the main training snapshot
format.

## Initial Local Component-Layout Benchmark Result

Local component-group benchmark, run on 2026-06-19:

```bash
scripts/py -m fisheye.diagnostics.benchmark_mask_rle_storage \
  /nvme1/recordings/2026-01-28T19-22-28Z_arena_2_DefaultScreen/zarr/2026-01-28T19-22-28Z_arena_2_DefaultScreen_analysis.zarr \
  --families refined_subject_masks_runs \
  --runs latest \
  --sample-rows all \
  --decode-benchmark-rows 1024 \
  --encode-workers 4 \
  --write-temp-zarr \
  --delete-temp \
  --json-report /tmp/palette_mask_rle_component_full_workers4.json \
  --jsonl-report /tmp/palette_mask_rle_component_full_workers4.jsonl \
  --markdown-report /tmp/palette_mask_rle_component_full_workers4.md
```

Result for
`refined_subject_masks_runs/refined_subject_masks_smart_finalizer_pilot_20260504`:

| surface | value |
| --- | ---: |
| shape | `23287 x 4 x 512 x 512` |
| dense logical bytes | 22.7 GiB |
| dense physical bytes | 14.1 MiB |
| RLE logical bytes | 40.9 MiB |
| component-RLE physical bytes | 3.9 MiB |
| dense-to-RLE logical ratio | 568.98x |
| dense-to-component-RLE physical ratio | 3.60x |
| encoder | process_shards/4 |
| encode rate | 797.4 rows/s |
| decode benchmark rate | 5819.4 rows/s |

Interpretation:

- Dense Zarr compression is already very effective for these masks.
- Whole-ROI RLE still produced a meaningful physical storage reduction on this
  refined subject-mask run.
- Decode speed is comfortably above current review/training row-access needs.
- Encode speed is acceptable for a finalizer-side optional compact writer, but
  should not be inserted into latency-sensitive paths until writer benchmarks
  are run on larger GoodCopBadCop-style runs.
- The process-sharded encoder is row-sharded: each worker encodes all
  components for an assigned row range, then the driver concatenates shard
  payloads and offset-adjusts each component's `indptr`. This preserves the
  canonical row/component order while avoiding unsafe concurrent appends to the
  final ragged `counts` arrays.

## Implementation Checklist

### Phase 0: Benchmark And Decision Gate

1. Implement the read-only benchmark utility.
2. Run it on representative `subject_mask_runs`,
   `refined_subject_masks_runs`, and any remaining supported eye-mask runs.
3. Report dense logical bytes, dense physical bytes, RLE logical bytes, RLE
   physical bytes, object count, encode speed, batch decode speed, and random
   row decode speed.
4. Keep dense storage as default unless whole-ROI RLE gives meaningful physical
   savings after current Zarr compression.

### Phase 1: Shared API Boundary

1. Add a `MaskStore` / `MaskSurface` API that can expose dense masks from
   multiple physical encodings.
2. Implement a dense `masks_roi` backend first with no behavior change.
3. Implement RLE encode/decode helpers with exact parity tests for empty masks,
   holes, multiple components, all-foreground masks, and every component order.
4. Add `MaskStore.read_dense(row_indices, channels=None)` as the consumer-facing
   method. Consumers should not inspect `mask_rle/` directly.

### Phase 2: RLE Writer In Shadow Mode

1. Add a writer option:
   `--mask-storage dense_uint8|rle_v1|dense_and_rle`. **Implemented for
   refined-subject finalization and exposed through the subject-mask cluster
   batch submitter.**
2. Start with `dense_and_rle` for selected runs so dense readers remain
   unaffected while RLE parity is audited.
3. Use shard-then-reduce or two-pass prefix-sum writing for `mask_rle/counts`;
   never concurrent append.
4. Validate written RLE before marking the run complete. **Implemented at the
   shared writer boundary** with explicit validation modes:
   `validation_mode="full"` decodes every compact row/channel back through
   `MaskStore` and compares it against the dense source. This is the strongest
   audit mode, but it touches the full dense logical surface.
   `validation_mode="invariants"` checks schema, row/channel shape, component
   payload presence, monotonic `indptr`, per-row RLE count sums, presence/area
   consistency, and bbox bounds without reconstructing dense masks. This is the
   production batch default. `validation_mode="none"` is reserved for low-level
   debugging and should not be used for production runs.
5. Stamp `mask_storage_encoding`, dense-cache status, chunk policy, worker chunk
   policy, and RLE schema ID in run attrs and registry extracts. The registry
   now extracts dense/RLE logical byte counts, backend-reported stored byte
   counts when available, dense-cache materialization provenance, RLE refresh
   timestamps, and stale row/component scope. Stored byte counts may be `NULL`
   for zarr backends that do not expose `nbytes_stored`.

### Phase 3: Consumer Migration

1. Migrate diagnostics and training exporters to `MaskStore` first. In progress:
   `analysis/subject_shape_runs`, refined-subject eye geometry/backfill,
   component contour generation/backfill, subject-mask training-source audit,
   subject-mask training export, subject-mask batch output validation, and the
   refined subject-mask contract validator now read through `MaskStore`.
   Recording-step status checks fall back to chunked `MaskStore` reads when
   frame-count and `metrics/mask_present` summaries are missing, so compact-only
   refined-subject runs can still report coverage.
   Subject-mask training export uses `MaskStore` only for source analysis
   reads; the merged training artifact remains dense-only and stamps
   `mask_storage_format = "dense_uint8"` plus
   `mask_storage_surface = "masks_roi"`.
   Subject-body and swim-bladder batch review selection also use `MaskStore`
   row counts for stale-row filtering, so compact-only refined runs no longer
   require dense `masks_roi` solely to bound pending review rows.
   Subject-shape overlay visualization reads component masks through
   `MaskStore`, including skeleton/contour debug overlays from compact-only
   refined runs.
   Provenance consistency diagnostics count compact refined-subject mask rows
   through `MaskStore`, so compact-only runs are not reported as row-missing
   merely because dense `masks_roi` was not materialized.
   The subject-mask/keypoint eye-coverage diagnostic also opens the resolved
   subject run through `MaskStore`; it computes component presence in row chunks
   and runs the eyes-union assignment dry-run chunked, so compact-only refined
   subject runs are no longer rejected for missing dense `masks_roi`.
   Eye-geometry source resolution exposes a dense array-like eye-mask view backed
   by `MaskStore`, so refined-subject eye geometry/export paths can consume
   compact-only refined runs without requiring dense `masks_roi`.
   Subject-body mask QC reads the `subject_body` component through `MaskStore`
   before writing unchanged `components/subject_body/qc` outputs, so compact-only
   refined runs can still receive body topology/shape QC.
   Swim-bladder patch review uses the resolved `SourceSubjectMaskRun.masks_roi`
   surface for source overlays when available and falls back to `MaskStore` for
   compact-only source masks instead of treating missing dense `masks_roi` as an
   empty overlay. It still relies on refined-subject review materialization
   before writing edits to compact-only refined runs.
   Subject-mask inspector overlay/summary reads masks through `MaskStore`, so it
   can inspect dense raw runs next to compact-only refined runs without
   requiring a dense `masks_roi` compatibility cache.
   Refined-eye compatibility materialization reads eye components from the
   refined-subject physical mask store through `MaskStore` and still writes the
   legacy dense `refined_eye_masks_runs/<run>/masks_roi` artifact for Crimson and
   historical consumers.
   Refined-source assembly can also import compact refined component sources and
   still writes a dense assembled run. Training exports still write dense
   `subject_mask_runs/<run>/masks_roi` artifacts. Refined subject-mask review
   can open compact-only runs by materializing a dense `masks_roi`
   compatibility cache first; it does not yet edit compact RLE in place.
   The non-UI `fisheye.refinement.refine_subject_masks` dry-run/planning path
   inspects existing refined runs through `MaskStore` without materializing
   dense masks, while the actual apply path remains an edit boundary and uses
   the shared refined-review opener to materialize `masks_roi` before writeback.
   `fisheye.utils.merge_subject_mask_runs` accepts source subject-mask runs
   backed by either dense `masks_roi` or compact `mask_rle` through `MaskStore`,
   while intentionally writing a dense merged `subject_mask_runs/<run>/masks_roi`
   output because merged raw subject-mask runs remain training/export-friendly
   dense artifacts.
   `fisheye.refinement.finalize_subject_masks` can consume source
   `subject_mask_runs` backed by compact `mask_rle` when neither dense
   `masks_roi` nor `mask_probs_roi` is present, and records `mask_rle` as the
   source surface in component provenance.
   The SAM subject-prompt visualizer loads optional source body overlays through
   `MaskStore`, so compact raw subject-mask runs can be inspected without
   materializing dense `masks_roi`.
2. Migrate remaining review UI paths and compact-RLE consumer paths next.
   Crimson's dense refined-subject-mask reader now resolves the run and places
   masks through `source_crop_row_ids`; its remaining migration is compact
   `mask_rle` decode for dense-free `rle_v1` runs.
3. Migrate edit/writeback paths only after read paths are stable; **implemented
   for Palette-owned refined-subject dense edits with a `mask_bitpacked`
   mirror**. The edit path refreshes touched bitpacked row/channel cells from
   dense immediately and marks compact RLE stale for explicit refresh.
4. Add a materializer command that can create, refresh, or delete dense
   `masks_roi` compatibility caches from compact masks, and can refresh compact
   `mask_rle` or compact `mask_bitpacked` from the current dense cache after
   review/edit.
   **Implemented for refined-subject runs**:
   `scripts/py -m fisheye.utils.materialize_refined_subject_mask_store`.
   The refined subject-mask contract validator also uses `MaskStore` for
   validation and conservative backfill: missing run-level metrics and
   per-component metric mirrors can be repaired from compact `mask_rle` without
   recreating dense `masks_roi`.
   Stage-completion array validation now treats compact `mask_rle` as a valid
   refined-subject physical mask store when the parent shape metadata and each
   component pointer table pass cheap structural checks (`indptr[0] == 0`,
   monotonic pointers, and `indptr[-1] == len(counts)`).
5. Add registry/audit fields for compact encoding, dense cache presence,
   encoded byte size, dense cache byte size, and materialization freshness.

Remaining default-migration blockers:

- Crimson compact-RLE readers must either implement Palette's
  `MaskStore`-equivalent materialization contract or explicitly materialize
  dense masks before display/edit. Crimson's dense refined-subject-mask path is
  no longer a blocker for `dense_uint8` or `dense_and_rle` runs.
- Legacy eye-mask training/export paths remain dense by design and should not
  be used as proof that refined-subject compact storage is fully deployed.
- Remaining readme/workflow snippets that index
  `refined_subject_masks_runs/*/masks_roi` directly should be updated or
  clearly labeled dense-only compatibility examples.
- Operators need a real-recording smoke for `--mask-storage rle_v1` followed by
  subject-shape, eye-geometry, component-contour, training-export, and review
  materialization checks before `rle_v1` becomes the default.
- Run the cluster smoke in two steps: first `--mask-storage dense_and_rle` to
  validate compact RLE while preserving dense compatibility, then
  `--mask-storage rle_v1` to validate compact-only publication and consumers.

### Phase 4: Default Writer Migration

1. Make compact RLE the default only for new large refined subject-mask runs
   after training exporters/promoters, review, Crimson, and diagnostics all
   read analysis sources through `MaskStore`; training artifacts/loaders remain
   dense-only unless a future training-artifact contract explicitly changes
   that boundary.
2. Keep probability arrays dense or quantized; do not RLE `mask_probs_roi`.
3. Keep dense training snapshots available when throughput matters more than
   storage size.
4. Treat existing dense runs as valid legacy data; do not rewrite them until a
   reader-compatible backfill tool exists.

### Phase 5: Optional Tight-BBox RLE v2

1. Add `encoded_bbox_xyxy_roi` in v1 so bbox distributions are known before
   changing the physical encoding.
2. Prototype bbox-local RLE as a new schema ID, not as a silent change to v1.
3. Require parity against whole-ROI dense masks after paste-back.
4. Promote bbox-local RLE only if it materially improves storage or transfer
   size beyond whole-ROI RLE.
