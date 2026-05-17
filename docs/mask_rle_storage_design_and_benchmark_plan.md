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

## Is Zarr a Poor Fit for RLE?

Not necessarily. Zarr is strongest for typed arrays, and RLE can be represented
as typed arrays instead of JSON-per-row blobs.

A Zarr-native COCO-style RLE layout can use flat payload arrays plus offsets:

```text
mask_rle_counts        uint32  (total_run_count,)
mask_rle_indptr        int64   (N * C + 1,)
mask_present           bool    (N, C)
mask_shape             attrs: [H, W]
mask_encoding          attr: "coco_rle_fortran_v1"
mask_value_semantics   attr: "binary_0_1"
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

- Training datasets should consume the mask API, not direct `masks_roi` arrays.
- Exporters can choose the output policy:
  - dense training snapshot for maximal compatibility;
  - compact authoritative snapshot plus materialization cache;
  - compact source plus dense exported artifact.
- The trainer still receives dense tensors.

### Diagnostics, Profiles, and Metrics

Dense scans are present in profile/audit utilities such as:

- `src/fisheye/utils/eye_mask_profile.py`
- `src/fisheye/utils/audit_subject_mask_training_sources.py`
- `src/fisheye/diagnostics/check_eye_masks.py`
- `src/fisheye/diagnostics/check_mask_components.py`

Needed change:

- Metrics should compute from `MaskStore.read(...)` or from precomputed compact
  metrics such as area/bbox when available.
- Storage benchmarks should report both physical Zarr size and logical decoded
  mask shape.

## Proposed RLE Schema

For each mask run, support either dense arrays or a compact `mask_rle/` group.

```text
<mask_run>/
  attrs:
    mask_storage_encoding = "rle_binary_v1"
    mask_encoding_order = "fortran"
    mask_value_semantics = "binary_0_1"
    masks_roi_materialized = false

  mask_rle/
    counts                  (total_run_count,) uint32
    indptr                  (N * C + 1,) int64
    shape                   (2,) int32          # [H, W]
    row_channel_present     (N, C) bool
    row_channel_area_px     (N, C) int32        # optional but useful
    row_channel_bbox_xyxy   (N, C, 4) int32     # optional but useful
```

Compatibility rules:

- If `masks_roi` exists, legacy readers may use it.
- If `masks_roi` is absent and `mask_rle/` exists, modern readers must
  materialize dense masks on demand.
- Writers must record whether `masks_roi` is authoritative or a cache.
- Probability arrays remain dense/quantized arrays and are not represented as
  binary RLE.

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

## Recommended Order of Work

1. Implement the read-only benchmark utility.
2. Run it on the approved migrated training Zarrs and summarize by family.
3. Decide whether RLE, bitpacking, or reference/alias runs give the best return.
4. Add a `MaskStore` reader API with dense and RLE backends.
5. Migrate training loaders to `MaskStore`.
6. Migrate Crimson/review read paths to `MaskStore`.
7. Add write support and copy-on-write semantics for edited compact masks.
8. Only then consider making compact binary masks a default writer policy.
