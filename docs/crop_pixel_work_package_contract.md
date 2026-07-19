# Crop Pixel Work-Package Contract

<!-- contract-meta
status: implemented-local
schema: palette.crop_pixel_work_package v1
last_updated: 2026-07-18
owner: jeremy
depends_on: docs/composite_crop_storage_contract.md,
  docs/instance_track_subject_identity_contract.md,
  docs/dask_zarr_write_safety.md
-->

## Purpose

`crop_runs/<run>` remains the complete immutable logical crop authority: it
defines observation identity, frames, geometry, source signatures, pixel
provenance, and the pixel contract. A crop pixel work package is a durable,
subset-only realization of those pixels for downstream incremental inference.
It is not a crop run, a new authoring authority, or a canonical stage selector.

This separation permits one changed-row package to feed keypoint and
subject-mask inference concurrently without persisting a second complete crop
array or asking both consumers to decode and crop the same source frames.

## Files And Identity

One logical package consists of a JSON manifest plus generation-specific files:

```text
<name>.json
<name>.<package-id-prefix>.<generation>.bin
<name>.<package-id-prefix>.<generation>.rows.npz
```

The raw C-order `uint8[D,H,W]` binary contains only the selected rows. The row
index contains:

```text
crop_row_indices       int64[D]
instance_key           uint64[D]
source_row_signature   uint8[D,32]
frame_indices          int64[D]
roi_coordinates_full   int32[D,2]
pixel_sha256           uint8[D,32]
```

Rows are unique and ordered by ascending source crop row. `package_id` is a
stable SHA-256 identity over the crop binding, selected crop rows, keys, source
signatures, per-row pixel digests, and pixel contract. It remains stable across
equivalent retries; generation filenames may differ.

The manifest is published last with atomic replacement. Payload objects are
generation-specific, so a failed overwrite cannot make the previous complete
manifest refer to partially replaced data. Failed generations may leave
unreferenced files and are safe to remove during explicit work-package cleanup.
Cleanup must never remove the files referenced by a live manifest or by a
running downstream job.

## Validation

Opening a package fails closed unless:

- schema, layout, status, dtype, order, shape, and pixel contract are valid;
- manifest and row-index digests match;
- the binary size and full-payload digest match;
- every per-row pixel digest matches;
- crop rows and `instance_key` values are unique;
- `package_id` recomputes exactly; and
- when the source archive is available, keys, row signatures, frame indices,
  geometry, crop signature/revision, signature-spec digest, and source pixel
  fingerprint still match the bound crop run.

Package creation requires modern `instance_key` and `source_row_signature`
arrays. Legacy `refined_roi_path` overrides are rejected: those pixels must be
folded into the logical crop definition before a shared package can be cited.

## Consumer And Publication Rules

Package row zero is not crop row zero. Every package-backed output must write:

```text
source_crop_row_ids[i] = package.crop_row_indices[i]
instance_key[i] = crop.instance_key[source_crop_row_ids[i]]
```

Frame counts are recomputed over the selected rows while retaining the complete
recording frame domain. Keypoint and mask consumers stamp the package ID and
manifest path in attrs and stage provenance.

Package-backed inference may write only noncanonical shard parents:

```text
keypoint_shard_runs/<run>
subject_mask_shard_runs/<run>
```

It may not directly write `keypoints_runs` or `subject_mask_runs`, update latest
pointers, or publish registry success. These runs are stamped
`incremental_materialization_role=delta_replacement_rows`. Ordinary collection
finalizers reject that role because a subset is not a complete collection
partition.

A keyed incremental compactor combines compatible rows from the exact prior
complete base with replacement rows from these shards, order the result on the
target crop rowset, validate complete `source_crop_row_ids` and `instance_key`
coverage, and publish a new complete snapshot. Keypoints are published as a
bounded, physically standalone sharded snapshot. Raw probability masks use a
depth-one immutable base-plus-delta snapshot so unchanged probability pixels
are not decoded and rewritten. See
`docs/keyed_downstream_compaction_contract.md`.

Subject-mask inference does not depend on keypoints. Both inference branches
may therefore fan out from the same completed package. Eye assignment and any
other keypoint-bound mask refinement bind the complete refined-keypoint run at
finalization, after both branches have completed.

## DAG Lifecycle

The intended incremental path is:

```text
keyed materialization plan
  -> build and validate one crop pixel work package
      -> keypoint shard inference -----------+
      -> subject-mask shard inference -------+ parallel
  -> keyed raw keypoint/mask compaction with prior complete bases
  -> keypoint refinement/compaction
  -> subject-mask refinement/compaction (waits for keypoints when required)
  -> validate complete canonical snapshots
  -> atomically promote
  -> delete unreferenced package generations after all dependents finish
```

Retries reuse a valid package with the same selected rows. A package is retained
until every submitted consumer has reached a terminal state and any retry window
has closed. It remains reproducible from the logical crop run and source pixels,
so it is workflow scratch rather than permanent canonical analysis data.

## Operator Interface

Dry-run is the default:

```bash
scripts/py -m fisheye.utils.build_crop_pixel_work_package \
  /path/to/recording_analysis.zarr \
  --crop-run <exact-crop-run> \
  --crop-rows-npy /path/to/target_crop_rows.npy \
  --manifest /workflow/work-packages/edit-17.json
```

Add `--apply` to write and validate the package. Production apply operations and
inference belong in LSF compute jobs, never on a login node.

After all dependent jobs are terminal, cleanup is also dry-run-first:

```bash
scripts/py -m fisheye.utils.cleanup_crop_pixel_work_package \
  /workflow/work-packages/edit-17.json
```

Add `--apply` only after reviewing the unreferenced generation list.

## Synthetic Evidence

`tools/benchmark_crop_pixel_work_package.py` compares a complete package with a
selected-row package while preserving the same validation work. On the
2026-07-18 workstation run (`2,048` rows, `32x32` ROIs, three repetitions), a
20-row package wrote `20,480` pixel bytes instead of `2,097,152` bytes, a
`102.4x` payload reduction. Median build times were about `8.4 ms` versus
`28.1 ms`. This proves subset scaling and is not a PRFS or model-throughput
claim; the production canary still needs compute-node source reads and GPU
fan-out timing.
