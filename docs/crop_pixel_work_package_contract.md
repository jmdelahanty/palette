# Crop Pixel Work-Package Contract

<!-- contract-meta
version: 1
status: draft
implementation: partial
schema: palette.crop_pixel_work_package v1
last_updated: 2026-07-29
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

The source binding and the pixel source are independent fields. New packages
use `immutable_crop_run_manifest_v1` for strict crop-v2 runs or
`signed_crop_run_v1` for maintained signed acquisition sources that have not
yet migrated to the strict manifest envelope.
`legacy_crop_signature_revision_v1` remains the explicit compatibility binding
for packages made from older crop runs. In particular, a current acquisition
crop video is bound by `signed_crop_run_v1`; this does not make its pixels
legacy.

Current exact pixel-source profiles include:

- `source_pixels=raw_camera_video`: decode a full-frame camera video and apply
  the bound crop geometry;
- `source_pixels=acquisition_crop_video`: decode the already-cropped Orange
  frame and retain crop-video/crop-meta lineage.
- `source_pixels=hybrid_acquisition_crop_video_offline_supplement`: route each
  row through `source_pixel_kind_codes` to one of those two current sources.

They may share PyNvVC luma value semantics, but they are not substitutable.

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
  geometry, immutable run reference or signed-run identity, and the exact
  authoritative pixel-source contract still match the bound crop run.

Package creation requires modern `instance_key` and either persisted
`source_row_signature` arrays or the verified auxiliary-proxy bootstrap defined
in `keyed_downstream_compaction_contract.md`. Package consumers persist the
exact selected bootstrap signatures, so later compaction validates the rows
actually used. Legacy `refined_roi_path` overrides are rejected: those pixels
must be folded into the logical crop definition before a shared package can be
cited.

Maintained acquisition crop-video producers are not a legacy exception. New
`palette.analysis_acquisition_crop_run.v1` outputs mint stable detection-origin
`instance_key` values and persist `source_row_signature` directly. Their signed
current-source reference binds the Orange PyNvVC luma pixel contract, crop-video
descriptor, crop-meta row/video-frame identities, geometry, and complete rowset
fingerprint. Historical acquisition runs that lack those arrays remain readable
but cannot create a modern work package until they are republished or migrated.

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
`incremental_materialization_role=delta_replacement_rows` by default. Ordinary
collection finalizers reject that role because a keyed edit/inference subset is
not a complete collection partition.

There is one explicit maintained exception for clipped or whole-recording
parallelization. A video-window package may be passed with
`--roi-work-package-role complete_collection_partition` only when the writer
can prove, before inference, that:

- the package uses
  `global_crop_rows_from_authenticated_acquisition_video_window_v1` semantics;
- its acquisition frame-window binding is exact and matches the declared clip;
- its selected global crop rows are one contiguous interval;
- that interval is exactly
  `frame_row_offsets[start_frame:end_frame_exclusive]` from the authoritative
  recording-level crop run;
- every selected acquisition frame lies within the bound window; and
- collection, clip, work-unit, and shard identities are all present.

The writer persists a digest-bound
`palette.subject_mask.complete_collection_partition` contract and stamps
`canonical_finalization_policy=collection_shard_finalization_allowed`. The
collection finalizer independently recomputes the contract digest, compares all
collection attributes, and verifies the persisted row and frame arrays. A role
label without this proof is rejected. This exception supports complete
recording partitioning; it does not weaken the default delta/compaction path.

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

## Bounded Acquisition-Source Canary

The first real producer/consumer checkpoint uses the maintained RedScare
acquisition crop video without modifying its source archive. A GPU-node job:

1. creates a minimal analysis archive on node-local scratch;
2. republishes acquisition crop metadata through the current writer;
3. materializes a deterministic prefix of at most 2,048 crop rows once;
4. reopens that package through both keypoint and subject-mask input paths;
5. optionally runs both real model consumers into selector-ineligible shard
   parents in the local archive; and
6. publishes only strict JSON evidence and logs under `.palette_benchmarks`.

Use `scripts/submit_crop_pixel_materialization_canary_bsub.sh`. It is dry-run
unless `--apply` is supplied, requires a clean commit-pinned cluster worktree,
rejects shared mounts as compute scratch, disables registry writes, and records
that production selectors and archives were unchanged. The local work package
and model outputs are ephemeral integration evidence, not training artifacts.

This checkpoint precedes the keypoint storage contract. Its row identity,
coordinate, pixel-source, and edit-lifecycle evidence informs that contract; it
does not select keypoint chunk or shard sizes by itself.

The resulting keypoint/body-frame/QC boundary is documented in
`docs/keypoint_storage_contract_v2.md`.

The real RedScare canary passed as LSF job `153227442` at Palette commit
`229ceadd600b27c384684e474fe3940fd077ac13`. Its immutable receipt is:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
crop_pixel_materialization/workflows/
20260729_redscare_acquisition_crop_consumers_229ceadd_v4/receipt.json
```

The receipt SHA-256 is
`8fa5ec642b34e1f365ae6b24e2513cc5e06d213e10acb59e8694e590f06fb0fe`;
its recomputed internal evidence digest is
`252ffab529bfdf771560bc64d368c63369e5e4a8e6a1680ded3b435977b66c15`.
The canary materialized `2,048` `384x384` rows (`301,989,888` bytes) in
`1.880 s`. The complete worker/publish/package setup took `13.411 s`, YOLO pose
inference took `9.5 s`, unified subject-mask inference took `24.1 s`, and the
complete workflow took `65.820 s`. Peak child RSS was `2,554,011,648` bytes.

Both real consumers preserved the same `source_crop_row_ids`, unique
`instance_key`, `source_row_signature`, and package ID. They independently read
the same pixel digest. The source root metadata hash was unchanged, and no
selector, registry, or production archive state changed.

Two fail-closed canary findings were corrected before that pass:

- PyNvVC materialization now runs in a short-lived worker process so its
  exclusive CUDA context is released before model consumers start; and
- immutable output validation recognizes only the exact declared
  shared-columnar short-array optimization, including requested and effective
  shard shapes, while still rejecting undeclared ordinary arrays.

## Synthetic Evidence

`tools/benchmark_crop_pixel_work_package.py` compares a complete package with a
selected-row package while preserving the same validation work. On the
2026-07-18 workstation run (`2,048` rows, `32x32` ROIs, three repetitions), a
20-row package wrote `20,480` pixel bytes instead of `2,097,152` bytes, a
`102.4x` payload reduction. Median build times were about `8.4 ms` versus
`28.1 ms`. This proves subset scaling and is not a PRFS or model-throughput
claim. The later real canary above supplies compute-node source-read and GPU
fan-out evidence; this synthetic comparison remains useful only for
subset-scaling isolation.
