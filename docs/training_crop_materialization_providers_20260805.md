# Training Crop Materialization Providers

Date: 2026-08-05; implementation checkpoint updated 2026-08-06

Status: implemented selector-ineligible composed publication contract;
production registration and keypoint/mask review-run activation remain gated
by an end-to-end Batman training canary.

## Decision

A materialized training crop is defined by its logical arrays, pixel contract,
geometry binding, and stable row identity. It is not defined by how its pixels
were obtained.

Palette supports three first-class pixel-materialization providers:

1. `source_video_pynvvc_luma` decodes the authoritative source video with the
   canonical PyNvVideoCodec luma contract and crops the requested rows.
2. `verified_flat_roi_cache` verifies an exact flat-cache manifest, source
   crop binding, shape, pixel contract, and payload SHA-256 before copying its
   rows.
3. `sampled_training_images_full` crops the reviewed positive instances from
   the training Zarr's own lossless `raw_video/images_full` surface. It binds
   the compact sampled-frame axis, sparse acquisition-frame mapping, stable
   `instance_key`, and complete frame-decision digest without pretending that
   an external crop-v2 row already exists.

The cache provider is an optimization. It is not an authority, and it is not a
required input. New training Zarrs may be built directly from source videos
without ever creating a flat cache.

All providers produce a self-contained `crop_runs/<run>/roi_images` array.
After successful materialization, readers and training jobs do not need the
source cache or source video to read those pixels. The selected provider and
its exact inputs remain recorded as provenance.

## Shared Output Contract

Every output written by either provider has:

- `zarr_purpose = "training"` on the destination archive;
- `crop_storage_mode = "materialized"`;
- dense `uint8` `roi_images` with the canonical crop ROI layout;
- exact geometry and lineage arrays for the selected source observations;
- stable `instance_key` values when the source crop contract provides them;
- `source_crop_row_ids` for providers backed by an external crop-v2 authority;
- `source_frame_indices` binding crop rows to source acquisition frames;
- `training_materialization_schema =
  "palette.training_crop_materialization.v1"`;
- `training_materialization_provider` equal to one of the three exact provider
  identifiers above;
- the canonical Orange mono PyNvVideoCodec-luma pixel contract;
- `stage_selector_eligible = false` until the enclosing training publication
  passes validation and is activated explicitly.

The cache provider additionally records the cache manifest and payload
digests. Cache bytes are copied into the destination Zarr; deleting the
ephemeral cache after publication cannot invalidate the training artifact.

Strict materialized inputs also carry the digest-protected
`training_crop_materialization_binding`. It declares every copied geometry
array, hashes the lightweight identity arrays, records the exact provider
evidence, and is checked through both direct and consolidated metadata. Opening
the binding does not reread and hash all ROI pixels. Atomic publication hashes
the physical files, while the cache provider has already authenticated its
logical flat payload.

The sampled-images provider has a separate logical geometry validator because
recording crop-v1 and sampled training data intentionally have different frame
axes. `frame_indices` indexes the compact training image axis;
`source_acquisition_frame_index` and `source_frame_indices` carry the sparse
acquisition identity. Its F+1 offset array indexes the compact local axis. The
provider validates these mappings, exact float32 detection projections, crop
placement, fixed ROI extent, zero-padding policy, decision digest, and all
identity dtypes without weakening recording-level crop-v1.

## 2026-08-06 Sampled-Images Canary

The first Batman canary used 200 reviewed full-resolution frames and passed:

- 181 reviewed positive instances and 19 explicit negative frames;
- 181 unique source-preserved `uint64 instance_key` values;
- dense `uint8 [181,348,348]` crop pixels;
- three explicitly zero-padded edge crops;
- zero byte differences across an exhaustive crop-to-`images_full` check;
- exact local-frame to acquisition-frame mapping;
- direct and consolidated materialization-binding validation; and
- no crop selector or registry activation.

The initial explicit assembly remained node-local. The reusable publisher was
then exercised from clean implementation commit
`2540817883d7122acbb26a72e3b8d09249f73b3b` and published the checked,
selector-ineligible artifact at:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/training/
batman_training_canary_20260806_v1/
2026-07-21T19-38-32Z_arena_2_Batman_reviewed_crops_training_v1.zarr
```

The final strict materialization-binding digest is
`e73d1b84de3a7e9eb77de3007a53acc82c8eee565983eed9595e477dc415a3b7`.
The final direct/consolidated reopen confirmed 181 unique keys, terminal offset
181, the final archive path as the self-contained source identity, no crop
selectors, `training_artifact_status = complete`, and deferred registry
activation. No hidden publication sibling or node-local temporary artifact
remained. This is benchmark/canary evidence, not a registered or selected
training publication.

## First-Class Detection Review

Crop materialization enriches a sampled training dataset; it does not replace
the full-frame detection-review lane. The composed publisher requires all of:

- `raw_video/images_full`;
- `raw_video/images_ds`;
- `raw_video/original_frame_indices`;
- one complete selected `detect_runs/<run>`;
- one complete selected `refined_detect_runs/<run>` approved for training; and
- exact `uint64 instance_key`, `float32 bbox_norm_coords`, and sampled-frame
  indices on both detection tables.

Detection review therefore happens before crop materialization. Before writing
a crop, the publisher joins every selected crop observation to
the refined detection review by stable `instance_key`, original acquisition
frame, and exact normalized box. This preserves multiple fish in one sampled
frame and prevents the historical one-row-per-frame promoter from silently
collapsing observations.

The persisted `training_dataset_composition` receipt binds the full-frame
detection-review surface and the crop-review surface together. Keypoint and
subject-mask candidates extend that receipt after their reviewed finalizers
are implemented.

The storage composition and maintained task/session browser detection reviewer
support multiple `instance_key` rows per sampled frame. The browser receives
`instance_key` as a decimal string (never a JavaScript number), submits the
complete detection collection for the server-selected frame, and uses a null
key only for a new observation. The server rejects duplicate or foreign-frame
keys, preserves surviving identities, allocates new identities in the curated
writer, and treats omitted existing keys as deletions. Empty frames remain
valid review targets. The historical standalone frame reviewer and legacy
dense compatibility runs remain single-box adapters; they do not weaken the
multi-instance task workflow or cause storage rows to be collapsed.

## Provider Examples

Materialize existing training-crop geometry directly from its source video:

```bash
scripts/py -m fisheye.utils.regenerate_training_crops_pynvvc \
  /path/to/recording_training.zarr \
  --source-crop-run crop_geometry \
  --target-crop-run crop_training_pixels \
  --video-path /path/to/source.mp4
```

Reuse a verified cache whose geometry is owned by an analysis Zarr:

```bash
scripts/py -m fisheye.utils.regenerate_training_crops_pynvvc \
  /path/to/recording_training.zarr \
  --source-zarr-path /path/to/recording_analysis.zarr \
  --source-crop-run crop_v2 \
  --target-crop-run crop_training_pixels \
  --roi-cache-manifest /path/to/cache.flat_roi_cache.json
```

Create a new selector-ineligible training Zarr on node-local scratch and
publish the complete archive through a checked hidden sibling:

```bash
scripts/py -m fisheye.utils.publish_training_crop_materialization \
  /path/to/batman_training.zarr \
  --create-artifact \
  --base-training-zarr /path/to/batman_sampled_detection_review_training.zarr \
  --source-zarr /path/to/batman_analysis.zarr \
  --source-crop-run crop_v2 \
  --run-id crop_v2_training \
  --scratch-root /scratch/$USER/palette-training/job-001 \
  --roi-cache-manifest /path/to/cache.flat_roi_cache.json \
  --source-instance-keys 101,102,205
```

When the reviewed sampled artifact already contains lossless full-resolution
frames, materialize its complete positive rowset directly and retain explicit
negative frames only in frame supervision:

```bash
scripts/py -m fisheye.utils.publish_training_crop_materialization \
  /path/to/batman_reviewed_crops_training.zarr \
  --create-artifact \
  --base-training-zarr /path/to/batman_reviewed_training_base.zarr \
  --sampled-images-full \
  --refined-detect-run refined_detect_reviewed \
  --run-id crop_reviewed_348_images_full_v1 \
  --roi-size 348 \
  --scratch-root /scratch/$USER/palette-training/job-002
```

This route refuses incomplete frame decisions, preserves every positive
`instance_key` (including multiple rows in one frame), writes no placeholder
crop for a negative frame, validates every crop byte against
`raw_video/images_full`, consolidates once after all writes, verifies the
hidden destination copy, and leaves the crop family unselected.

Omit `--source-instance-keys` only when the sampled full-frame base covers the
complete crop-v2 rowset. For an ordinary sampled training Zarr, pass the keys
belonging to its sampled and reviewed detections. Selection is by stable
observation identity, not frame ordinal. Two selected fish in one frame
therefore remain two rows, and the output `frame_row_offsets` is recomputed for
that selected rowset.

With `--create-artifact`, the base sampled training Zarr is copied to
node-local scratch, enriched, validated, and published as one checked new
archive. This retains its full/downsampled frames and detection/refined-
detection review runs. Crop-only whole-artifact creation is intentionally not
supported.

Without `--create-artifact`, the same command requires those detection-review
surfaces in the destination and atomically appends one immutable crop run.
Neither mode updates a stage selector or the registry.

For the lower-level in-place `regenerate_training_crops_pynvvc` command, use
`--dry-run` first. Dry-run opens the destination read-only and does not create
groups or metadata. The atomic publisher instead refuses replacement and keeps
the destination absent until the checked rename.

## Relationship to the Training Workflow

This provider choice is independent from training-Zarr creation and label
promotion:

1. A sampled training import stores `raw_video/images_full`,
   `raw_video/images_ds`, and `raw_video/original_frame_indices`.
2. Detection and refined-detection candidates are reviewed and approved for
   training in that same training Zarr. They remain the first downstream
   review surface.
3. Crop pixels are materialized by either provider above and joined by stable
   row identity and source-frame identity.
4. Keypoint and subject-mask inference writes non-authoritative candidates.
5. Review and promotion create the accepted training authorities.

Materializing crop pixels does not itself promote detections, keypoints, or
masks.

The exact materialized training input is wired into the maintained producers
with `--require-training-materialization-binding`. That option intentionally
permits only terminal `keypoint_shard_runs` or `subject_mask_shard_runs` output.
Canonical keypoint-v2 and subject-mask publication must still finalize against
the original crop-v2 authority; copied training pixels do not become a second
coordinate authority.

## Fail-Closed Boundaries

- An external crop-v2 manifest is not copied and relabelled as if it were
  authoritative inside the training archive. The training group records a
  source binding instead.
- A flat cache with the wrong archive, crop run, shape, pixel contract, or
  payload digest is rejected before its pixels are used.
- The historical frame-axis detection promotion backend supports one review row
  per source frame. It now rejects duplicate frames instead of silently
  collapsing multiple observations. General multi-instance promotion must join
  on `instance_key`. New crop-v2 training construction does exactly that and
  does not route multi-subject data through the old frame writer.
- The new-artifact publisher refuses an existing destination, builds on bounded
  node-local scratch from a sampled detection-review base, verifies a complete
  physical copy in a hidden sibling, and atomically renames the whole training
  Zarr.
- Crop enrichment is refused when sampled full frames or complete detection
  review runs are missing, or when any crop key/frame/box join differs.
- Strict keypoint/mask training input produces terminal candidates only. A
  source-authority-bound finalization step is still required before those runs
  become reviewable canonical/refined surfaces.

## Implementation Checklist

- [x] Preserve direct source-video PyNvVideoCodec materialization.
- [x] Add verified external flat-cache materialization.
- [x] Add sampled `images_full` provider identity and strict sampled-axis
      binding validation.
- [x] Require canonical merged detection exports to preserve exact stable
      `instance_key` values and fail closed when they are missing or collide.
- [x] Use one provider-neutral training materialization schema.
- [x] Record an exact provider identifier and the supported-provider contract.
- [x] Copy cache pixels into the destination Zarr.
- [x] Preserve multiple crop rows and stable `instance_key` values.
- [x] Reject tampered cache payloads.
- [x] Make dry-run destination access read-only.
- [x] Reject duplicate-frame detection promotion before identity collapse.
- [x] Add instance-key-based multi-instance crop selection and label lineage.
- [x] Add the strict training source-binding adapter for keypoint and mask
      candidate production.
- [x] Add node-local atomic crop-run publication for existing training Zarrs.
- [x] Add checked whole-training-artifact publication for new training Zarrs.
- [x] Preserve sampled full/downsampled frames and detection review as
      first-class surfaces in whole-artifact publication.
- [x] Bind detection review and crops by instance key, source frame, and box.
- [x] Run one exhaustive selector-ineligible Batman sampled-images crop
      canary on node-local scratch.
- [x] Route the sampled-images writer through the reusable atomic whole-
      training-artifact publication CLI, with bounded node-local construction,
      complete positive/negative supervision, exhaustive crop-byte checks,
      consolidated binding validation, and checked hidden-sibling rename.
- [ ] Add reviewed canonical/refined keypoint and subject-mask finalization for
      the training archive; do not relabel copied crop geometry as crop-v2.
- [ ] Register only a completed canary artifact after its review surfaces pass.
- [ ] Run one selector-ineligible Batman training canary through detection,
      keypoint, and subject-mask review before production activation.
