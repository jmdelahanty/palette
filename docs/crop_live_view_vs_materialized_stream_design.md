# Live Crop Views vs Materialized Crop Streams

Date: 2026-03-06
Type: Decision note / phased migration plan (not a contract)

## Context

`Palette` currently materializes crop pixels under `crop_runs/<run>/roi_images`
and many downstream consumers read those arrays directly. The design question is
whether crop pixels should remain materialized by default, become optional, or
move into a different datastream shape.

## Decision

Current recommendation:

- Keep materialized crops (`crop_runs/<run>/roi_images`) as the default and
  production-safe mode for now, even though warm-cache geometry-only inference
  is now close to parity on the first smoke benchmark.
- Do not allow a geometry-only crop run to become `crop_runs.latest` until
  reader migration is complete.
- Treat bulk ROI inference on large full-frame sources as a cache-backed
  workflow, not a pure live-decode workflow.
- Do not introduce a separate crop-image datastream right now.
- Keep traditional detection/keypoint pipelines explicitly
  imported/materialized-only.
- Treat live cropping as a compatibility project first, then revisit whether it
  should become the default.

## Why This Is The Current Decision

### 1. Mixed-mode inference exists, but the broader reader set is not migrated

YOLO keypoint inference, YOLO eye-mask inference, and keypoint retry now have a
shared mixed-mode crop reader. But many other consumers, validators, and
workflow assumptions still expect materialized ROI tensors. A geometry-only run
therefore cannot become the default/latest crop run yet without breaking
non-migrated readers.

### 2. Training and export are still coupled to per-recording crops

Current training/export prep for pose and eye-mask workflows still expects
per-recording crop pixels to exist. Even when the final consumer is a merged
training artifact, the upstream export/build path still reads
`crop_runs/<run>/roi_images`.

### 3. Viewers, tuners, and review tools are heavy ROI readers

This is not just a training concern. Many review/tuning tools repeatedly or
randomly access ROI pixels, so live cropping would require broader reader
migration and latency validation than a simple pipeline flag suggests.

### 4. Writer/schema contracts still define crop pixels as part of the stage

The crop writer, crop-stage array spec, and diagnostics currently assume
`roi_images` is part of a healthy crop run. A geometry-only mode is therefore a
schema and compatibility change, not just a runtime option.

### 5. A separate crop-image datastream would add coordination cost

The crop run already stores the geometry, source provenance, and crop signature
needed to identify ROI tensors. Splitting pixels into a separate top-level
datastream now would create extra lineage/latest-run coordination before the
reader compatibility problem is solved.

### 6. Large full-frame decode can dominate ROI inference throughput

For large source videos, live crop reconstruction can be functionally correct
but still too slow for bulk ROI-model inference. In practice, full-frame decode
from `4512x4512` source video may run around `90 FPS`, while keypoint/eye-mask
inference over materialized ROI tensors can run closer to `300 FPS`.

That means a geometry-only archive design still needs a fast path for repeated
ROI inference. The right answer is a temporary ROI cache, not a second
permanent canonical crop-image datastream.

### 7. Traditional pipelines are intentionally materialized-only

Not every consumer should become mixed-mode/live-crop capable. Traditional
detection and traditional keypoint inference are designed around imported or
materialized image arrays and should fail clearly when pointed at archives that
do not satisfy that contract.

## Target Architecture

The long-term direction can still be:

- `crop_runs/<run>` remains the canonical crop-stage anchor for geometry and
  provenance.
- `roi_images` becomes optional in the future, with storage mode made explicit.
- Live crop reconstruction uses existing crop metadata plus full frames or
  recorded `source_video_path`.
- Temporary ROI caches can be created as runtime accelerators for decode-limited
  workflows.
- Materialization remains available as an explicit cache/training artifact.

This means the main architectural choice is not "new datastream vs old
datastream". It is "shared crop identity/provenance in `crop_runs`, with pixel
materialization optional once readers support it."

## Why Crop Geometry/Provenance Still Matters Beyond Detections

Detection outputs answer "where is the fish?" Crop-stage metadata answers
"what exact ROI patch did downstream stages use, and how do we reproduce or
map it back?"

That distinction matters because a detection bbox alone does not fully define
the ROI tensor used by downstream stages:

- A bbox does not capture the full crop policy.
  The crop stage also depends on ROI size, centering rules, clipping/padding,
  source resolution, and source-selection lineage.
- Downstream predictions are usually in ROI coordinates, not full-frame
  coordinates.
  Keypoints and masks need the ROI offset and shape to be mapped back into full
  image space.
- Row identity and alignment matter.
  `crop_runs` provides a stable mapping between crop rows and detection/frame
  rows via arrays like `frame_indices`, `frame_counts`, and
  `detection_indices`.
- Reproducibility and regeneration matter.
  Provenance such as `source_detect_run`, `detection_source_path`,
  `detection_source_type`, `source_video_path`, `roi_size`, and
  `crop_signature` tells readers exactly which detection source and crop
  configuration produced the ROI set.

So even in a future `geometry_only` mode, `crop_runs` still provides important
value beyond raw detections/bboxes. The main thing that becomes optional is the
persisted ROI pixel array (`roi_images`), not the crop-stage geometry/provenance
contract itself.

## Minimum Live-Crop Provenance Contract

Goal: every ROI-derived datum should be traceable through the lineage of how it
was produced, even when ROI pixels are not persisted.

### Canonical crop geometry

These fields define the actual ROI patch that downstream stages should use:

- `frame_indices`
- `frame_counts`
- `roi_coordinates_full`
- `roi_size`
- `bbox_norm_coords`
- `detection_indices`

Best-practice rule:

- live crop reconstruction should use stored crop geometry
  (`roi_coordinates_full` + `roi_size`) as the canonical patch definition
- it should not recompute the crop window from detections/bboxes if that can be
  avoided

Detections answer where the fish is. Crop geometry answers what exact ROI patch
was actually used downstream.

### Crop lineage / source provenance

These fields explain where the crop came from:

- `crop_storage_mode`
- `source_detect_run`
- `detection_source_path`
- `detection_source_type`
- `crop_signature`
- source frame reference:
  - preferred: internal frame source such as `raw_video/images_full`
  - fallback: `source_video_path`

### Render / reproduction provenance

These fields explain how to reproduce the same live-cropped ROI pixels:

- decode backend
- interpolation policy
- clipping/padding policy and pad value
- grayscale/color conversion policy
- output dtype / channel layout / precision

For new mixed-mode support, `frame_counts` and `detection_indices` should be
treated as part of the complete provenance contract, not as optional nice-to-
have metadata.

## Recommended Zarr Shape

Keep `crop_runs/<run>` as the single canonical crop-stage location.

### Always required

- `frame_indices`
- `frame_counts`
- `roi_coordinates_full`
- `bbox_norm_coords`
- `detection_indices`
- `roi_size` attr
- source/provenance attrs (`source_detect_run`, `detection_source_path`,
  `detection_source_type`, `crop_signature`)

### Optional in the future

- `roi_images`

### Suggested storage-mode attr

- `crop_storage_mode`: `materialized | geometry_only`

This is enough for readers to know whether to consume stored ROI pixels or
reconstruct them live.

## Mixed-Mode Recording Archives

`Palette` should eventually support repositories where some recordings have
materialized ROI pixels and others do not.

Recommended handling:

- Keep `crop_runs/<run>` as the canonical crop-stage record in both modes.
- Use `crop_storage_mode` to declare whether a run is `materialized` or
  `geometry_only`.
- Route all consumers through a shared ROI resolver:
  - read `roi_images` directly when present,
  - otherwise reconstruct ROI pixels live from crop geometry and frame/video
    sources.
- Avoid ambiguous "latest" semantics while modes are mixed.
  `crop_runs.latest` should remain backward-compatible and continue to resolve
  to a materialized-compatible run. Mixed-mode support should add explicit
  pointers instead of redefining `latest`.

This allows mixed repositories to exist without forcing an all-at-once
migration.

## Manual Bounding-Box Edit Policy

The user-facing workflow for refined/manual detections should allow idempotent
in-place edits to the active manual refined-detect state. If a user drags,
resizes, or otherwise corrects a bbox in Crimson, that should usually update
the existing manual refined-detect result rather than forcing a brand-new
user-visible run.

Recommended policy:

- Treat the latest manual refined-detect state as editable.
- Preserve stable detection identity when the change is a move/resize of an
  existing bbox.
- Treat downstream crop/keypoint/eye-mask data as patchable row-aligned outputs
  when detection identity is preserved.

### In-place patch class

These edits should be treated as in-place manual patches, not as a new
user-visible run:

- move an existing bbox
- resize an existing bbox
- small geometry corrections where detection identity remains the same
- edits where per-frame detection count and row identity do not change

For this patch class, downstream behavior should be:

- update the manual refined bbox row
- update the corresponding crop geometry row
- update `roi_images` for that row only if the crop run is materialized
- invalidate or refresh any temporary ROI cache derived from the affected crop
  revision
- recompute keypoints for the affected rows
- recompute eye masks for the affected rows, or mark curated eye-mask rows
  stale when an explicit human resolution step is required

This is the model already suggested by the targeted patch utilities in
`patch_crops_from_refined.py` and `patch_keypoints_from_crops.py`.

### Version-bump class

These edits should be treated as a new refined-detect revision, even if the UI
still presents them as the latest manual state:

- add a detection
- delete a detection
- split one detection into multiple detections
- merge multiple detections into one
- any edit that changes per-frame detection count, row ordering, or row
  identity

These cases are not safe to treat as a simple row patch because downstream row
alignment (`frame_indices`, `detection_indices`, `frame_counts`) can change.

### Internal lineage requirement

Allowing in-place manual edits in the UI does not mean lineage should be
mutable or ambiguous internally.

Best-practice requirement:

- keep one user-facing "latest manual" state,
- but version it internally with revision metadata and updated signatures

At minimum, any manual bbox edit that changes effective crop geometry should
cause:

- the relevant refined-detect revision metadata to advance,
- the affected crop run signature/revision to advance,
- downstream caches and derived runs to record the source revision/signature
  they were built from

This prevents a stale temporary ROI cache or stale downstream run from being
mistaken for output generated from the current manual geometry.

### Geometry-only implication

This policy applies equally to materialized and `geometry_only` crop runs.

For `geometry_only` runs:

- the crop patch updates stored crop geometry/provenance rows,
- temporary ROI caches must be invalidated by crop signature/revision changes,
- downstream keypoint/eye-mask patching still follows the same row-identity
  rules

So the correct long-term model is not "manual edits require a totally new
user-visible run." It is:

- mutable latest manual review state for operators,
- explicit internal revision/signature tracking for lineage,
- targeted downstream patch/update behavior when row identity is preserved.

## Runtime ROI Cache Policy

Geometry-only archive support does not imply that every downstream workload
should read ROIs by repeatedly decoding the full source video.

Recommended runtime model:

- Keep canonical analysis archives lineage-first and allow `crop_runs/<run>` to
  be `geometry_only`.
- For bulk ROI-model inference on large or decode-limited frame sources, allow
  an explicit temporary ROI cache.
- Keep that cache outside the canonical archive by default, so it behaves as a
  scratch/runtime accelerator rather than part of the archival contract.

Suggested cache policy surface:

- `roi_cache_policy = never | auto | always`

Recommended semantics:

- `never`: always read materialized `roi_images` when present, otherwise
  reconstruct live from crop geometry and full-frame sources.
- `auto`: if a crop run is `geometry_only` and the source frames are expected to
  bottleneck ROI inference, create or reuse a temporary cache of ROI tensors.
- `always`: always materialize or reuse a temporary ROI cache before ROI-model
  inference.

Recommended cache identity:

- analysis archive identity/path
- crop run name
- `crop_signature`
- source frame identity/fingerprint
- ROI size and frame source choice

Recommended invariants:

- temporary caches must not change `crop_runs.latest`,
  `crop_runs.latest_materialized`, or crop review status
- temporary caches must not be treated as curated training artifacts
- temporary caches may be deleted and regenerated without changing canonical
  archive lineage

This separates "lean canonical archive" from "fast runtime execution" without
forcing every analysis archive to permanently carry duplicate ROI tensors.

## Current Benchmark Signal

Initial smoke-archive benchmarking on `2026-03-07` gives a useful first signal
for large-video ROI workloads:

- archive: `2026-01-28T19-22-28Z_arena_1_DefaultScreen_geometry_smoke_analysis`
- ROI count: `23,287`
- ROI size: `512x512`
- source video: `4512x4512`

After aligning the temporary ROI cache layout with the crop run's own
`roi_chunk_len` and storage settings, warm-cache geometry-only inference became
very close to the materialized baseline:

- keypoints:
  - materialized: `72.0s`
  - geometry-only + cache reuse: `73.5s`
- U-Net eye masks:
  - materialized: `358.9s`
  - geometry-only + cache reuse: `355.5s`

What remains expensive is first-time cache population:

- keypoints on the first geometry-only run with `roi_cache_policy=always`:
  `347.1s`

Interpretation:

- geometry-only plus a shared temporary ROI cache is now a viable runtime model
  for sequential ROI stages on this smoke archive,
- warm-cache performance is no longer the blocker,
- first-use cache materialization cost is still real and should be treated as a
  deliberate throughput tradeoff,
- `roi_cache_policy=auto` and a pure `geometry_live` benchmark still need to be
  validated before changing the default archival policy.

## Latest Pointer Policy

Recommended pointer semantics:

- `crop_runs.attrs["latest"]`: latest materialized-compatible run
  (backward-compatible behavior)
- `crop_runs.attrs["latest_materialized"]`: explicit latest materialized run
- `crop_runs.attrs["latest_any"]`: latest run regardless of storage mode

Optional later addition:

- `crop_runs.attrs["latest_geometry_only"]` only if a real consumer needs that
  lookup directly

This avoids one pointer name meaning different things to different consumers.
It also allows capability-based reader selection:

- materialized-only readers:
  explicit run -> `latest_materialized` -> legacy `latest`
- mixed-mode readers:
  explicit run -> `latest_any`

Geometry-only runs should not become the effective meaning of `latest` until
legacy/materialized-only readers are fully migrated.

## Training Artifact Policy

Keypoint and eye-mask training artifacts should continue to persist ROI images
by default, even if source analysis archives eventually support geometry-only
crop runs.

Reasoning:

- training performs repeated random ROI access across many epochs,
- training artifacts should be portable and self-contained,
- deterministic ROI tensors matter for debugging, benchmarking, and export,
- TensorRT/export/calibration workflows fit better with stable materialized
  inputs.

Recommended policy:

- Analysis / production archives may eventually support mixed crop storage
  modes.
- Merged keypoint and eye-mask training zarrs should remain materialized
  artifacts by default.
- If a source recording is geometry-only, the training export step should
  materialize ROI pixels into the training artifact rather than pushing live
  cropping into the training loop.
- Temporary ROI caches used for fast inference should remain distinct from
  durable training artifacts.

## Archive Role Policy

`Palette` should treat analysis/production archives and training artifacts as
different products with different guarantees.

### Analysis / production archives

- should remain lineage-first and as lean as practical,
- should not require imported/downsampled image payloads to exist as part of
  the contract,
- may carry cached or materialized image data when useful, but that should not
  be the required baseline contract.

### Training artifacts

- should be intentionally duplicated, self-contained dataset artifacts,
- should persist the actual ROI/image tensors used for training,
- should be stable, portable, versioned, and self-documenting,
- should not depend on live video decode during the training loop.

This makes the duplication boundary explicit: duplication is desirable when it
creates an immutable, documented training artifact, and undesirable when it is
an accidental or mandatory burden on all analysis archives.

## Traditional Pipeline Policy

Traditional image-processing pipelines are not the main target for mixed-mode
crop reading.

Current policy:

- YOLO keypoint / eye-mask inference and retry paths are the primary
  mixed-mode/live-crop-compatible readers.
- Traditional detection and traditional keypoint inference remain
  imported/materialized-only consumers.
- Traditional readers should prefer `latest_materialized` and fail clearly when
  only geometry-only crop runs exist or required imported frame arrays are
  missing.

This keeps the migration focused on the main ROI-model paths without weakening
the contract of older materialized-image workflows.

## Current Implementation Status

Implemented as of 2026-03-06:

- legacy analysis and training archives have a metadata backfill path for
  `crop_storage_mode`, `latest_materialized`, and `latest_any`
- a shared ROI resolver exists for mixed-mode crop reading
- YOLO keypoint inference, YOLO eye-mask inference, and keypoint retry use the
  shared mixed-mode resolver
- U-Net eye-mask inference also uses the shared mixed-mode resolver
- geometry-only crop writing exists as an explicit writer mode
- a shared temporary ROI cache exists for geometry-only crop runs, including an
  isolated GPU/kvikIO external-video cache builder
- detect/keypoint/eye-mask batch runners now pre-resolve registry models before
  execution
- traditional detection and traditional keypoint inference now fail early when
  materialized/imported image requirements are not met

## What To Do Next

Execution checklist:
- `docs/crop_storage_mode_migration_todo.md`

Related policy:
- `docs/legacy_archive_migration_policy.md`

### Phase 1: Metadata and resolver groundwork

Mostly complete. Existing archive backfill and the shared ROI resolver now
exist. Crop writers still remain materialized by default.

1. Add `crop_storage_mode` attr support to crop writers and readers, but keep
   all writers materialized by default.
2. Add one shared ROI resolver abstraction that can:
   - return `roi_images` when present,
   - otherwise reconstruct ROI pixels from full-frame sources using
     canonical crop geometry (`frame_indices`, `roi_coordinates_full`,
     `roi_size`) and existing source/video provenance.
3. Keep `crop_runs.latest` restricted to materialized-compatible runs during
   this phase.

### Phase 2: Migrate the most important readers

Mostly complete for the main ROI-model paths.

1. Migrate sequential ROI inference first:
   - YOLO keypoints
   - YOLO eye masks
   - keypoint retry
2. Keep traditional/imported-image pipelines explicitly materialized-only.
3. Validate parity, throughput, and provenance behavior.

### Phase 3: Runtime ROI cache support

Largely complete for the main ROI-model paths.

1. A shared temporary ROI cache now exists for decode-limited ROI inference.
2. Cache policy is exposed as `never | auto | always`.
3. Caches stay outside canonical archives by default.
4. Cache identity depends on archive/run/signature lineage and does not affect
   `latest` pointers, review status, or training artifact semantics.
5. Remaining work is to tune `roi_cache_policy=auto`, benchmark
   `geometry_live`, and validate on more than one archive.

### Phase 4: Migrate training/export, viewers, and validators

1. Update pose/eye-mask training prep and merged export paths to accept either
   materialized crops or live-crop reconstruction.
2. Update validators/diagnostics so `roi_images` is no longer treated as
   universally required when `crop_storage_mode=geometry_only`.
3. Migrate Palette/Crimson read-only viewers, tuners, and review tools to the
   shared resolver or explicit cache path.
4. Revisit schema docs and stage contracts once both modes are genuinely
   supported.

### Phase 5: Introduce geometry-only writing as opt-in

1. Add an explicit writer mode for geometry-only crop runs.
2. Do not let geometry-only runs become `crop_runs.latest` until the main
   consumer set is migrated.
3. Add an explicit materialization command that can generate `roi_images` later
   from a geometry-only run.

### Phase 6: Re-evaluate the default

Only after the above phases should `Palette` decide whether geometry-only
should become the default for some or all workflows.

## Policy Until Migration Is Complete

- Materialized crops remain the default.
- Training/export should continue to assume materialized crops unless explicitly
  updated.
- New crop consumers should use the shared ROI resolver instead of directly
  dereferencing `crop_group["roi_images"]`.
- Bulk ROI inference on large-frame geometry-only archives should prefer an
  explicit temporary ROI cache over repeated full-frame decode.
- Traditional/imported-image pipelines should continue to require materialized
  image inputs.
- New docs should avoid implying that geometry-only is already safe as the
  default mode.

## Open Questions To Revisit Later

- What should drive `roi_cache_policy=auto`:
  source resolution, measured decode throughput, repeated downstream reuse, or
  an explicit workflow flag?
- Which review/tuning workflows are latency-sensitive enough to require cache
  creation rather than direct live decode?
- Should runtime caches live under per-recording scratch space, a shared global
  cache root, or a user-configured cache path?

## Bottom Line

Live cropping is a valid target architecture, but for large-frame ROI inference
it should usually be paired with a temporary ROI cache rather than treated as a
pure live-decode hot path. The first smoke benchmark now shows that
geometry-only plus warm-cache reuse can match the materialized baseline closely.
The next step is still not an immediate default switch; it is
cache-aware mixed-mode support, `auto`-policy tuning, broader benchmarking, and
continued use of materialized crops where they are the right operational
contract.
