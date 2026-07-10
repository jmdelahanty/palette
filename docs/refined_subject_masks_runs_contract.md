# Refined Subject Masks Runs Contract (Draft v1)
<!-- contract-meta
version: 1
status: draft
last_verified: 2026-06-25
-->

Purpose: define the runtime/storage contract for editable, refined
subject-mask artifacts that hold canonical component masks for body, swim
bladder, and modern left/right eye refinement under the same component model.

For the contour/cache ownership and row-local edit propagation policy, see
[refined_subject_mask_geometry_cache_and_propagation_design.md](refined_subject_mask_geometry_cache_and_propagation_design.md).
For stable row identity, frame lookup indexes, and optional track identity, see
[realtime_sparse_row_index_contract.md](realtime_sparse_row_index_contract.md).

## Scope

- Define `refined_subject_masks_runs/<run>` as the canonical refined/editable
  subject-mask stage.
- Support refined body masks and refined swim-bladder masks.
- Support canonical refined `eye_left` and `eye_right` masks when raw/model
  sources provide assignable eye evidence.
- Support component-scoped review and reasons.
- Reserve space for component-specific derived geometry such as contours,
  centroids, and centerline/spline-related outputs.
- Define the implemented eye-capable refined layout for:
  - `eye_left`
  - `eye_right`
  - per-eye ellipses
  - per-eye contours
  - cross-eye relation metrics such as `eye_separation`

## Non-goals

- Removing read support for historical `refined_eye_masks_runs`.
- Defining the final exact geometry array schema for body contours or splines.
- Defining `analysis/subject_shape_runs`.
- Defining merged training artifact layout.

## Relationship To Existing Stages

Near-term canonical relationship:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_subject_masks_runs/<run>
  -> analysis/subject_shape_runs/<run> # derived analysis geometry
```

Legacy eye-specialized compatibility path during transition:

```text
crop_runs/<run>
  -> subject_mask_runs/<run>
  -> refined_eye_masks_runs/<run>
```

Policy:

- `refined_subject_masks_runs` is the refined stage for generic subject-mask
  components.
- `refined_eye_masks_runs` remains supported during the transition as the
  eye-specific refined compatibility and historical stage.
- registry/query/operator surfaces should prefer unified subject-mask component
  rows for eye availability, with legacy eye stages projected in only as
  compatibility inputs when native eye-capable subject-mask rows are absent.
- the target steady-state for new eye-capable refined authoring is still
  `refined_subject_masks_runs`
- future unification should align eye refinement under the subject-mask
  component model without forcing a destructive migration now
- sparse multi-source workflows should not require an assembled raw
  `subject_mask_runs/<run>` intermediate before refinement

## Canonical Label Scope

Recommended minimum currently implemented component scope:

- `subject_body`
- `swim_bladder`

Optional/compatibility seed labels:

- `eyes_union`
- `eye_left`
- `eye_right`

Canonical target for eye-capable refined authoring:

- `label_schema_id = "subject_v1_lr"`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

Compatibility/raw model-output schema:

- `subject_v1_union` remains valid for raw compatibility and export use
- `subject_v1_union` is not the preferred long-term refined eye authoring
  schema because it loses anatomical eye identity

Refined eye-authoring policy:

- `eyes_union` is allowed as raw/model output, import, provenance, or
  transitional seed input, but it is not the canonical refined eye authority.
- when eye content is promoted into `refined_subject_masks_runs`, subject-mask
  refinement/finalization should materialize `eye_left` and `eye_right`
  components when anatomical side can be assigned
- if a union or unordered eye source cannot be assigned safely, the refined run
  should record ambiguity/review state instead of claiming complete refined
  left/right eye availability
- operator, geometry, and training consumers that require refined eye identity
  should consume `eye_left` / `eye_right`, not `eyes_union`

Writers must always persist:

- `label_schema_id`
- `mask_labels`
- `available_channels`

Readers must never infer component meaning from channel index alone.

## Row Identity And Frame Lookup

`refined_subject_masks_runs/<run>` is sparse and row-aligned:

```text
one row = one refined subject instance
channels = semantic mask components for that instance
```

Do not use physical row position as durable identity. Physical row order is an
array layout detail and may change when rows are sorted, compacted, migrated,
or late-appended.

Modern refined subject-mask runs must carry explicit crop-row lineage:

- `source_crop_run` attr is required.
- `source_crop_row_ids` array is required with shape `(N,)`.
- `frame_indices` array is required with shape `(N,)` and stores
  archive-global frame indices.
- For every mask row `i`,
  `crop_runs/<source_crop_run>/frame_indices[source_crop_row_ids[i]]`
  must equal `refined_subject_masks_runs/<run>/frame_indices[i]`.
- ROI-local masks and contours must be placed using
  `crop_runs/<source_crop_run>/roi_coordinates_full[source_crop_row_ids[i]]`.

Consumers must not assume refined-mask row `i` equals crop row `i` except as a
warned legacy/off-contract fallback.

Legacy refined subject-mask runs that predate `source_crop_row_ids` may be
upgraded with:

```bash
scripts/py -m fisheye.utils.validate_refined_subject_mask_contract /path/to/analysis.zarr \
  --run latest \
  --backfill
```

The validator only writes `source_crop_row_ids = arange(N)` when the refined run
and `crop_runs/<source_crop_run>` have the same row count and matching
`frame_indices` plus any shared row-identity arrays such as `detection_indices`,
`source_refined_row_ids`, and `source_detect_row_index`. If those checks fail,
regenerate from a modern `subject_mask_runs` source instead of guessing row
lineage.

Writers should preserve stable logical row identity when available:

- copy `source_refined_row_ids` from upstream refined-detect/crop lineage when
  a refined subject-mask row corresponds to an existing refined detection row
- preserve row IDs across row-local mask edits
- allocate new non-reused row IDs when a truly new subject instance row is
  added in a future manual-add or multi-subject workflow

The current near-term single-fish runs may rely on copied
`source_refined_row_ids`, but future subject-mask authoring needs an explicit
subject-row ID field for rows that are not backed by a refined-detect row.

For interactive viewing, writers should add a `frame_index/` lookup cache when
the run is large:

```text
frame_index/frame_numbers
frame_index/row_start
frame_index/row_count
frame_index/row_indices
```

This lets consumers find all rows for a frame without assuming rows are stored
in frame order. A late-appended row for frame `101` is safe when this index is
rebuilt or updated, because frame `101` resolves through the index rather than
through physical array locality.

`track_id` is a separate optional biological/temporal identity. It should only
be written when a run has joined against an exact tracking source, and it must
not replace stable row identity for edits or stale downstream recomputation.

## Evolution Policy

This contract is intended to support:

1. refined body/swim-bladder masks while eyes remain specialized elsewhere,
2. fuller multi-component refined subject masks, and
3. eye alignment under the subject-mask component model.

V1 directly supports case 1 and now supports case 2/3 for the implemented
subject-mask U-Net and smart-finalizer path:

- raw `subject_v1_union` outputs are finalized into `subject_v1_lr`
- `eyes_union` is assigned into `eye_left` and `eye_right` using declared
  assignment keypoint lineage
- refined-subject eye geometry and eye-pair relation metrics are written from
  the generated LR masks

Manual review/editor workflows and compatibility materialization are still
transition-state surfaces, but the storage contract no longer treats LR eye
components as future-only.

## Interactive Review Save And Apply Semantics

`refined_subject_masks_runs/<run>` is the canonical editable Zarr surface for
subject-mask review, but browser paint/lasso interactions should not require a
canonical Zarr rewrite for every small UI action.

## Mask Storage Authority And Chunking

Dense `masks_roi` is the authoritative pixel surface for modern editable
refined subject masks. Compact stores are derived caches:

- `masks_roi` is the only live review/edit/writeback target.
- `mask_bitpacked` is an optional fixed-size compact display/publication cache.
- `mask_rle` is an optional compact archive/fallback display cache.
- component contours, component metrics, relations, bitpacked masks, and RLE
  are derived from dense masks and can be regenerated.

Training artifacts must also use dense `masks_roi` as the label source of
truth. Compact mask stores may be read from analysis sources through
`MaskStore` during export, but the training zarr output must materialize dense
`subject_mask_runs/<run>/masks_roi` and must not store compact label surfaces
as training targets.

Use an explicit dense encoding name for new provenance:

- canonical encoding name: `dense_uint8_v1`
- storage surface: `masks_roi`
- logical shape: `(N, C, H, W)`
- dtype: `uint8`
- value semantics: binary `0/1`
- row axis: refined subject-mask row, not frame and not crop row
- channel axis: semantic component labels declared by `mask_labels`
- spatial axes: ROI-local mask pixels with the same `H,W` as the source crop
  image surface

Existing `dense_uint8` attrs and CLI values are the compatibility spelling for
this v1 dense encoding. New writers should prefer recording both the stable
contract fields (`mask_storage_authority = "masks_roi"`,
`editable_mask_surface = "masks_roi"`, `training_mask_surface = "masks_roi"`)
and the explicit encoding (`dense_uint8_v1`) while continuing to accept
historical `dense_uint8` on read.

Do not hard-code `512x512` as a semantic invariant. Refined mask size follows
the source crop geometry:

```text
masks_roi.shape == (N, C, roi_height, roi_width)
```

The default modern dense chunk policy is component-separated and full-spatial:

```text
masks_roi.chunks == (min(128, N), 1, roi_height, roi_width)
```

For the common modern `512x512` four-component refined run, this gives:

```text
masks_roi chunks = [128, 1, 512, 512]
```

For smaller or larger crop-video surfaces, only the spatial chunk dimensions
change. For example, a `348x348` crop-video-backed run should use
`[128, 1, 348, 348]` rather than padding the mask store to `512x512` unless the
source crop image itself was explicitly padded to that size.

The default modern bitpacked cache policy is playback/publication-oriented, not
the edit authority:

```text
mask_bitpacked/masks_packed.shape == (N, C, roi_height, ceil(roi_width / 8))
mask_bitpacked/masks_packed.chunks == (
  min(512, N),
  min(4, C),
  roi_height,
  ceil(roi_width / 8),
)
```

For a common `512x512`, four-component refined run, this gives:

```text
mask_bitpacked/masks_packed chunks = [512, 4, 512, 64]
```

Freshly generated runs should make cache state explicit:

- `masks_roi_materialized = true`
- `mask_storage_authority = "masks_roi"`
- `editable_mask_surface = "masks_roi"`
- `training_mask_surface = "masks_roi"` when the zarr is a training artifact
- `mask_store_encodings` includes `dense_uint8_v1` and any derived caches
- `mask_bitpacked_materialized = true/false`
- `mask_rle_materialized = true/false`
- `derived_mask_caches_stale = false`
- `mask_bitpacked_stale = false` when bitpacked exists
- `mask_rle_stale = false` when RLE exists
- `metrics_stale = false`
- `contours_stale = false`

After a dense mask writeback, writers must update the touched dense
row/component and mark derived products stale instead of treating derived arrays
as authoritative:

- `derived_mask_caches_stale = true`
- `contours_stale = true` when contours exist
- `metrics_stale = true` when derived metrics exist
- `mask_bitpacked_stale = true` when bitpacked exists
- `mask_rle_stale = true` when RLE exists

Regeneration/validation/promotion jobs may clear these stale flags only after
successfully refreshing the affected derived arrays from current dense
`masks_roi`.

Recommended browser model:

- checkpoint frequent UI edits into the labeling/session store so the browser
  can recover an overlay after refresh, disconnect, or crash
- for the web-labeling v1 implementation, use the labeling SQLite sidecar as
  the checkpoint/session metadata store
- render canonical `masks_roi` plus unapplied session edits in the UI
- apply edits to the canonical Zarr run only on an explicit save/apply action,
  submit, or finalize
- do not treat browser close as an apply action
- do not require assignment completion before applying mask edits to Zarr
- keep approval/review-state changes explicit and separate from save/apply
- represent v1 mask checkpoint/apply payloads as full replacement dense ROI
  masks per `(row, component)`, not as stroke-delta replay

This gives reviewers the desired "continue where I left off" behavior in two
ways:

- applied edits are durable in `refined_subject_masks_runs/<run>/masks_roi` and
  visible when the same assignment is reopened later
- unapplied checkpoints can still be recovered as a session overlay when the UI
  is reopened before the next canonical apply

Canonical apply must validate:

- active assignment/session ownership
- target run path
- target `edit_revision`, treating a missing revision on older runs as `0`
  before the first successful apply
- row identity, using `source_crop_row_ids`, `frame_indices`, and any available
  stable source/refined row IDs rather than trusting physical row position alone
- component names and channel identity from `mask_labels`

Canonical apply should coalesce edits by component, target array, and physical
Zarr chunk before writing. After a successful apply:

- increment `edit_revision`
- append a durable edit event with a retry-stable `apply_id` or equivalent
  idempotency key
- persist only the authoritative dense row/component and minimal revision,
  review, editor, reason, and timestamp state synchronously
- mark metrics, reasons, contours, geometry, `mask_bitpacked`, and `mask_rle`
  stale for the touched row/component when those derived surfaces exist
- refresh derived products only during an explicit validation, promotion, or
  maintenance operation; a UI may compute ephemeral feedback without making
  it part of the durable save transaction
- refresh registry/QC summaries after apply, not after every session checkpoint

The Palette-owned write boundary must hold a lock that covers every physical
chunk and shared metadata record it mutates. A refined-run-wide lock is an
acceptable conservative first implementation. Finer per-physical-chunk locks
are safe only when concurrent stale-scope and revision updates cannot overwrite
one another. The writer must re-read and compare the target row revision after
acquiring the lock and before changing `masks_roi`.

When `mask_rle` is marked stale by an apply, writers should set at least:

- `mask_rle_stale = true`
- `mask_rle_stale_reason`
- `mask_rle_stale_at_utc`
- `mask_rle_stale_component_names`
- `mask_rle_stale_row_count`
- `mask_rle_stale_row_min` and `mask_rle_stale_row_max` when rows are present
- `mask_rle_stale_since_edit_revision`

Approval guard:

- saving or applying pixels must not imply component approval
- if unapplied session edits exist, component approval should either be blocked
  with a clear message or require a successful apply first
- subject-mask approval remains component-level, with run-level review state
  derived from `component_review_statuses`

Minimum browser-specific tests:

- checkpointing a replacement mask does not mutate `masks_roi`
- reopening the session restores the replacement-mask overlay
- applying writes the replacement mask to `masks_roi`
- applying increments `edit_revision` after success
- retrying the same `apply_id` does not double-apply or double-increment
  `edit_revision`
- applying with a stale `target_edit_revision` fails cleanly
- component approval is blocked or forced through apply when unapplied edits
  exist
- applied edits are allowed while the labeling assignment remains open

## Assembly And Finalization Semantics

`refined_subject_masks_runs/<run>` is not merely a bag of assembled component
masks. It is the canonical refined/editable working artifact, and it should be
treated as valid only after subject-mask refinement/finalization has
materialized the canonical QA surface.

Required behavior:

- the preferred future seed path is a single raw
  `subject_mask_runs/<run>` containing all model-predicted subject-mask
  probability components plus model/config/provenance
- sparse multi-source assembly may seed a new
  `refined_subject_masks_runs/<run>` directly, but this is a compatibility and
  repair path rather than the steady-state model-output path
- thresholding raw probabilities into binary masks is part of
  refined-subject finalization, not a requirement of native raw model output
- the seed/assembly step must be followed by subject-mask finalization before
  the run is treated as a valid refined artifact
- finalization is responsible for canonical run/component metrics, reasons,
  review scaffolding, provenance updates, and any refinement-time geometry
  derived by this stage family
- refined candidates store the post-refinement binary mask and QC surface; they
  do not need to duplicate the pre-refinement thresholded mask because the
  source raw probability run and threshold/refinement policy are the recoverable
  "before" state
- for eye-capable runs, finalization is also responsible for promoting
  `eyes_union` or unordered eye seeds into canonical `eye_left` / `eye_right`
  components when the assignment is safe, or marking the affected rows/components
  ambiguous for review when it is not
- finalization is also responsible for component-specific topology cleanup:
  body masks may close small gaps, fill holes, remove detached islands, and keep
  one best body component; swim-bladder masks may fill small holes and choose
  one compact internal component; eye-union masks may preserve two valid eye
  components instead of keeping only the largest
- topology cleanup must write metrics/reasons that expose removed area and
  probability mass, and rows with large or ambiguous cleanup deltas must be
  marked for review instead of silently approved
- subject-body mask-level QC is owned by this refined-mask stage, not by
  downstream subject-shape extraction. See
  [subject_body_mask_qc_design.md](subject_body_mask_qc_design.md) for the
  additive QC group and review-gating policy for connected but implausible body
  masks such as attached dish scratches.

Initial allowed seed sources for unified assembly:

- raw `subject_mask_runs`
- transitional `refined_eye_masks_runs` for eye components
- canonical `refined_subject_masks_runs` component sources when assembling a
  new coherent refined run from previously split refined component runs

Current implementation note:

- the shipped assembler/finalizer accepts a single raw
  `subject_mask_runs/<run>` via `--subject-run`; all available canonical
  components in that source are copied as refined seeds and finalized into one
  coherent `refined_subject_masks_runs/<run>`
- it also accepts raw `subject_mask_runs` component sources for
  body/eyes/swim bladder when repairing or combining split sources
- it also accepts direct `refined_eye_masks_runs` sources for canonical
  `eye_left` / `eye_right` component seeding
- raw `eyes_union` is treated as refinement input/provenance, not as a
  canonical refined component; a `--subject-run` exposing available
  `eyes_union` can be assigned into `eye_left` / `eye_right` when explicit
  assignment keypoint attrs or source keypoint lineage resolve to usable
  anatomical eye keypoints
- `assignment_keypoints_run` / `assignment_keypoint_group` are preferred over
  `source_keypoints_run` / `source_keypoint_group` for `eyes_union` assignment,
  because raw subject-mask segmentation may be crop-only while the LR split is
  a later deterministic refinement step
- generated LR eye components record `eyes_union` as the source channel plus
  assignment method/keypoint provenance; the refined-subject finalizer then
  writes the standard eye geometry/QC surface from the generated LR masks
- current production finalization treats assignment-time eye ellipse fitting as
  the authoritative refined-subject eye-geometry measurement. When assignment
  geometry is complete, the finalizer must write
  `components/eye_left|eye_right/geometry/*` and
  `relations/eye_pair/metrics/*` from the assignment payload with
  `eye_geometry_postcompute_backend = "assignment_reuse"` and
  `eye_geometry_source_measurement =
  "eyes_union_assignment_measure_mask"`. A second eye-ellipse pass over saved
  masks is a repair/backfill fallback, not the normal production path.
- if keypoint lineage is missing or the assignment produces no usable LR rows,
  assembly fails closed instead of creating a misleading refined eye surface
- it now accepts `refined_subject_masks_runs/<run>` as an explicit component
  source for split-run consolidation; the new component provenance points to
  the immediate refined source and carries the upstream component provenance
  under `upstream_component_provenance`
- refined component sources are approved-only by default: assembly from an
  existing `refined_subject_masks_runs/<run>` requires the requested component
  to have `component_review_statuses[component].state == "approved"`, with
  `--allow-unapproved-components` reserved for draft/QA assembly
- source review state is recorded as component provenance, but target component
  approval is not inherited by default; pass `--promote-source-review` only when
  the operator explicitly wants approved source review payloads copied onto the
  assembled/finalized target run
- `fisheye.refinement.finalize_subject_masks` is the smart finalizer for raw
  probability-first `subject_mask_runs`; it writes deterministic row chunks,
  cleanup metrics, source-seed masks, component provenance, reason tags,
  review-triage counts, process-shard execution metadata, and optional eye
  geometry
- the supported production parallel backend is `process_shards`: each worker
  opens the archive once, owns a contiguous whole-physical-chunk-aligned row
  shard, and writes only that shard; `serial_driver` remains a deterministic
  correctness/debug fallback

Source ROI pixel/decode provenance preservation:

- finalization from a raw `subject_mask_runs/<run>` must preserve the source
  crop snapshot attrs, including `source_crop_storage_mode`,
  `source_crop_signature`, `source_crop_revision`,
  `source_roi_image_representation`, `source_roi_pixel_contract_name`, and
  `source_roi_pixel_contract`
- finalization must also preserve ROI read/cache attrs when the raw subject-mask
  run exposed them: `source_roi_read_mode`, `roi_cache_policy`,
  `source_roi_cache_used`, `source_roi_cache_backend`,
  `source_roi_cache_key`, `source_roi_cache_path`,
  `source_roi_cache_canonical_path`,
  `source_roi_cache_expected_archive_path`,
  `source_roi_live_acceleration_requested`,
  `source_roi_live_acceleration_effective`,
  `source_roi_live_acceleration_fallback_reason`, and
  `source_roi_live_gpu_chunk_frames`
- `source_roi_cache_path` is the effective runtime path from the raw inference
  job and may be node-local scratch; `source_roi_cache_canonical_path` is the
  durable cache identity when a staged cache was used
- for legacy raw eye-stage data, the compatibility bridge remains:
  `refined_eye_masks_runs` or `eye_masks_runs`
  -> projected/backfilled `subject_mask_runs/<run>`
  -> assembled/finalized `refined_subject_masks_runs/<run>`

Safety rule:

- the assembler must reject split refined component sources unless crop
  lineage, row lineage, row count, detection source, and ROI shape match
- a historical refined source-view crop signature mismatch is allowed only
  when the mismatch is limited to
  `source_crop_signature.detection_source_path` and
  `source_crop_signature.detection_source_type`, and the sources otherwise
  share crop identity, row lineage, row count, detection source, and ROI shape
- production assembly from split refined component sources must also reject
  pending, missing, or non-approved component review states; unapproved sources
  are only allowed with an explicit draft/QA override

## Retired Additive Unified Eye/Swim Migration Procedure

This procedure was a transitional bridge for historical
`refined_eye_masks_runs` data. It is retired after the 2026-07-01 eye-mask
severance census found zero active recordings requiring conversion. New
RedScare-style composition should source eye and swim components from
`refined_subject_masks_runs` / subject-mask component channels directly, not
from the deleted `--refined-eye-run` bridge.

Historical batch result from the 2026-04-25 recording migration:

- 52 recording training zarrs scanned
- 51 approved-compatible unified eye/swim runs written
- 50 runs became fully approved after explicit legacy refined-eye review
  promotion
- 1 run remained pending because the legacy refined-eye source review was
  pending

Those outputs remain valid historical refined-subject runs. The command surface
that created them is no longer the current workflow.

## Output Layout

```text
refined_subject_masks_runs/
  attrs:
    latest                                  "<run_id>"
    latest_complete                         "<run_id>"
  <run_id>/
    attrs:
      palette_run_completion_contract       "palette.zarr_run_completion.v1"
      palette_run_completion_status         "complete"
      palette_run_name                      "<run_id>"
    frame_indices                           (N,) int32
    frame_counts                            (F,) int32           # recommended
    detection_indices                       (N,) int32           # recommended
    source_crop_row_ids                     (N,) int64
    detection_source                        (N,) int8
    masks_roi                               (N, C, H, W) uint8 required dense authority/edit surface for modern runs
    mask_bitpacked/                         # optional derived compact display/publication cache
      attrs:
        schema_id                           "palette_mask_bitpacked_binary_v1"
        mask_encoding                       "bitpacked_binary_v1"
        layout                              "packed_width_array"
        logical_shape                       [N, C, H, W]
        encoded_shape                       [N, C, H, ceil(W / 8)]
      masks_packed                          (N, C, H, ceil(W / 8)) uint8
    mask_rle/                               # optional derived archive/fallback cache
      attrs:
        schema_id                           "palette_mask_rle_binary_v1"
        mask_encoding                       "coco_rle_fortran_v1"
        mask_value_semantics                "binary_0_1"
        layout                              "component_groups"
        encoded_shape_hw                    [H, W]
        component_names                     [<component_name>, ...]
      components/
        <component_index>_<component_name>/
          counts                            (total_counts,) uint32
          indptr                            (N + 1,) int64
          present                           (N,) bool
          area_px                           (N,) int32
          bbox_xyxy                         (N, 4) int32
    available_channels                      (C,) bool
    edit_applied                            (N, C) bool
    metrics/
      mask_present                          (N, C) bool
      area_px                               (N, C) float32
      centroid_xy                           (N, C, 2) float32   # recommended
      centroid_valid                        (N, C) bool         # recommended
      bbox_xyxy                             (N, C, 4) float32   # recommended
      bbox_valid                            (N, C) bool         # recommended
    components/
      <component_name>/
        provenance/                        # attrs-only subgroup for component lineage/update provenance
        reason_bytes                        (N, width) uint8     # recommended
        reason                              (N,) string          # optional mirror
        mask_present                        (N,) bool            # recommended
        area_px                             (N,) float32         # recommended
        geometry_valid                      (N,) bool            # optional
        edit_applied                        (N,) bool            # recommended
        metrics/                            # optional component-local QC summary arrays
          ...
        geometry/                           # optional extension point
          ...
        contours/                           # optional component-local contour storage
          ...
    relations/                              # optional cross-component derived values
      <relation_name>/
        metrics/
          ...
```

## `refined_subject_masks_runs/<latest>`

Required arrays:

- `detection_source`
  - shape: `(N,)`
  - expected to align with the source crop run
- physical mask store
  - modern editable runs must include dense `masks_roi`
  - optional compact `mask_bitpacked` and `mask_rle` stores are derived caches
  - consumers should use `fisheye.shared.mask_store.open_mask_store(...)`
    when they need to tolerate historical compact-only archives
- `masks_roi`
  - shape: `(N, C, H, W)`
  - dense refined binary masks
  - default compatibility surface for historical readers
  - live review/edit authority surface for modern editable runs
  - required for new editable analysis outputs and training artifacts
  - can be materialized/refreshed from compact `mask_bitpacked` or `mask_rle` with
    `scripts/py -m fisheye.utils.materialize_refined_subject_mask_store --apply`
  - should not be removed from editable analysis or training outputs
- `mask_bitpacked`
  - compact fixed-size exact binary masks
  - schema: `palette_mask_bitpacked_binary_v1`
  - encoding: `bitpacked_binary_v1`
  - layout: `packed_width_array`
  - array: `mask_bitpacked/masks_packed`
    - shape: `(N, C, H, ceil(W / 8))`
    - dtype: `uint8`
    - bit order: `little`
  - required attrs include `logical_shape`, `encoded_shape`,
    `component_names`, `packed_axis="width"`, `packed_bitorder="little"`,
    and `packed_width_bytes`
  - written as an additive mirror by
    `finalize_subject_masks --mask-storage dense_and_bitpacked`
  - not a live edit/writeback authority for modern runs
  - preferred compact display/publication cache because it is fixed-size and
    row/channel addressable
  - if dense `masks_roi` is materialized and edited while `mask_bitpacked`
    exists, Palette-owned refined-subject edit paths must mark bitpacked stale
    until a validation, promotion, or maintenance job refreshes it from dense
  - explicit refresh is available with
    `scripts/py -m fisheye.utils.materialize_refined_subject_mask_store --refresh-bitpacked --components eye_left --rows 42 --apply`
    for repair workflows or non-interactive dense edits
  - compact-only historical archives should be materialized to dense before any
    review/edit workflow
- `mask_rle`
  - compact component-separated exact binary masks
  - written as an additive mirror by
    `finalize_subject_masks --mask-storage dense_and_rle`
  - consumers that need dense masks should use `fisheye.shared.mask_store.open_mask_store(...)`
    rather than assuming a single physical encoding for historical archives
  - not yet the direct edit/writeback surface; if dense `masks_roi` was
    materialized and edited, compact `mask_rle` should be considered stale until
    regenerated with
    `scripts/py -m fisheye.utils.materialize_refined_subject_mask_store --refresh-rle --apply`
  - component-scoped dense edits can refresh only the affected compact component
    groups with
    `scripts/py -m fisheye.utils.materialize_refined_subject_mask_store --refresh-rle --components eye_left --apply`
    instead of rewriting unrelated mask components
  - direct in-place mutation of compact RLE rows is intentionally not the review
    write path; painting/editing should mutate dense `masks_roi`, mark compact
    RLE stale, then refresh the edited components from dense
  - edit paths that mutate materialized dense masks must stamp
    `mask_rle_stale = true` plus `mask_rle_stale_at_utc`,
    `mask_rle_stale_reason`, `mask_rle_stale_component_names`, and row summary
    attrs (`mask_rle_stale_row_count`, `mask_rle_stale_row_min`,
    `mask_rle_stale_row_max`)
  - when a component-scoped RLE refresh covers all names in
    `mask_rle_stale_component_names`, the stale marker is cleared; if only a
    subset is refreshed, the stale marker remains with the remaining component
    names
  - modern readers must reject stale compact RLE by default; only diagnostic
    callers should pass `allow_stale_rle=True`, and production consumers should
    use dense `masks_roi` or refresh the compact store first
  - registry sync exposes storage audit fields in
    `subject_mask_performance_latest` and
    `recording_subject_mask_performance_latest`, including dense, bitpacked,
    and RLE logical bytes, backend-reported stored bytes when available,
    dense-cache materialization provenance, compact refresh timestamps, and
    stale row/component scope
  - finalizer attrs include `smart_finalizer_mask_rle_validation_mode` and
    `smart_finalizer_mask_rle_summary`; production cluster runs use invariant
    validation by default, while full dense round-trip validation remains
    available for canaries and audits
- `available_channels`
  - shape: `(C,)`
  - run-level declaration of which components are semantically available in the
    refined run
- `edit_applied`
  - shape: `(N, C)`
  - true when the refined mask row/channel was changed relative to the source
    subject-mask run

Required common `metrics/` arrays:

- `mask_present`
  - shape: `(N, C)`
- `area_px`
  - shape: `(N, C)`
- `centroid_xy`
  - shape: `(N, C, 2)`
- `centroid_valid`
  - shape: `(N, C)`
- `bbox_xyxy`
  - shape: `(N, C, 4)`
- `bbox_valid`
  - shape: `(N, C)`

These are the shared run-level mask geometry arrays. They apply uniformly to
every refined component channel and are represented in code by
`REFINED_SUBJECT_MASKS_SPEC`.

Required lineage arrays:

- `frame_indices`
- `source_crop_row_ids`

Recommended lineage arrays:

- `frame_counts`
- `detection_indices`

## Required attrs

- `source_subject_mask_run`
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`
- `label_schema_id`
- `mask_labels`
- `output_semantics = "multilabel"`
- `refinement_semantics = "canonical_component_masks"`
- `method`
- `created_at_utc`
- `duration_seconds`

Required when the source crop run exposes detect-review linkage:

- `source_detect_review_status_ref`

Required review attrs:

- `refined_subject_mask_review_status`
- `component_review_statuses`

Optional attrs:

- `source_keypoints_run`
- `source_keypoint_group`
- `source_refined_eye_masks_run`
- `source_subject_shape_run`
- `summary_statistics`
- `component_summary_statistics`

Crop-snapshot semantics:

- `source_crop_run` + `source_crop_storage_mode` + `source_crop_signature` +
  `source_crop_revision` form the refined run's portable crop snapshot.
- `source_detect_review_status_ref` remains a separate stable lineage field and
  must not be folded into `source_crop_signature`.
- Current `refined_subject_masks_runs/<run>` writers preserve this crop
  snapshot from the upstream `subject_mask_runs/<run>` source rather than
  re-deriving it from whichever crop run happens to be latest later.

## `available_channels` semantics

`available_channels` means the refined run contains semantically valid refined
data for that component at all.

Meaning:

- `available_channels[c] == true` means component `c` is intentionally present
  in this refined run
- `available_channels[c] == false` means component `c` is a placeholder channel
  and must not be treated as a true negative

Required invariants:

- if `available_channels[c] == false`, then `masks_roi[:, c]` must be all-zero
- if `available_channels[c] == false`, then `edit_applied[:, c]` must be all-false
- if `available_channels[c] == false`, then `metrics/mask_present[:, c]` must be all-false

## `edit_applied` semantics

`edit_applied[n, c]` records whether the refined channel for row `n` differs
from the source subject-mask channel in a way that should be treated as a human
or deterministic refinement, rather than a plain copy-through.

This field is intended to support:

- QA summaries
- training provenance
- future review UI filtering

It does not by itself imply manual editing; the review payload should carry the
review method.

## Current Consumer Status

As of 2026-06-20, Crimson's refined subject-mask reader has been smoke-tested
against a GoodCopBadCop dense refined run using explicit
`source_crop_row_ids -> crop_runs/<source_crop_run>/roi_coordinates_full`
placement. That path no longer relies on refined-mask row position matching crop
row position.

Crimson compact `mask_rle` decode remains pending. Modern editable Palette runs
must carry dense `masks_roi`; compact-only historical runs are display/archive
surfaces and should be materialized to dense with
`scripts/py -m fisheye.utils.materialize_refined_subject_mask_store --apply`
before review/editing.

## Strict Contract Validation

Use the Crimson-facing validator before asking downstream readers to special-case
an archive:

```bash
scripts/py -m fisheye.utils.validate_refined_subject_mask_contract <archive>.zarr
```

Default behavior is validate-only. It resolves
`refined_subject_masks_runs.attrs["latest"]`, checks `mask_labels` /
`available_channels` channel semantics, verifies required run arrays and
run-level metrics, requires available component subgroups to expose
`reason_bytes`, `mask_present`, `area_px`, and `edit_applied`, and fails when
required review or provenance fields are missing.

Backfill is explicit:

```bash
scripts/py -m fisheye.utils.validate_refined_subject_mask_contract <archive>.zarr --backfill
```

The backfill path is intentionally conservative. It may recreate
`available_channels` from declared component availability, recreate `masks_roi`
from component-local mask arrays when channel order is proven by `mask_labels`,
and derive missing mask metrics or component-local mirrors from existing
`masks_roi`. It must not split `eyes_union` into left/right eyes, invent review
state, or fake missing component provenance.

## Review Payloads

Run-level review payload:

- `refined_subject_mask_review_status`

Component-level review payload mapping:

- `component_review_statuses`

Canonical review keys:

- `state`
- `method`
- `intended_use`
- `reviewer`
- `notes`
- `timestamp_utc`

Example:

```json
{
  "refined_subject_mask_review_status": {
    "state": "approved",
    "method": "manual",
    "intended_use": "training",
    "reviewer": "alice",
    "timestamp_utc": "2026-03-10T20:15:00Z"
  },
  "component_review_statuses": {
    "subject_body": {
      "state": "approved",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T20:14:00Z"
    },
    "swim_bladder": {
      "state": "needs_review",
      "method": "manual",
      "intended_use": "training",
      "timestamp_utc": "2026-03-10T20:14:30Z"
    }
  }
}
```

## Component-Scoped Provenance

Run-level `source_subject_mask_run` remains required as the coarse lineage
pointer for the refined run as a whole, but it is not sufficient once one
refined run may contain components seeded from different upstream artifacts.

Canonical home:

- `components/<component_name>/provenance/`

The provenance subgroup should be attrs-only in v1 unless a later contract
needs per-row lineage.

Required provenance attrs for an available component:

- `source_stage`
  - stage family that seeded the component, for example `subject_mask_runs`,
    or transitional `refined_eye_masks_runs`
- `source_run`
  - source run id within that stage family
- `source_method`
  - upstream run `method` used to seed or replace this component
- `source_channels`
  - list of source channel names used to seed this component
- `source_crop_run`
- `source_crop_storage_mode`
- `source_crop_signature`
- `source_crop_revision`

Required when the crop source exposes detect-review linkage:

- `source_detect_review_status_ref`

Recommended provenance attrs:

- `source_label_schema_id`
  - the source run's `label_schema_id`
- `last_update_stage`
  - stage/tool that most recently changed this component in the current refined
    run
- `last_update_mode`
  - recommended values include `create`, `interactive`, `batch`, `projection`
- `last_update_method`
  - method/tool label for the last change
- `updated_at_utc`
  - component-local last update timestamp

Semantics:

- `source_*` identifies the upstream artifact that seeded or replaced the
  component in this refined run
- `source_label_schema_id` is the source artifact's `label_schema_id`
- `source_*` does not imply the current component is byte-identical to that
  source after later edits
- subject-mask finalization is expected to run after seeding and may update
  `last_update_*` while preserving the original `source_*` origin
- `last_update_*` records the most recent operation that changed this component
  inside the current refined run
- if a component is copied through during refined-run creation and never edited,
  writers may record `last_update_mode = "create"`

This distinction is required for future mixed-source refined runs such as:

- `subject_body` seeded from a SAM subject-mask run
- `eye_left` and `eye_right` seeded from a UNet/refined-eye workflow
- `swim_bladder` seeded from a different raw subject-mask source

## Component Subgroups

`components/<component_name>/` is the standard extension point for
component-specific refinement metadata.

Recommended arrays per available component:

- `quality_code`
  - shape: `(N,)`
  - compact machine-generated review-routing enum
- `quality_score`
  - shape: `(N,)`
  - numeric severity for "next problematic frame" navigation
- `reason_bytes`
  - shape: `(N, width)`
  - null-terminated UTF-8 primary encoding
- `reason`
  - shape: `(N,)`
  - optional string mirror
- `mask_present`
  - shape: `(N,)`
- `area_px`
  - shape: `(N,)`
- `edit_applied`
  - shape: `(N,)`

Optional arrays:

- `geometry_valid`
  - shape: `(N,)`
- component-specific quality flags
- component-specific review artifacts

Optional subgroups:

- `provenance/`
  - component-scoped lineage and last-update attrs
- `metrics/`
  - component-local fixed-shape QC summary arrays
- `geometry/`
  - component-local derived geometry arrays
- `contours/`
  - component-local contour stores when contour ownership belongs to one
    component

Optional component attrs:

- `component_schema_id`
- `anatomical_scope`
- component-local policy attrs such as `pectoral_fin_policy`

Recommended current `subject_body` defaults:

- `component_schema_id = "subject_body_v1"`
- `anatomical_scope = "body_core"`
- `pectoral_fin_policy = "excluded_or_unresolved"`

Recommended examples for `components/<component>/metrics/`:

- `component_count`
- `largest_component_fraction`
- `hole_count`
- `hole_area_fraction`
- `area_ratio_prev`
- `area_delta_zscore`
- `centroid_jump_px`
- `bbox_area_ratio_prev`
- `mask_present_gap`
- `component_count_jump`
- `sigma_noise`
- `curvature_var`
- `ipr`
- `solidity`

Common cross-component geometry such as centroid and bbox should stay at
run-level `metrics/`, while component-specific QC should live under
`components/<component>/metrics/`. These component-local arrays are represented
in code by `REFINED_SUBJECT_COMPONENT_METRICS`.

Why per-component subgroups:

- body and swim bladder will not share identical derived geometry
- eye refinement is even more specialized
- this avoids freezing the whole stage around one component’s geometry layout

Review queue policy:

- `quality_code`, `quality_score`, and reason tags are machine-generated
  review-routing signals, not human approval state
- temporal QC should add reason tags and quality score contributions but should
  not overwrite masks by itself
- UI navigation should be able to filter by component and jump to the next row
  with highest unresolved `quality_score`

## Cross-Component Relation Subgroups

`relations/<relation_name>/` is the standard extension point for derived values
that conceptually span multiple components and are not owned by one component.

Canonical example for eye-capable refined runs:

- `relations/eye_pair/metrics/`
  - `separation_px`
  - `separation_valid`

Why this belongs under `relations/` rather than under one eye component:

- `eye_separation` is a pairwise derived value
- duplicating it under both eye components creates synchronization risk
- it should not require inventing a fake mask component such as `eye_pair`
- this relation surface is represented in code by
  `REFINED_SUBJECT_EYE_PAIR_METRICS`

## Geometry Ownership Policy

Refined subject-mask runs own mask-local geometry primitives: values that are
computed directly from one refined component mask and are useful for mask QC,
review navigation, visualization, or lossless downstream reuse. They do not own
interpreted biological coordinate-frame metrics.

Component-specific mask-local geometry should live under:

- `components/<component>/geometry/`

Component-specific contour stores should live under:

- `components/<component>/contours/`

Run-level common mask geometry can stay under `metrics/` when it is fixed-shape
and naturally shared across every component:

- `area_px`
- `centroid_xy`
- `centroid_valid`
- `bbox_xyxy`
- `bbox_valid`

Component-local mirrors are allowed when they make component-native consumers or
review tooling simpler, but the source of truth must remain documented by the
writer's schema attrs.

Recommended component-local primitives:

### `subject_body`

- default display contours under `components/subject_body/sampled_contours/`
- optional full ragged contours under `components/subject_body/contours/` for
  explicit analysis/archive/export builds
- centroid, area, bbox, mask-present, and validity metrics
- simple shape descriptors directly derived from the mask, such as component
  count, hole fraction, solidity, or an unoriented ellipse/PCA summary when the
  convention is explicitly documented
- approximate long-axis QC descriptors directly derived from the mask, such as
  `major_axis_length_px` or `feret_diameter_px`, when the method and sensitivity
  to contour noise are documented
- optional debug seeds for later shape fitting, if they are clearly marked as
  non-canonical seeds rather than final biological body axes

### `swim_bladder`

- default display contours under `components/swim_bladder/sampled_contours/`
- optional full ragged contours under `components/swim_bladder/contours/`
- centroid, area, bbox, mask-present, and validity metrics
- simple blob/ellipse summaries directly derived from the swim-bladder mask

### `eye_left` / `eye_right`

- eye-specific review, reasons, QC, and provenance remain component-local under
  `components/eye_left|eye_right/`
- eye-specific geometry should live under:
  - `components/eye_left/geometry/ellipse_params`
  - `components/eye_left/geometry/ellipse_success`
  - `components/eye_right/geometry/ellipse_params`
  - `components/eye_right/geometry/ellipse_success`
- eye display contour stores should live under:
  - `components/eye_left|eye_right/sampled_contours/points_xy`
  - `components/eye_left|eye_right/sampled_contours/valid`
  - `components/eye_left|eye_right/sampled_contours/source_point_count`
- optional full ragged eye contours live under
  `components/eye_left|eye_right/contours/{ptr,len,points_xy}`
- cross-eye relation metrics should live under:
  - `relations/eye_pair/metrics/separation_px`
  - `relations/eye_pair/metrics/separation_valid`

Geometry policy:

- refined component masks remain the canonical source artifact
- geometry derived from those masks should carry its own validity flags
- geometry primitives stored here should be recomputable from
  the logical mask store (`masks_roi`, compact `mask_bitpacked`, or compact
  `mask_rle` through `MaskStore`) plus the documented method/policy attrs
- downstream `analysis/subject_shape_runs` should consume refined masks and/or these
  mask-local primitives, not raw `subject_mask_runs`

Sampled contour policy:

- `sampled_contours.attrs["schema_id"]` is
  `sampled_component_contours_v1`;
- `points_xy` has shape `(N,K,2)` in ROI-pixel `xy` order, `valid` has shape
  `(N,)`, and `source_point_count` records the pre-sampling contour length;
- current K values are body `128`, eyes `64`, and swim bladder `32`;
- the physical row chunk is `1024` by default, keeping body point payloads near
  1 MiB uncompressed while bounding Crimson row-window reads;
- sampling is uniform closed-contour arc length and is derived directly from
  the authoritative dense mask;
- sampled contours are display caches, never edit/training authority and never
  the source for eye ellipse geometry;
- full ragged contours are an explicit compatibility/analysis/archive/export
  opt-in; Crimson commit `f50bc59` reads sampled contours and retains ragged
  fallback for historical runs.

Metric-QC policy:

- `components/<component>/metrics.attrs["schema_id"]` should be
  `refined_subject_component_mask_metrics_v1`.
- `components/<component>/metrics.attrs["qc_schema_id"]` should be
  `refined_subject_component_metric_qc_reasons_v1`.
- `components/<component>/metrics.attrs["qc_policy"]` records the conservative
  component-specific gates used to derive generated metric-QC reason tags.
- Generated metric-QC reason tags use the `needs_review_metric_*` prefix so
  refresh/backfill tools can replace generated tags without deleting manual
  review tags.
- `scripts/py -m fisheye.utils.backfill_refined_subject_mask_metrics` refreshes
  mask-local metrics and generated metric-QC reason tags for existing refined
  subject-mask runs without recreating mask pixels.

## Boundary With `analysis/subject_shape_runs`

`analysis/subject_shape_runs/<run>` is the analysis home for interpreted
biological geometry that requires a coordinate convention, anatomical polarity,
temporal context, track identity, or relationships between components.

Keep these out of `refined_subject_masks_runs` as canonical outputs:

- body centerline/spline used as an anatomical coordinate frame
- canonical body B-spline fits, including centerline or outline fits with
  smoothing/knot parameters
- canonical biological body length derived from a centerline or B-spline arc
  length
- head/tail-polarized body axis or heading inferred from masks
- body curvature or bend metrics
- swim-bladder position relative to body axis or centerline
- swim-bladder distance to eye pair, body centroid, or anatomical landmarks
- eye angles relative to body/head heading
- temporally smoothed or track-aligned shape metrics

Reasoning:

- `refined_subject_masks_runs` is the curated mask-pixel authority.
- `analysis/subject_shape_runs` is a deterministic derived-analysis layer.
- Recomputing interpreted shape metrics should not mutate or re-author the
  refined masks.
- The shape stage can carry its own method version, source refined-mask run,
  optional source keypoints/heading run, track/temporal context, and failure
  state.

Practical rule:

- If the value answers "what geometry did this one refined component mask have?",
  store it with `refined_subject_masks_runs`.
- If the value answers "what biological pose/shape/relationship does this
  animal have?", store it in `analysis/subject_shape_runs` or a more specific
  downstream analysis run.

Body B-spline rule:

- refined body components may store contours and non-canonical debug seeds
- refined body components may store approximate long-axis QC descriptors such as
  Feret diameter or PCA/ellipse major-axis length
- the canonical B-spline fit belongs in `analysis/subject_shape_runs` because it depends
  on fit method, knot/parameterization policy, smoothing, validity criteria, and
  usually anatomical polarity
- the canonical biological body length should be derived from the validated
  centerline/B-spline arc length in `analysis/subject_shape_runs`, not from raw
  mask area or an unqualified contour diameter

Current implementation note:

- When both `eye_left` and `eye_right` are present,
  `refined_subject_masks_runs` materializes:
  - `components/eye_left|eye_right/geometry/ellipse_params`
  - `components/eye_left|eye_right/geometry/ellipse_success`
  - `relations/eye_pair/metrics/{separation_px,separation_valid}`
- The finalizer can additionally materialize fixed-K sampled contours for all
  selected components. Full ragged `contours/{ptr,len,points_xy}` are controlled
  separately. Production wrappers default to sampled contours on and full
  ragged contours off after the successful 2026-07-10 PRFS/Crimson canary.
- These arrays are derived from the refined subject-mask component masks during
  refined-run creation/finalization.

## Reason Encoding Policy

If `reason_bytes` is present for a component subgroup, writers should also set:

- `reason_encoding = "utf8-null-terminated"`
- `reason_bytes_width = <int>`
- `reason_bytes_null_terminated = true`
- `reason_fallback_order = ["reason_bytes", "reason", "detection_source"]`

Recommended reason tags may include:

- `clean`
- `manual_correction`
- `manual_creation`
- `incomplete`
- `missing_component`
- `geometry_issue`
- `overlap`
- `ambiguous_boundary`

These are examples, not a frozen vocabulary yet.

## Body / Swim-Bladder Expectations In V1

Recommended minimum v1 support:

- `subject_body` refined masks may be available
- `swim_bladder` refined masks may be available
- either component may be unavailable without invalidating the whole run

That means the stage must support cases like:

- body-only refinement run
- swim-bladder-only refinement run
- body + swim-bladder refinement run

without inventing separate stage families.

## Relationship To `refined_eye_masks_runs`

During transition:

- `refined_eye_masks_runs` remains supported for historical archives and
  existing eye-specific tooling
- legacy eye-specific retune/failure tooling may still target standalone
  historical refined-eye runs explicitly
- canonical eye saves and eye review-state changes in
  `refined_subject_masks_runs` may now refresh the matching
  `refined_eye_masks_runs/<run>` as a derived compatibility artifact
- derived compatibility refined-eye runs should be treated as read-only in
  legacy viewers so canonical eye authority does not drift back out of
  `refined_subject_masks_runs`

Target steady-state:

- `refined_subject_masks_runs` is the canonical refined authoring surface for
  new eye-capable subject-mask work
- `refined_eye_masks_runs` becomes a compatibility or adapter artifact rather
  than a second independent canonical authoring surface

Required provenance rule:

- if eye components in `refined_subject_masks_runs` are seeded from
  `refined_eye_masks_runs`, component provenance must point to that true source
  stage/run rather than collapsing everything to the run-level
  `source_subject_mask_run`

## Registry Implications

This stage should eventually project to the registry at two levels:

1. coarse step presence
   - `step_name = "refined_subject_masks"`
2. component-level refined availability and review state
   - `subject_body`
   - `swim_bladder`
   - later eye component(s) if added

The registry must be able to distinguish:

- raw subject-mask availability
- refined body/swim-bladder availability
- refined eye availability projected from unified refined-subject component rows
- specialized refined eye availability during the transition period

Registry stage completion is valid only after the refined run carries a complete
Zarr run-completion marker. `emit_stage_completion(..., status="ok",
step_name="refined_subject_masks", run_name=<run>)` refuses to write `ok` if
`refined_subject_masks_runs/<run>` is missing, incomplete, or only published via
`latest` without `palette_run_completion_status = "complete"`.

## Migration Policy

Recommended transition:

1. keep `refined_eye_masks_runs` unchanged
2. introduce `refined_subject_masks_runs` for body/swim bladder
3. extend `refined_subject_masks_runs` contracts to cover unified eye-local
   geometry and cross-eye relations
4. align registry and review payloads across refined stages
5. move new eye-capable refined authoring to `refined_subject_masks_runs`
6. treat `refined_eye_masks_runs` as a compatibility artifact once adapter
   readers/materializers exist

Implementation note as of 2026-04-02:

- canonical eye edits and eye component review-state updates now materialize a
  compatibility `refined_eye_masks_runs/<run>` view from the canonical
  `refined_subject_masks_runs/<run>` eye components when anatomical
  `eye_left`/`eye_right` components are available
- the compatibility run now serves legacy readers such as eye-specific profile,
  export, and visualization tools, while canonical authoring authority remains
  in `refined_subject_masks_runs`

This contract is intentionally non-destructive.

## Validation Invariants

- all row-aligned arrays share the same first dimension `N`
- at least one physical mask store exists: dense `masks_roi` or compact
  `mask_rle`
- if dense `masks_roi` exists, then
  `masks_roi.shape[1] == available_channels.shape[0] == edit_applied.shape[1]`
- if compact `mask_rle` exists, then `mask_rle.attrs["component_names"]`
  must align with the refined component/channel order
- `metrics/mask_present.shape == metrics/area_px.shape == (N, C)`
- if a component subgroup exists, its per-row arrays must have first dimension `N`
- unavailable channels must remain zero/false across mask/edit/metrics arrays

## Open Questions

- Should `components/<component>/mask_present` and `area_px` remain duplicated
  from `metrics/`, or should one be derived-only?
- At what point should compatibility materialization of `refined_eye_masks_runs`
  become opt-in rather than routine once unified refined-subject eye writes are
  available?

## Related Docs

- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [subject_mask_registry_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_registry_contract.md)
- [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
- [derived_analysis_run_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/derived_analysis_run_contract.md)
- [subject_shape_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_shape_runs_contract.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)
- [pose_kinematics_run_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/pose_kinematics_run_design.md)
