# Crop Storage Producer/Consumer Census

Status: ground-truth census; no crop schema, writer, selector, or profile is
promoted by this document

Date: 2026-07-27

Palette baseline: `e3936b9a`

## Outcome

Palette does not currently have one exact crop storage contract. `crop_runs`
contains several distinct logical and physical profiles:

1. ordinary materialized crops derived from canonical raw detections;
2. acquisition-video-backed geometry-only crops;
3. hybrid acquisition-video plus offline-cache geometry-only crops;
4. clipped-collection proxy crops;
5. an explicitly historical depth-one composite representation; and
6. materialized training artifacts that also happen to use `crop_runs`.

The right next step is to freeze one immutable crop observation table and then
bind an exact source-pixel provider to it. The future analysis-archive profile
is geometry-only: it does not persist `roi_images`. It would be unsafe to turn
the current union of optional arrays into one permissive schema. Dense training
artifacts, compatibility proxies, and scratch/work-package pixel caches must
remain separate contracts.

It is safe to implement that logical contract and a selector-ineligible writer
now. It is not yet safe to replace the production crop writer or selectors:
the exact refined-detection-to-crop coordinate lineage and downstream
completeness gates remain open
(`docs/refined_detection_crop_handoff_contract_v1.md:73-102`).

## Why Crop Geometry Remains Separate From Detection

Crop geometry should not be folded into the canonical detection schema.
Detection and crop answer different questions and change for different reasons:

- detection records what observation exists, its stable `instance_key`, its
  acquisition frame, and its authoritative image-space bbox;
- crop records which exact source-pixel window a downstream purpose requests
  around that observation; and
- a cache or training artifact records how those source pixels were decoded and
  transformed into a model-facing tensor.

The dependency is:

```text
immutable detection snapshot
  + versioned crop policy
  + source-pixel authority
  -> immutable geometry-only crop snapshot
  -> optional work-package/cache/training materialization
```

One detection snapshot may legitimately have several crop snapshots: a tight
keypoint window, a larger context window for segmentation, a variable-size
inspection window, or a resized/padded training input. Putting crop geometry in
the detection contract would either allow only one of those policies or require
revising/duplicating detection identity whenever a crop policy changes.

Separation must not mean weak association. A crop run manifest should bind:

- the exact source detection/refined-detection run and manifest digest;
- the source logical-content digest and recording identity;
- the versioned crop-policy payload and digest;
- the exact source-pixel authority and digest; and
- complete `instance_key`/source-row coverage or an explicit subset contract.

Every downstream product should cite an exact crop run ID and manifest digest.
A single undifferentiated `crop_runs.latest` is insufficient when multiple crop
policies are valid. A future selector may identify a default by purpose/profile,
but it must not mutate the source detection manifest or make crop policy part of
detection identity.

Persisting geometry is still worthwhile even when a crop could be recomputed
from bbox plus policy: the arrays cheaply freeze rounding, padding, per-row
size, source-pixel routing, and row order against implementation drift. The
writer should validate those persisted values against the bound detection and
policy rather than ask every consumer to rederive them.

## Method And Scope

The checked-in static writer census reports 104 crop-classified creation sites
across 13 files: 45 production, 57 training, and 2 diagnostic. This is a count
of statically classified array-creation calls, not 104 independent crop schemas.
It also misses dynamically resolved sites in the primary `tracking/crop.py`
writer, so it is evidence rather than the contract
(`docs/diagnostics/zarr_production_writer_census.json`).

This census therefore combines:

- the generated writer census;
- direct inspection of every maintained `crop_runs` producer;
- the shared `CropImageSource` and materialized-only readers;
- selector, identity, coordinate, and pixel contracts; and
- current shared storage-planning infrastructure.

Training exports are inventoried because they use the same group name, but they
are not candidates for the analysis-archive crop schema.

## Producer Profiles

| Profile | Current producer | Pixel representation | Publication behavior | Classification |
| --- | --- | --- | --- | --- |
| Ordinary crop | `tracking/crop.py` | top-level dense `uint8 roi_images` | creates an ineligible candidate, validates coordinates, then updates crop selectors | Current production; only canonical raw `detect_runs/<run>` is admitted by the hardened publication preflight (`src/fisheye/tracking/crop.py:617-633`, `src/fisheye/tracking/crop.py:826-842`) |
| Incremental standalone | `tracking/incremental_crop.py` | top-level dense `uint8 roi_images`; matching rows may be copied from a prior complete run | creates a fresh immutable run and selects it only after source revalidation and readback | Implemented reference materializer; still raw-detection-bound (`src/fisheye/tracking/incremental_crop.py:1265-1317`, `src/fisheye/tracking/incremental_crop.py:1639-1674`) |
| Refined handoff | `shared/zarr/refined_detection_crop_source.py` plus `tracking/refined_detection_crop_handoff.py` | none | validates and plans only; receipt says `crop_publication_authorized=false` | Future-facing input boundary, not a writer (`src/fisheye/shared/zarr/refined_detection_crop_source.py:278-308`, `src/fisheye/tracking/refined_detection_crop_handoff.py:77-93`) |
| Acquisition crop video | `utils/build_analysis_acquisition_crop_run.py` | geometry plus an external acquisition crop video | keeps legacy `latest` materialized-compatible and advances `latest_any` | Current specialized geometry-only producer (`src/fisheye/utils/build_analysis_acquisition_crop_run.py:368-385`, `src/fisheye/utils/build_analysis_acquisition_crop_run.py:452-520`) |
| Hybrid acquisition/offline | `utils/build_hybrid_acquisition_offline_crop_run.py` | acquisition crop video for online rows; flat ROI cache for recovered rows | optional specialized `latest_any` publication | Current specialized geometry-only producer (`src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:688-741`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:847-918`) |
| Clipped proxy | `utils/create_clipped_collection_proxy_crop_run.py`; `utils/merge_clipped_proxy_crop_runs.py` | geometry plus external clipped-collection ROI cache | `stage_selector_eligible=false`, completion status `auxiliary` | Compatibility/collection adapter, not a general crop authority (`src/fisheye/utils/create_clipped_collection_proxy_crop_run.py:345-379`, `src/fisheye/utils/merge_clipped_proxy_crop_runs.py:275-304`) |
| Historical composite | `tracking/incremental_crop.py` | exact base-row or dense delta-row mapping | remains unselected; promotion is forbidden by the CLI | Implemented historical migration canary, not the future default (`src/fisheye/tracking/incremental_crop.py:1760-1825`, `src/fisheye/utils/materialize_incremental_crop.py:80-103`) |
| Training crop table | several export/append utilities | self-contained dense `uint8 roi_images` | training-specific run selection and provenance | Separate immutable training-artifact contract; never infer analysis semantics from it (`docs/crop_storage_mode_migration_todo.md:185-201`) |
| Scratch/work package | `CropImageSource`, flat cache, crop pixel work package | temporary or durable derived cache outside canonical crop authority | never changes canonical selectors | Runtime/compute artifact, not `crop_runs` authority (`docs/composite_crop_storage_contract.md:26-70`) |

## Current Array Inventory

`N` is the crop-observation row count and `F` the acquisition-frame count.
Where dense derived pixels exist outside the future analysis profile, `H,W`
denote that artifact's declared height and width. Neither the logical crop
contract nor its storage policy assumes 512×512. “Core candidate” means the field should be
considered for the future exact crop observation table; it is not a frozen
decision in this census.

### Identity And Frame Lookup

| Array | Observed shape/dtypes | Current writers/readers | Census disposition |
| --- | --- | --- | --- |
| `instance_key` | `[N] uint64` in modern ordinary/incremental/proxy paths | copied from the source detection rowset; incremental capture requires integer, nonnegative, unique keys (`src/fisheye/tracking/incremental_crop.py:494-526`) | Required core candidate. Observation/edit lineage only; not subject or track identity. |
| `source_refined_row_ids` | `[N] int64` where refined lineage exists | copied from refined `refined_row_ids`; absent or `-1` for non-refined rows (`src/fisheye/tracking/refined_detection_crop_handoff.py:63-70`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:710-717`) | Required only for a refined-source lineage profile, or required-with-sentinel if one unified source-union schema is chosen. |
| `source_detect_row_index` | `[N] int32` in ordinary metadata, but `[N] int64` in proxy/hybrid writers | raw-candidate lineage; manual refined additions may be `-1` (`src/fisheye/tracking/crop.py:1418-1438`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:719-726`) | Dtype conflict that must be frozen before publication. Prefer the refined-v1 source dtype rather than preserving writer accidents. |
| `detection_indices` | `[N] int32` or `int64` | current writers generally store a physical ordinal (`src/fisheye/tracking/crop.py:3641-3648`, `src/fisheye/tracking/crop.py:799-804`) | Compatibility candidate, not stable identity. Likely omit from the new core once exact source lineage and `instance_key` are required. |
| `frame_indices` | `[N] int32` or `int64` | ordinary canonical output equates it to acquisition-frame index; acquisition output uses int32; proxy/incremental use int64 (`src/fisheye/tracking/crop.py:805-815`, `src/fisheye/utils/build_analysis_acquisition_crop_run.py:322-348`) | Required core candidate, exact `int64`, sorted nondecreasing. Define as the row lookup frame domain, not a generic source-frame alias. |
| `source_acquisition_frame_index` | `[N] int64` | required by hardened ordinary coordinate publication, missing from specialized geometry-only profiles (`src/fisheye/tracking/crop.py:805-815`) | Required core candidate for full-acquisition crops. Clipped/alternate frame domains need an explicit binding profile. |
| `source_frame_indices` | `[N] int64` | acquisition/proxy compatibility alias of `frame_indices` | Compatibility-only unless a distinct source domain is defined. |
| `frame_counts` | `[F] int32` or `int64` | ordinary `np.bincount` writes platform integer, acquisition writes int32, incremental writes int64 (`src/fisheye/tracking/crop.py:3632-3639`, `src/fisheye/tracking/incremental_crop.py:1054-1057`) | Derived compatibility array. Do not make it the future lookup authority. |
| `frame_row_offsets` | absent from maintained crop writers | no current crop consumer can rely on it | Required future core candidate: exact `[F+1] int64` CSR offsets, with `[offsets[f], offsets[f+1])` covering zero, one, or many crop rows. |
| `source_row_signature` | `[N,32] uint8` | persisted only by modern incremental materialization; proxies use a tightly scoped bootstrap (`src/fisheye/shared/crop_snapshot_identity.py:162-238`) | Required future core candidate. It binds pixel source, frame, bbox, ROI settings, and reuse compatibility. |

### Geometry

| Array | Observed shape/dtypes | Authority/derivation | Census disposition |
| --- | --- | --- | --- |
| `bbox_norm_coords` | `[N,4] float32` in canonical/acquisition paths; hybrid rewrites float64 | source normalized `[cx,cy,w,h]` geometry (`src/fisheye/utils/build_analysis_acquisition_crop_run.py:298-320`, `src/fisheye/utils/build_hybrid_acquisition_offline_crop_run.py:675-680`) | Required core candidate, exact float32, authoritative as in detection v1. Hybrid float64 is a compatibility profile. |
| `bbox_img_xyxy` | `[N,4] source float`, currently float32 or float64 | required image-space derivative | Required core candidate, exact float32 and digest-validated from persisted normalized authority. |
| `centers_img_xy` | `[N,2] source float` | exact source-camera center; current ordinary placement rounds this surface (`src/fisheye/tracking/incremental_crop.py:817-864`) | Required core candidate, exact float32. |
| `source_crop_xywh` | `[N,4] source float`, float32 or float64 | source-camera placement of each ROI | Required derived coordinate surface, exact float32. Validate it against the exact integer origin/size arrays rather than using floating values to choose source pixels. |
| `roi_coordinates_full` | `[N,2] int32` | exact integer source-pixel top-left; equal to `source_crop_xywh[:,:2]` under the ordinary contract | Required core candidate and extraction-window authority. Retain the existing name for compatibility, but freeze axis order as `[x,y]`. |
| `bbox_roi_xyxy` | `[N,4] source float` | ROI-local bbox derived from image bbox and crop placement | Required core candidate, exact float32 with explicit ROI-to-camera transform. |
| `roi_sizes_full` | `[N,2] int32`, ordered `[width,height]` in the current acquisition writer | exact integer extraction-window size | Required core candidate. Repeating a fixed run size is modest redundancy, but it gives fixed-size and per-row-variable crops one exact row schema and removes dependence on a scalar attr for pixel reconstruction. |
| `bbox_crop_norm_coords` | `[N,4] float32/float64` | acquisition-video compatibility projection | Derived profile field, not common core. |
| `roi_coordinates_ds` | `[N,2] int32` | historical downsampled placement | Legacy-only; the current documentation already warns that its frame is ambiguous (`src/fisheye/docs/zarr_structure.md:348-357`). |
| `selected_live_detection_bbox_*`, `realtime_detection_bbox_roi_xyxy` | aliases of acquisition geometry | duplicated by acquisition and hybrid writers (`src/fisheye/utils/build_analysis_acquisition_crop_run.py:417-425`) | Acquisition compatibility aliases. Do not admit them to the new common schema. |

### Pixel Representations And Source Bindings

| Array/group | Shape/dtype | Meaning | Census disposition |
| --- | --- | --- | --- |
| `roi_images` | `[N,H,W] uint8` | complete materialized grayscale crop tensor | Excluded from the future analysis-Zarr contract. It remains an immutable derived payload in training Zarrs, compatibility runs, or caches/work packages. The pixel contract already fixes uint8, C order, row order, coordinates, and zero padding (`src/fisheye/shared/roi_pixel_contract.py:23-65`). |
| `composite_payload/source_codes` | `[N] uint8` | row selects base or delta | Historical composite only. |
| `composite_payload/source_row_indices` | `[N] int64` | physical row in selected base/delta source | Historical composite only. |
| `composite_payload/delta_target_row_indices` | `[D] int64` | target rows materialized in delta | Historical composite only. |
| `composite_payload/delta_instance_key` | `[D] uint64` | exact delta observation identity | Historical composite only. |
| `composite_payload/roi_images_delta` | `[D,H,W] uint8` | dense changed-row payload | Historical composite only. The depth-one payload and checks are defined at `docs/composite_crop_storage_contract.md:84-140`. |
| `source_crop_video_frame_indices` | `[N] int64 | -1` | frame in acquisition crop video | Acquisition/hybrid representation binding. |
| `source_pixel_kind_codes` | `[N] int8` | selects acquisition video versus supplemental cache | Acquisition/hybrid representation binding. |
| `supplemental_cache_row_indices` | `[N] int64 | -1` | row in supplemental flat cache | Hybrid representation binding. |
| `source_crop_meta_row_indices`, `source_crop_local_frame_ids`, `source_recording_frame_ids` | `[N] int64` | Orange acquisition provenance | Acquisition-specific lineage, not common crop geometry. |
| `source_clip_indices`, `source_clip_local_frame_indices` | `[N] int64` | clipped-collection lineage | Clipped profile only. |
| `source_crop_row_ids` | `[N] int64` | proxy-local/direct crop row mapping | Compatibility/profile lineage. A new canonical run already has an unambiguous physical row index; do not treat this as stable identity. |
| `source_proxy_crop_run_index`, `source_proxy_crop_row_ids` | `[N] int32`, `[N] int64` | merge provenance for per-clip proxies | Merged-proxy-only. |
| `detection_success`, `detection_source`, `crop_state_codes` | `[N] bool/int8/int8` | acquisition or historical status labels | Profile/compatibility fields. Refined-v1 reason/source registries should not be weakened into these older integer labels. |

### Materialization Plan

Incremental standalone and composite runs also persist a
`materialization_plan` group through the keyed-delta helper. It is publication
evidence, not part of the crop observation table. The writer uses
`instance_key` plus `source_row_signature` to choose copy, compute, and omit
actions and processes dense pixels a complete output chunk at a time
(`src/fisheye/tracking/incremental_crop.py:1438-1475`).

## Current Physical Layouts

| Surface | Current policy | Consequence at representative scale |
| --- | --- | --- |
| Ordinary canonical coordinate arrays | direct `create_array`, usually 1,000 rows per chunk (`src/fisheye/tracking/crop.py:783-815`) | At 1.188M rows, each such array can create about 1,188 payload objects. The policy is row-count based and unsharded. |
| Ordinary `roi_coordinates_full` | `min(chunk_size,N)` rows, unsharded (`src/fisheye/tracking/crop.py:4382-4387`) | Layout inherits a processing constant rather than access bytes. |
| Geometry-preload helper | 16,384 rows, all trailing axes, no outer shard (`src/fisheye/shared/zarr/chunk_profiles.py:15-48`, `src/fisheye/shared/zarr/chunk_profiles.py:94-128`) | A scalar int32 chunk is 64 KiB, int64 is 128 KiB, float32 `[N,4]` is 256 KiB. One row count produces inconsistent byte sizes. |
| Incremental compact arrays | `columnar.store_array`: 4,096 rows for 1-D, 1,024 for wider arrays; outer shard requested at 131,072 rows (`src/fisheye/shared/zarr/columnar.py:39-94`, `src/fisheye/tracking/incremental_crop.py:1037-1089`) | Object count is improved, but inner chunks range from about 8 KiB to 32 KiB for common crop metadata. Both inner and outer sizes vary with bytes per row. |
| Materialized `roi_images` | default 32 rows, full `H,W`, Blosc LZ4 level 1 bitshuffle, normally unsharded (`src/fisheye/shared/crop_roi_layout.py:8-54`, `src/fisheye/shared/crop_roi_layout.py:85-104`) | At the illustrative 512×512 uint8 size, one inner chunk is 8 MiB and 1.188M rows create about 37,125 unsharded payload objects. This is a current compatibility/training/cache concern, not the target analysis-Zarr layout. Other ROI sizes produce proportionally different bytes, which is exactly why future materializers must plan from bytes and access units. |
| External ordinary `roi_images` | caller-controlled row chunks and optional row shards; defaults are still row-count based (`src/fisheye/tracking/crop.py:4039-4055`) | Layout can range widely across invocations and is not a stable schema declaration. |
| Scratch ROI cache | 128-row uncompressed, unsharded default (`src/fisheye/shared/crop_roi_layout.py:8-11`, `src/fisheye/shared/crop_roi_layout.py:57-71`) | Deliberately a local throughput cache; it must not define published storage. |

The shared planner already computes chunks and shards from uncompressed bytes,
access shape, and write mode, and records object estimates
(`src/fisheye/shared/zarr/storage_intent.py:38-180`,
`src/fisheye/shared/zarr/storage_planner.py:148-225`). The policy-owned array
factory enforces exact dtype, Zarr v3, storage-plan identity, and codec profile
(`src/fisheye/shared/zarr/array_factory.py:183-248`). No maintained crop writer
uses that boundary today.

## Consumer Census And Access Patterns

### Shared mixed-mode readers

`CropImageSource` is the maintained mixed-mode boundary. Automatic selection is
`latest_any`, then legacy `latest`, then `latest_materialized`; the traditional
resolver separately requires literal materialized `roi_images`
(`src/fisheye/shared/crop_image_source.py:118-156`,
`src/fisheye/shared/crop_image_source.py:159-231`).

On open, it eagerly loads complete `roi_coordinates_full` and `frame_indices`
arrays, then resolves one of materialized, composite, acquisition-video,
external-video, or cache-backed pixels
(`src/fisheye/shared/crop_image_source.py:740-819`). Pixel payloads are consumed
by contiguous slices or arbitrary row-index lists
(`src/fisheye/shared/crop_image_source.py:1129-1164`).

Maintained consumers using this boundary include:

- YOLO keypoints and retry;
- U-Net/SAM/other modern subject-mask inference;
- crop work-package and flat-cache construction; and
- several diagnostics and overlay visualizers.

Representative call sites are
`src/fisheye/detection/detect_keypoints_yolo.py:1845-1858`,
`src/fisheye/segmentation/infer_unet_subject_masks.py:1986-2000`, and
`src/fisheye/utils/run_sam_subject_masks.py:598`.

### Materialized-only and direct readers

Many consumers still dereference `roi_images` directly or call the explicit
materialized resolver. These include:

- traditional keypoint detection
  (`src/fisheye/detection/detect_keypoints_traditional.py:112`,
  `src/fisheye/detection/detect_keypoints_traditional.py:468`);
- training loaders and exporters
  (`src/fisheye/training/zarr_subject_mask_dataset.py:218`,
  `src/fisheye/training/zarr_yolo_dataset_loader.py:1296`);
- keypoint and subject-mask tuning/review
  (`src/fisheye/tune/keypoint_tuner.py:947`,
  `src/fisheye/tune/refined_subject_mask_review.py:4052`);
- labeling and patch/repair utilities
  (`src/fisheye/labeling/web_runtimes.py:424`,
  `src/fisheye/utils/patch_crops_from_refined.py:466`); and
- historical visualization/export tools
  (`src/fisheye/visualization/visualize_crops.py:216`,
  `src/fisheye/utils/export_keypoint_training_zarr.py:829`).

This does not require the new analysis contract to mandate dense pixels for
every representation. It requires explicit consumer capabilities: mixed-mode
readers use `CropImageSource`; materialized-only readers fail closed or first
materialize/export an exact standalone artifact.

### Access classes for planning

| Array family | Current read pattern | Initial planner classification to benchmark |
| --- | --- | --- |
| `frame_row_offsets` | not yet present; future frame lookup | `EAGER`; retain complete index once, as with detection offsets |
| identity, frame, lineage, and geometry columns | eager whole-array in `CropImageSource`; row/window reads in review/export code | `WINDOWED` is the scalable default, while allowing small arrays to collapse to one chunk/shard |
| derived `roi_images` / `roi_images_delta` outside analysis Zarrs | sequential inference batches and random single-row review | `PER_ROW`, access unit `(1,H,W)`; compute chunk rows from actual dtype and `H,W`, never a fixed crop-size assumption |
| composite mappings | row-indexed resolution, often in requested batches | `INDEXED` or `WINDOWED`, aligned with the logical crop row grid |
| small code maps and fixed metadata tables | whole-array | `EAGER` |

## Selector And Publication Findings

1. There is no exact crop run manifest comparable to canonical/refined
   detection manifests. Current crop producers primarily use mutable group attrs,
   completion attrs, and stage provenance.
2. Crop writers do not consolidate metadata or validate direct/consolidated
   declaration equivalence. The exact detection publication flow does both, but
   crop publication has not adopted it.
3. Automatic mixed-mode selection checks completion and selector eligibility,
   but an explicitly named crop run is returned without those checks
   (`src/fisheye/shared/crop_image_source.py:133-155`). The materialized resolver
   similarly checks representation but not explicit-run completion
   (`src/fisheye/shared/crop_image_source.py:176-208`). A public v1 reader should
   validate explicit input fail-closed, with a distinct benchmark/repair option
   for ineligible artifacts.
4. Historical pointers have overlapping meanings: `latest`,
   `latest_complete`, `latest_materialized`, `latest_any`, and
   `latest_composite`. The incremental writer updates several together, while
   geometry-only and composite writers intentionally update subsets
   (`src/fisheye/tracking/incremental_crop.py:1666-1674`,
   `docs/composite_crop_storage_contract.md:156-174`).
5. A future crop authority must identify both the logical observation snapshot
   and its pixel representation. A geometry-only run and its exact materialized
   realization should not compete through ambiguous “latest” semantics.

## Schema And Documentation Drift

The existing `CROP_SPEC` is a compatibility validator, not the current writer
contract. It declares `frame_indices`, `frame_counts`, and
`detection_indices` as int32, makes `instance_key` optional, and omits
`source_acquisition_frame_index`, `centers_img_xy`, `source_crop_xywh`,
`bbox_roi_xyxy`, `source_row_signature`, and any frame offsets
(`src/fisheye/shared/zarr/stage_arrays.py:1130-1166`).

The descriptive structure document is closer to the hardened ordinary writer,
but still states that future crops only accept raw detections. That remains true
of current publication code, while the newer refined handoff can now validate
and plan from refined-v1 but deliberately cannot publish
(`src/fisheye/docs/zarr_structure.md:397-424`,
`docs/refined_detection_crop_handoff_contract_v1.md:73-100`).

Other drift to resolve:

- `frame_counts` has conflicting int32/int64 implementations;
- `source_detect_row_index` has conflicting int32/int64 implementations;
- hybrid acquisition geometry is float64 while canonical detection geometry is
  float32;
- ordinary and incremental physical layouts are unrelated;
- specialized acquisition writers duplicate bbox arrays under aliases; and
- no crop array currently has an exact shared `ArrayContract` binding. The
  shared catalog stops at detection/refined detection, keypoints, masks, and
  contours (`src/fisheye/shared/zarr/array_contracts.py:320-838`).

## Recommended Contract Boundary

Freeze three layers rather than one permissive union:

### 1. Immutable crop observation snapshot

The candidate common row table should contain exact observation identity,
frame lookup, source-detection binding, source-camera geometry, crop placement,
ROI-local geometry, and row signatures. It should support `N != F`, empty
frames, and multiple observations per frame.

Likely required fields are:

```text
instance_key                    uint64[N]
frame_indices                   int64[N]
source_acquisition_frame_index  int64[N]
frame_row_offsets               int64[F+1]
bbox_norm_coords                float32[N,4]
bbox_img_xyxy                   float32[N,4]
centers_img_xy                  float32[N,2]
source_crop_xywh                float32[N,4]
roi_coordinates_full            int32[N,2]
bbox_roi_xyxy                   float32[N,4]
source_row_signature            uint8[N,32]
```

The source-lineage envelope should then select an exact raw, refined,
acquisition, or clipped binding with no silent fallback. This census does not
yet decide whether source-specific row arrays are required-with-sentinel at the
run root or live in typed subgroups.

### 2. Analysis pixel-provider profile

Analysis Zarrs do not persist crop images. Bind exactly one declared provider
to the observation snapshot:

- `source_video_geometry_v1`: no dense pixels; exact source-video and decode
  contract;
- `acquisition_crop_video_v1`: exact crop-video frame mapping;
- `hybrid_acquisition_cache_v1`: exact per-row source codes and supplemental
  cache manifest.

The crop observation table is authoritative for geometry and lineage. Dense
pixel payloads may be generated into a keyed work package, node-local/shared
cache, or training artifact, but they are not copied back into the canonical
analysis run. Bounding-box edits occur in a new refined-detection revision and
generate a successor crop snapshot or invalidate/rebuild keyed pixel caches;
crop pixels are never edited in place in the new contract.

### Crop-size semantics

Crop size is data, not a schema-wide constant. The common observation table
should persist both the exact integer top-left and exact integer width/height
for every row. A run may declare one of two validated modes:

- `fixed_per_run`: every `roi_sizes_full` row equals the declared run size; or
- `variable_per_row`: positive width/height may differ across rows.

Geometry-only analysis supports either mode naturally. A dense Zarr tensor
cannot represent variable `H,W` in one rectangular `[N,H,W]` array without a
separate transformation. A training or cache materializer must therefore make
one explicit choice: partition rows by shape, pad to a declared shape, or
resize to a declared model input. That transformation and its interpolation,
padding, and source crop-manifest digest belong in the derived artifact's
contract. Ragged/object arrays should not be introduced implicitly.

### 3. Artifact class

Record whether the containing archive is:

- geometry-only analysis publication;
- self-contained training artifact;
- selector-ineligible benchmark/canary;
- compatibility proxy; or
- scratch/work-package cache.

Artifact class controls retention and selection, not array dtype. Training
Zarrs remain dense and may be sharded because they are immutable publications;
their chunk rows are derived from the actual dtype and declared `H,W`.
Editability belongs to their authoring source, not to the published training
tensor.

## Implementation Checklist

### Contract freeze

- [ ] Name/version the crop observation schema independently of physical
      profiles and run-manifest versions.
- [ ] Freeze exact dtypes, shapes, axis names, fill values, and row ordering.
- [ ] Require unique `instance_key`; state again that it is not subject identity.
- [ ] Require sorted `frame_indices` and exact int64 `frame_row_offsets` with
      shape `F+1`, first value zero, last value `N`, and exact agreement with
      rows.
- [ ] Freeze raw/refined/acquisition/clipped source-lineage envelopes.
- [ ] Define crop-snapshot identity from the source detection manifest digest,
      crop-policy digest, and source-pixel authority digest; never include crop
      policy in detection identity.
- [ ] Decide whether `detection_indices`, `frame_counts`, and
      `source_frame_indices` are omitted or explicitly compatibility-only.
- [ ] Require exact `roi_sizes_full [N,2] int32` and freeze
      `fixed_per_run | variable_per_row` semantics; do not assume 512×512.
- [ ] Freeze the authoritative pixel/decode contract for every representation.

### Executable shared schema

- [ ] Add crop `ArrayContract` declarations and one exact schema binding under
      `fisheye.shared.zarr`.
- [ ] Add strict decoded-array validation, unexpected-array rejection, derived
      geometry validation, offsets validation, and signature validation.
- [ ] Add a strict run-manifest envelope with source snapshot digest, recording
      identity, pixel representation, logical-content digest, storage plans,
      writer ownership, and publication state.
- [ ] Add exact direct/consolidated metadata normalization and digest validation.
- [ ] Keep `CROP_SPEC` as a named legacy/compatibility validator or replace it
      only after all callers are migrated.

### Physical planning

- [ ] Route every new crop array through `ArrayContract -> ArrayIntent ->
      StoragePlan -> create_array_from_plan`.
- [ ] Plan from uncompressed bytes and access units, never processing-frame or
      arbitrary row constants.
- [ ] Start tabular arrays with the general published profile and benchmark any
      crop-specific override.
- [ ] For training and cache materializers, benchmark `roi_images` using each
      artifact's actual dtype and `H,W`. The 512×512 one-row/four-row cases are
      illustrative inputs, not fixed policy.
- [ ] Require whole non-overlapping chunk/shard ownership for parallel writes.
- [ ] Record resolved chunk/shard shapes, logical bytes, object estimates, codec
      chain, and effective worker ownership in the run manifest.

### Writer and reader integration

- [ ] First implement a selector-ineligible geometry-only analysis writer from
      the exact refined-v1 binder; do not create `roi_images` in the analysis
      archive.
- [ ] Preserve exact `instance_key` and `source_refined_row_ids`; compute new or
      changed rows and copy only signature-equal predecessor rows.
- [ ] Keep acquisition and clipped adapters outside the common writer until each
      can produce a complete typed source binding.
- [ ] Make explicit public-reader selection validate completion, manifest,
      selector eligibility, and logical/storage declarations.
- [ ] Replace ambiguous single-latest assumptions with exact crop-run binding or
      a typed purpose/profile selector when more than one crop policy is valid.
- [ ] Inventory remaining direct `roi_images` readers into either intentional
      materialized-only consumers or `CropImageSource` migration targets.
- [ ] Keep training exporters dense and self-contained; allow geometry-only
      analysis sources only through an explicit materialization step.

### Canary and activation

- [ ] Publish a small selector-ineligible refined-source geometry-only crop
      canary with at least two ROI sizes represented in contract tests.
- [ ] Validate exact values, multi-instance frames, empty frames, offsets,
      source lineage, pixel parity, direct/consolidated metadata, and codecs.
- [ ] Benchmark publication, object count, eager metadata open, random single-row
      review, sequential inference, and training export.
- [ ] Run one full-duration regression only after the small contract canary
      passes.
- [ ] Require downstream keypoint, subject-mask, tracking, and training
      completeness before refined/crop authority activation.
- [ ] Activate with an owner/generation-guarded final metadata transaction; do
      not rewrite scientific payload during selector activation.

## Immediate Next Implementation Slice

The smallest safe code slice is:

1. add exact crop array contracts for the proposed common observation table,
   including per-row `roi_sizes_full`;
2. add `CropDimensions` and `CropSchema` logical validation with no Zarr writes;
3. derive `frame_row_offsets` and validate a multi-instance/empty-frame fixture;
4. add a crop storage-plan function using the existing byte-budget planner;
5. add strict manifest builders/validators for only the geometry-only analysis
   provider profile; and
6. build a selector-ineligible in-memory/shadow writer test from the already
   validated refined-v1 handoff, without a dense pixel array.

Acquisition, clipped, composite, training, selector, and production-writer
migrations should follow as explicit adapters after that core passes. This
keeps current processing safe while giving the parallel producer/DAG work one
stable contract API to target.
