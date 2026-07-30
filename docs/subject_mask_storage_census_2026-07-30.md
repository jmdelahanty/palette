# Subject-Mask Storage Census

Status: read-only ground-truth investigation; this document does not promote a
schema, storage profile, writer, selector, registry entry, or archive

Date: 2026-07-30

Palette baseline: `6da8ac904087551696c3f8f740fe5c4817b237f3`

## Outcome

Palette already has a strong scientific direction for modern segmentation:

- raw `subject_mask_runs` are probability-first;
- modern `refined_subject_masks_runs` have physically present dense `uint8`
  `masks_roi` as their editable and training authority;
- compact bitpacked/RLE masks, contours, and measurements are derived caches;
  and
- mask observations are bound to stable `instance_key` and exact crop
  geometry rather than inferred from row ordinal.

The missing piece is one exact, executable storage and publication contract.
Subject-mask writers still choose fixed row counts and call Zarr creation
directly instead of using the shared byte-derived `StoragePlan`. Generic stage
schemas still admit legacy aliases and compact-only refined runs that the
future-facing coordinate publisher correctly rejects. Promotion preserves the
draft's physical chunks rather than replanning a logically identical immutable
snapshot.

The next design step should freeze four related but distinct profiles:

1. immutable raw probability snapshots;
2. mutable refined drafts with dense mask authority;
3. immutable refined publications with regenerated read-optimized caches; and
4. immutable dense training artifacts.

It would be unsafe to collapse these access and mutability patterns into one
permissive schema or one physical layout.

## Scope And Method

This census used only read-only operations. It combined:

- direct inspection of maintained raw, refined, compact-cache, promotion,
  review, training-export, and training-reader code;
- comparison with the shared array-contract, storage-intent, storage-profile,
  and byte-planning infrastructure;
- direct Zarr v3 metadata inspection of the selected Sleepyfish raw and refined
  runs, without decoding or hashing mask pixels; and
- review of checked-in subject-mask design and benchmark documents.

No writer was invoked. No archive, selector, registry, branch, source file, or
metadata node was changed as part of the investigation.

The inspected full-duration archive was:

```text
/groups/johnson/johnsonlab/jeremy/recordings/
  sleepyfish_2026_05_05_17_45_30_cam2010095/zarr/
  sleepyfish_2026_05_05_17_45_30_cam2010095_analysis.zarr
```

Potential object counts below are metadata-grid estimates. They do not require
opening, decoding, or hashing every stored object.

## Producer/Consumer/Lifecycle Specification

This section is a proposed contract boundary derived from current code. It is
not a claim that all four profiles are implemented today.

### Cross-profile invariants

| Invariant | Recording-bound raw/refined profiles | Merged training profile |
| --- | --- | --- |
| Format | Zarr v3 with an exact schema/profile ID; no dtype probing or aliases on the canonical path | Zarr v3 with a distinct immutable training-artifact schema/profile ID |
| Observation identity | Required unique `instance_key : uint64[N]`; identity of an observation/edit lineage, never an animal or track | Required `(source_dataset_id, instance_key)` unless global key uniqueness is separately proven |
| Crop binding | Exact crop run/snapshot identity, manifest/content digest, selected `source_crop_row_ids`, source-pixel authority, and coordinate descriptors | Exact source crop and mask manifest/content digests for every source dataset row |
| Frame domain | Acquisition frame plus exact `frame_row_offsets : int64[F+1]`, either physically owned or explicitly bound to the crop index | Preserve source dataset plus source acquisition frame; do not invent one ambiguous global frame axis |
| Row ordering | Freeze one order, preferably `(source_acquisition_frame_index, instance_key)`; repeated offsets represent empty or multi-observation frames | Freeze export row order and bind source-row identity independently of training shuffle order |
| ROI extent | One run declares homogeneous `H,W`, or an exact padding/valid-window contract; never assume 512x512 globally | One exact pixel tensor profile per artifact; RGB and luma require distinct schema/profile IDs |
| Storage | Every array declares access, mutability, writer ownership, codec, `StoragePlan`, and object/byte estimates | All arrays use an immutable training profile with co-access-aligned chunks/shards |
| Publication | Candidate remains incomplete/ineligible until decoded values, provenance, metadata, and storage validate | Entire artifact publishes atomically after pixels, labels, splits, source index, and manifest validate |
| Metadata | Exact direct/consolidated equivalence and a sealed metadata generation before normal selection | Same requirement at artifact root; permissive unconsolidated reads remain compatibility-only |
| Mutation | Completed raw/published runs are immutable; edits and additions create successors | No mutation; create a new artifact version |

### Lifecycle summary

| Profile | Producer today | Authority | Normal consumers | Intended lifecycle | Current implementation status |
| --- | --- | --- | --- | --- | --- |
| Raw probability snapshot | `infer_unet_subject_masks` | Decoded `mask_probs_roi` | Refined finalizer, validation, tuning/inspection, training-source preparation | Running ineligible candidate -> validated complete candidate -> atomically selected immutable snapshot | Standalone U-Net is closest to target; composites still use generic completion/selection and both forms lack the exact shared schema/storage manifest, CSR frame index, and mandatory consolidation gate |
| Refined editable draft | Legacy review tooling and parts of finalization | Dense `uint8 masks_roi` | Review/edit UI, validation, cache regeneration | Draft creating -> editable -> accepted/frozen; never selected for ordinary readers | Not implemented as a distinct future-canonical lifecycle; legacy review conflates mutable state and `latest` |
| Refined immutable publication | Future-canonical finalizer/coordinate publisher; legacy copy promotion also exists | Dense `uint8 masks_roi`; caches derived | Crimson, shape/measurement analysis, export, inspection | Build a new run from accepted draft -> replan -> regenerate caches -> validate -> complete ineligible -> activate immutable | Strict activation exists, but no accepted-draft-to-new-physical-publication boundary exists |
| Dense training artifact | `export_subject_mask_training_zarr` | Dense `uint8 masks_roi` plus validity and crop pixels | PyTorch training/evaluation | Resolve exact eligible sources -> stream materialization -> validate whole root -> publish immutable artifact | Dense/compact boundary is good; source selection, exact dtype/profile enforcement, storage planning, and atomic root publication remain incomplete |

### Profile 1: raw probability snapshot

#### Maintained producer states

The canonical standalone producer is
`src/fisheye/segmentation/infer_unet_subject_masks.py`. It creates
`subject_mask_runs/<run>` as `running` and selector-ineligible, records
`latest_pending`, writes and rereads the payload, validates coordinate evidence,
marks completion, and performs the selector transaction
(`src/fisheye/segmentation/infer_unet_subject_masks.py:740-870`,
`src/fisheye/segmentation/infer_unet_subject_masks.py:2491-2545`). The strict
coordinate publisher owns the final activation/rollback logic
(`src/fisheye/shared/subject_mask_coordinate_publication.py:2835-3008`).

Completed selected runs are intended to be immutable. The producer refuses to
overwrite complete/selected children, and a base run cannot be deleted while a
composite references it (`src/fisheye/segmentation/infer_unet_subject_masks.py:797-830`,
`src/fisheye/shared/composite_subject_mask.py:779-825`).

Two additional representations participate in the current lifecycle:

1. `subject_mask_shard_runs/<run>` are complete, selector-ineligible work-unit
   snapshots. The same U-Net producer writes them, suppresses canonical
   selector/registry publication, and the recording finalizer consumes them in
   sorted crop-row order
   (`src/fisheye/segmentation/infer_unet_subject_masks.py:927-956`,
   `src/fisheye/refinement/finalize_subject_masks.py:839-968`).
2. A depth-one composite `subject_mask_runs/<run>` references immutable bases
   and stores only exact replacement rows plus full target-row maps. Bases may
   not themselves be composite, and a composite may not also expose a direct
   top-level `mask_probs_roi`
   (`src/fisheye/shared/composite_subject_mask.py:362-410`).

The composite path does not yet share the standalone publisher's fail-closed
activation transaction. `compact_subject_mask_deltas` marks the still-running
child selector-eligible, validates it later, and then uses generic completion,
which mutates `latest` and `latest_complete`; eligibility is not the final
mutation and strict coordinate/consolidation proof is not part of the same
transaction (`src/fisheye/utils/compact_subject_mask_deltas.py:945-1006`,
`src/fisheye/utils/compact_subject_mask_deltas.py:1157-1162`,
`src/fisheye/utils/compact_subject_mask_deltas.py:1239-1246`,
`src/fisheye/shared/zarr_run_completion.py:194-242`). Future standalone and
composite snapshots must converge on one activation contract.

The composite payload contains:

| Array | Exact current dtype/shape | Role |
| --- | --- | --- |
| `source_codes` | `uint8[N]` | Select base or delta source |
| `source_run_indices` | `int32[N]` | Select referenced source run |
| `source_row_indices` | `int64[N]` | Physical row in selected source |
| `delta_target_row_indices` | `int64[D]` | Target rows replaced by delta |
| `delta_instance_key` | `uint64[D]` | Stable identity of each replacement |
| `mask_probs_roi_delta` | Same dtype and `[D,C,H,W]` trailing shape as base | Replacement probabilities |

The logical composite reader supports scalar, slice, and indexed selection,
coalesces sorted contiguous source rows, and caps an internal read batch at
64 MiB (`src/fisheye/shared/composite_subject_mask.py:618-776`).

#### Exact raw surfaces to freeze

| Surface | Current strict form | Contract disposition |
| --- | --- | --- |
| `mask_probs_roi` | `uint8` or `float16 [N,C,H,W]`; decoded finite `[0,1]`; exact encoding/semantics attributes | Required authority. Prefer one exact encoding per version; do not probe. |
| `masks_roi` | Optional `uint8 [N,C,H,W]` exact threshold result | Derived cache only; absent by default in canonical U-Net output. |
| `available_channels` | `bool[C]`; strict standalone publication currently requires every value true | Required component availability. Sparse/component-scoped outputs require a separate future contract or remain compatibility-only. |
| `prob_max` | `float32[N,C]` | Required derived metric, exactly recomputed. |
| `mask_present` | `bool[N,C]` | Required derived metric. |
| `area_px` | `float32[N,C]` | Required derived metric. |
| `centroid_xy` / `centroid_valid` | `float32[N,C,2]` / `bool[N,C]` | Required derived metric and validity. |
| `bbox_xyxy` / `bbox_valid` | `float32[N,C,4]` / `bool[N,C]` | Required ROI-pixel derived metric and validity. |
| Crop/identity columns | `source_crop_row_ids int64[N]`, `instance_key uint64[N]`, `source_acquisition_frame_index int64[N]`, and currently dtype-preserving `source_crop_xywh[N,4]` | Required exact selection from the bound crop snapshot. The new canonical contract requires crop-v2 and exact `float32[N,4]`; current dtype-preserving behavior remains compatibility-only. |
| `frame_row_offsets` | Absent today | Required future `int64[F+1]`, or an exact manifest-bound crop-owned equivalent. |

The strict publisher requires the declared physical dtype, encoding, multilabel
output semantics, independent-sigmoid overlap policy, and probability semantics
to agree exactly, and it recomputes metrics in bounded row blocks. It rejects
zero-row canonical runs rather than publishing an identity-free mask surface
(`src/fisheye/shared/subject_mask_coordinate_publication.py:1725-1827`,
`src/fisheye/shared/subject_mask_coordinate_publication.py:1872-2025`).

#### Raw consumers and access

| Consumer | Current access |
| --- | --- |
| Recording finalizer | Contiguous row windows for one component (`src/fisheye/refinement/finalize_subject_masks.py:1333-1369`) |
| Merge utility | Row window plus one component (`src/fisheye/utils/merge_subject_mask_runs.py:209-221`) |
| Tuning preview | One observation/component, but currently requires top-level binary `masks_roi` and cannot consume canonical probability-only/composite output (`src/fisheye/tune/subject_mask_tuner.py:403-450`, `src/fisheye/tune/subject_mask_tuner.py:850-900`) |
| Strict publication validation | Complete logical scan in bounded row blocks |
| Whole-recording validator | Indexed sample rows, including composite resolution, but currently hard-requires physical `uint8` even though strict publication permits `float16` (`src/fisheye/cluster/whole_recording_analysis_validate.py:68-97`) |

Future consumers should share one logical probability adapter across standalone,
shard, and composite representations. Binary consumers must request an explicit
threshold/derived surface instead of assuming `masks_roi` is the raw authority.

SAM, traditional body/swim segmentation, merged legacy runs, and eye-mask
backfills remain compatibility producers. In particular, historical swim
partial refresh mutates an existing raw run in place
(`src/fisheye/segmentation/swim_bladder_segmentation.py:693-770`). New canonical
publication must express component additions or replacements as a new immutable
snapshot or depth-one delta, never post-completion mutation.

### Profile 2: refined editable dense draft

#### Current gap

Palette does not yet have a real future-canonical editable-draft state. The
future finalizer creates a publication-owned, selector-ineligible child in
`running` state and completes/activates it within the same producer flow
(`src/fisheye/refinement/finalize_subject_masks.py:2450-2464`,
`src/fisheye/refinement/finalize_subject_masks.py:6059-6116`). The mutation guard
then rejects any canonical/publication-owned run
(`src/fisheye/shared/refined_subject_mask_mutation.py:26-47`).

Conversely, legacy review-created runs are mutable while being marked complete
and `latest`, and edits rewrite `latest`
(`src/fisheye/tune/refined_subject_mask_review.py:1903-1918`,
`src/fisheye/tune/refined_subject_mask_review.py:2782-2799`). This conflates a
working draft with a published authority.

The future lifecycle should instead be:

```text
ABSENT
  -> DRAFT_CREATING
  -> DRAFT_EDITABLE
  -> DRAFT_ACCEPTED
  -> PUBLICATION_BUILDING
  -> PUBLICATION_VALIDATED
  -> COMPLETE_INELIGIBLE
  -> ACTIVE_IMMUTABLE
```

Build, validation, or activation failure produces a retained
`FAILED_INELIGIBLE` candidate. Drafts never update normal read selectors.

#### Draft authority and edit transaction

An exact editable profile should require:

- physically present binary `uint8 masks_roi[N,C,H,W]` as the only pixel
  authority and edit target;
- ordered unique component labels plus `available_channels bool[C]`;
- exact raw/crop identity columns and a standard frame lookup;
- one homogeneous `H,W` per run, or an explicit padding/valid-window contract;
- no publication-owner/canonical-immutable marker while editable; and
- selector-ineligible draft states only.

Its exact future-facing surfaces should be separated by authority:

| Surface family | Exact draft role |
| --- | --- |
| Dense authority | Binary `masks_roi uint8[N,C,H,W]`; the only pixel edit target |
| Observation/crop identity | `source_crop_row_ids int64[N]`, `instance_key uint64[N]`, `source_acquisition_frame_index int64[N]`, and placement exact-copied from the bound raw/crop contract |
| Component availability | `available_channels bool[C]` with an ordered, unique component registry |
| Edit/audit state | `edit_applied bool[N,C]`; per-component `row_revision int64[N]`, `manual_override bool[N]`, timestamp bytes `[N,40]`, and reason bytes `[N,128]` |
| Metrics and caches | Derived from dense authority; may be stale during editing and are never edit authority |

Current per-component revision/audit arrays are created in
`src/fisheye/shared/refined_subject_component_contours.py:446-481`. The strict
scientific manifest intentionally excludes mutable review/revision/reason state
(`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:3766-3783`).

Current review writeback already performs the essential pixel transaction: it
resolves an explicit row/component, checks expected revision/shape, writes only
`masks_roi[row,component]`, increments revision state, and marks derivatives
stale (`src/fisheye/tune/refined_subject_mask_review.py:3185-3383`). Historical
compact-only input must first be materialized to dense
(`src/fisheye/shared/mask_store.py:230-292`).

Dense materialization currently has a dangerous freshness side effect:
`materialize_dense_masks_roi_from_store` calls `update_mask_storage_attrs` with
the default `reset_stale_flags=True`, which can clear derived mask, metric, and
contour stale flags without regenerating or validating those surfaces
(`src/fisheye/shared/mask_store.py:192-244`,
`src/fisheye/shared/mask_store.py:277-281`). The future contract must make dense
materialization establish authority while preserving derivative staleness.
Only a successful per-surface regeneration and validation receipt may clear a
freshness flag.

The future draft should store merge-safe row/component dirty state or an
append-only stale log. Current run-level stale attributes are vulnerable to
last-writer-wins behavior if edit locking later becomes finer-grained
(`src/fisheye/shared/mask_store.py:356-414`).

Draft dense chunks should be regular and unsharded for random mutation. At
512x512, one component row is 256 KiB; candidate row depths of one and four are
defensible benchmark points. Outer sharding is inappropriate unless one
exclusive writer owns the complete shard for its lifetime.

### Profile 3: refined immutable publication

Publication must create a new run ID from an accepted draft. It must not turn
the draft tree itself immutable and must not use byte-copy promotion as the
physical implementation.

The publisher should:

1. bind the source draft ID, accepted revision, dense logical digest, raw run,
   crop contract, and exact component order;
2. logically copy dense masks and identity arrays while independently deriving
   immutable storage plans;
3. verify exact decoded equality of authority and lineage;
4. regenerate metrics, compact caches, and contours from the destination dense
   authority;
5. validate cache freshness/digests and a closed-world surface inventory;
6. seal direct/consolidated metadata equivalence and the run manifest;
7. mark the child complete while selector-ineligible; and
8. activate it through the guarded parent selector transaction.

The existing strict activation path already implements a generation/lease,
pending-selector, final-eligibility, and rollback sequence
(`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:4462-4636`).
The canonical loader rejects ineligible/incomplete selector targets and keeps
historical fallback separate
(`src/fisheye/shared/refined_subject_masks_io.py:320-400`,
`src/fisheye/shared/refined_subject_masks_io.py:586-689`).

The older promotion path instead copies a complete tree, requires identical
chunks, and validates the compatibility schema. It then mutates selectors and
the registry after only a completion check, while its import receipt is written
after those mutations; a receipt-write failure can therefore leave selection
changed without rollback
(`src/fisheye/utils/promote_refined_subject_mask_run.py:184-230`,
`src/fisheye/utils/promote_refined_subject_mask_run.py:304-355`,
`src/fisheye/utils/promote_refined_subject_mask_run.py:452-496`). It cannot be
the future editable-to-published compactor. The strict coordinate path already
provides the required ordering: seal the receipt first, then perform guarded
selector mutation (`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:4533-4548`,
`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:4588-4611`).

#### Published derived surfaces

| Surface | Current logical representation | Publication requirement |
| --- | --- | --- |
| Dense authority | `masks_roi uint8[N,C,H,W]` | Physically present, binary, exact logical copy from accepted draft; access-aware immutable storage |
| Run metrics | `mask_present`, `area_px`, `centroid_xy/valid`, `bbox_xyxy/valid` | Recompute exactly from destination dense authority |
| Bitpacked cache | `masks_packed uint8[N,C,H,ceil(W/8)]`, little bit order | Non-authoritative; source dense digest and fresh status; per-frame/component-aligned chunks |
| RLE cache | Per component `counts uint32[M]`, `indptr int64[N+1]`, `present bool[N]`, `area_px int32[N]`, `bbox_xyxy int32[N,4]` | Non-authoritative; index/value arrays planned independently; fresh source digest |
| Sampled contour | `points_xy float32[N,K,2]`, `valid bool[N]`, `source_point_count int32[N]` | Bounded display/analysis cache; exact component and dense source binding |
| Full contour | `ptr int64[N]`, `len int32[N]`, `points_xy float32[P,2]` | Rebuild canonical packed flat values; no append or orphaned points in publication |
| Review state | revisions, manual flags, reasons, timestamps | Draft/audit provenance only; excluded from canonical scientific table unless explicitly modeled |

Current row-local full-contour refresh appends new flat points and can leave
orphaned values (`src/fisheye/shared/refined_subject_component_contours.py:646-680`).
Publication must repack it. Because full contours dominate the observed
Sleepyfish object count, their flat values should either be heavily sharded or
moved to a separately opened derived run. Sampled contours are the better hot
playback surface.

Every derived cache receipt should freeze its schema/profile, source dense
logical digest, accepted draft revision, component/row coverage, generator
version, validation mode/result, and `stale=false` state. The existing strict
inventory already verifies that caches are non-authoritative and binds their
contents to the dense payload
(`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:3179-3205`,
`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:3503-3579`).

### Profile 4: immutable dense training artifact

#### Exact artifact boundary

The future training artifact should be a self-contained immutable root with:

| Surface | Exact future role |
| --- | --- |
| `crop_runs/<run>/roi_images` | One fixed pixel profile, initially `uint8[N,H,W]` luma |
| `subject_mask_runs/<run>/masks_roi` | Binary `uint8[N,C,H,W]` training labels |
| `target_valid_channels` | `bool[N,C]` supervision availability |
| Splits | Exact disjoint train/validation/test row indexes |
| Source index | Dataset ID plus source crop row, `instance_key`, acquisition frame, crop placement, source manifest/content digests |
| Root manifest | Exact pixel/label/split/source-index digests, schema/storage plans, producer/model provenance, and publication receipt |

Compact masks are source encodings only. The exporter opens source data through
`MaskStore`, which can decode dense, bitpacked, or RLE, and writes dense blocks
(`src/fisheye/utils/export_subject_mask_training_zarr.py:335-360`,
`src/fisheye/utils/export_subject_mask_training_zarr.py:949-964`). Existing
tests already prove RLE-source-to-dense-output equality and reject compact
training output (`tests/unit/fisheye/test_export_subject_mask_training_zarr.py:339-426`).

Canonical training labels should normally come from an approved refined dense
publication. A probability-first raw run does not provide reviewed binary
labels, and `MaskStore` does not treat `mask_probs_roi` as a label authority
(`src/fisheye/shared/mask_store.py:2051-2057`). Historical compact-only refined
input belongs behind an explicit dense-materialization compatibility step.

#### Current training gaps

- The exporter stamps `editable_mask_surface="masks_roi"` on an immutable
  training artifact (`src/fisheye/utils/export_subject_mask_training_zarr.py:1026-1044`).
- Pixel export preserves arbitrary source dtype/shape and the reader
  permissively normalizes integer or floating input, despite the documented
  preferred `uint8` luma profile
  (`src/fisheye/utils/export_subject_mask_training_zarr.py:769-818`,
  `src/fisheye/training/zarr_subject_mask_dataset.py:109-130`).
- Validation casts masks and validity before checking them and does not enforce
  stored binary values (`src/fisheye/utils/export_subject_mask_training_zarr.py:1302-1311`).
- Source provenance records paths and run names but not exact source manifest
  and logical-content digests
  (`src/fisheye/utils/export_subject_mask_training_zarr.py:989-1005`,
  `src/fisheye/utils/export_subject_mask_training_zarr.py:1071-1131`).
- Source resolution can fall back to a lexicographically last child without an
  exact completion/eligibility gate
  (`src/fisheye/utils/export_subject_mask_training_zarr.py:131-182`).
- Crop and mask runs are completed and selected independently rather than by
  one root-level artifact transaction
  (`src/fisheye/utils/export_subject_mask_training_zarr.py:1132-1143`).
- Export, validation, and reading force unconsolidated access; no training
  direct/consolidated equivalence gate exists
  (`src/fisheye/utils/export_subject_mask_training_zarr.py:730-796`,
  `src/fisheye/training/zarr_subject_mask_dataset.py:166-168`).
- Validation and ROI export materialize whole arrays, making peak RSS scale
  poorly (`src/fisheye/utils/export_subject_mask_training_zarr.py:935-947`,
  `src/fisheye/utils/export_subject_mask_training_zarr.py:1302-1321`).

The training reader co-loads an ROI first-axis chunk and the matching mask and
validity row ranges, while the sampler groups examples by ROI chunk
(`src/fisheye/training/zarr_subject_mask_dataset.py:297-397`). Therefore the
training `StoragePlan` should align the first-axis boundaries of all three hot
arrays and use `(1,C,H,W)` as the label access unit. That differs deliberately
from a Crimson display cache whose useful unit may be `(1,1,H,W)`.

The existing `training_immutable_v1` profile targets 1 MiB inner chunks,
32/128 MiB shards, a 4,096-object budget, and `zstd_fast_v1`
(`src/fisheye/shared/zarr/storage_profiles.py:220-231`). It should become the
starting policy, not an assumed winner; actual shuffled minibatch and worker
RSS benchmarks must select the final layout.

## Scientific Authority And Lifecycle

### Raw subject masks

The maintained raw contract is probability-first. `mask_probs_roi` is the
model-valued authority, while a thresholded dense label array is not the raw
scientific authority (`docs/subject_mask_runs_contract.md:84-86`). The current
U-Net writer supports `uint8` linear probabilities by default and a `float16`
alternative (`src/fisheye/segmentation/infer_unet_subject_masks.py:1776-1804`).

The future-facing coordinate publisher requires exact row-aligned identity and
geometry arrays:

- `source_crop_row_ids : int64[N]`;
- `instance_key : uint64[N]`;
- `source_acquisition_frame_index : int64[N]`; and
- `source_crop_xywh : float32[N,4]` for the new canonical contract. Current
  strict raw publication preserves the selected crop dtype, so non-float32
  sources remain a compatibility boundary rather than weakening the new schema.

It rejects ambiguous legacy row aliases such as `detection_source`
(`src/fisheye/shared/subject_mask_coordinate_publication.py:895-980`).
The exact float32 decision follows the existing crop-v2 array contract
(`src/fisheye/shared/zarr/array_contracts.py:833-846`); canonical writers should
reject a non-v2/non-float32 crop source instead of silently casting it.

### Refined subject masks

The maintained refined direction is unambiguous: dense `uint8 masks_roi` must
be physically present. It is the editable authority in a draft and the
immutable raster/scientific authority in a publication. Historical compact-only
runs may be read for compatibility, but they must be materialized to dense
masks before editing or training. The strict publication layer enforces the
dense authority and requires stale flags to be cleared before publication
(`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:2178-2233`).

The same layer exact-copies crop/observation identity from the bound raw run and
forbids legacy `frame_indices`, `frame_counts`, `detection_indices`,
`source_frame_indices`, `source_refined_row_ids`, and
`source_detect_row_index` from future-normal refined runs
(`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:724-763`).

Bitpacked masks, RLE, metrics, contours, and other summaries are derived
caches. Their cache inventory must declare them non-authoritative
(`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:3179-3205`).
Accepted dense edits therefore invalidate those derivatives until an explicit
validation, publication, or maintenance step regenerates them.

### Training artifacts

Subject-mask training artifacts are intentionally dense `uint8`. The exporter
rejects bitpacked or RLE training authority
(`src/fisheye/utils/export_subject_mask_training_zarr.py:1241-1268`). Crop
pixels may be materialized in a training Zarr even though the analysis archive
should retain geometry and source-pixel bindings rather than duplicate crop
images.

## Existing Contract Surfaces

The shared array catalog contains an exact dense mask contract:

```text
schema: palette.array.subject_masks_roi_dense v1
dtype: uint8
shape: [n_rois, n_channels, H, W]
axes: [roi, component, y, x]
units: binary_label
coordinate space: roi_pixel
```

It also contains an exact `float32[N,2]` ROI-pixel contour-point contract
(`src/fisheye/shared/zarr/array_contracts.py:1314-1335`). This is a useful
foundation, but it is not yet an exact raw/refined run schema covering every
required array, relationship, manifest field, or storage intent.

The general stage schemas remain compatibility-oriented:

- raw masks still admit optional `frame_indices`, `frame_counts`,
  `detection_indices`, and optional `instance_key`, while requiring
  `detection_source` (`src/fisheye/shared/zarr/stage_arrays.py:1467-1490`);
- refined masks expose the same legacy family
  (`src/fisheye/shared/zarr/stage_arrays.py:1535-1560`);
- identity validation permits missing `instance_key` for legacy runs
  (`src/fisheye/shared/zarr/stage_arrays.py:217-306`); and
- generic refined validation permits a valid compact-only representation when
  dense `masks_roi` is absent (`src/fisheye/shared/zarr/stage_arrays.py:521-595`).

These behaviors are appropriate only behind an explicit compatibility
boundary. They disagree with the future-facing coordinate publication rules
and must not define the new canonical schema.

The current lineage catalog is likewise based on historical row arrays and
does not include a standardized `frame_row_offsets`
(`src/fisheye/shared/row_lineage.py:18-46`).

## Physical Storage Policy In Current Code

### Shared planner

Palette now has the right general abstraction. `plan_storage` derives chunk
and shard shapes from:

- dtype and per-row/access-unit shape;
- target uncompressed inner-chunk bytes;
- access class;
- mutability and whole-shard ownership; and
- object and shard byte budgets.

See `src/fisheye/shared/zarr/storage_planner.py:1-14` and
`src/fisheye/shared/zarr/storage_planner.py:148-225`. Published profiles target
approximately 1 MiB inner chunks and large immutable shards
(`src/fisheye/shared/zarr/storage_profiles.py:140-181`). `ArrayIntent`
explicitly models a mask component as a `(1,1,512,512)` access unit
(`src/fisheye/shared/zarr/storage_intent.py:38-61`).

Subject-mask writers do not yet use this machinery.

The first post-census implementation now defines exact raw/refined scientific
core schemas and generates immutable `StoragePlan` manifests without importing
Zarr or changing writers (`src/fisheye/shared/zarr/subject_mask_schema.py`,
`src/fisheye/shared/zarr/subject_mask_storage.py`). At full Sleepyfish scale,
the current `published_http_v1` profile derives:

| Core payload | Inner chunk | Outer shard | Estimated payload objects |
| --- | --- | --- | ---: |
| Raw `uint8 mask_probs_roi` | `[4,1,512,512]` = 1 MiB | `[1144,1,512,512]` = 286 MiB | 4,088 |
| Raw `float16 mask_probs_roi` | `[2,1,512,512]` = 1 MiB | `[1024,1,512,512]` = 512 MiB | 4,568 |
| Published refined `uint8 masks_roi` | `[4,1,512,512]` = 1 MiB | `[1144,1,512,512]` = 286 MiB | 4,088 |

The float16 surface transparently exceeds the current per-array 4,096-object
budget; the planner reports that miss rather than inventing a layout beyond the
512 MiB shard ceiling. This is a benchmark/profile decision, not a schema
failure. The uint8 plan satisfies the per-array budget, while the complete raw
core is estimated at 4,104 payload objects, so manifests now state explicitly
that the existing budget is per-array and report the stage total separately.

The same byte policy adapts to recording size. A 1,188,001-entry offsets array
uses 131,072-element (1 MiB) inner chunks inside one indexed shard. A
200,001-entry offsets array is only about 1.53 MiB and stays one eager,
unsharded chunk. Mask chunks remain 1 MiB because their bytes per observation,
not the recording's frame count, determine their row depth
(`tests/unit/fisheye/test_subject_mask_storage.py`). These plans are candidates,
not promoted writer defaults.

### Raw probability writer

The current raw writer creates arrays directly and uses fixed row counts. Its
normal default is:

```text
inner chunks: [32, 1, H, W]
outer shards: [2048, 1, H, W]
```

See `src/fisheye/segmentation/infer_unet_subject_masks.py:1360-1429` and
`src/fisheye/segmentation/infer_unet_subject_masks.py:1776-1804`. Compression settings
are inherited from the source ROI store rather than selected by an exact mask
storage profile (`src/fisheye/segmentation/infer_unet_subject_masks.py:390-411`).

At 512x512 these fixed row counts produce very different byte sizes:

| Probability dtype | Inner chunk | Outer shard |
| --- | ---: | ---: |
| `uint8` | 8 MiB | 512 MiB |
| `float16` | 16 MiB | 1 GiB |

This is the failure mode the byte planner was introduced to prevent. A row
count is not a storage budget.

The current writer does write the four future-facing identity/geometry arrays,
but it gives them fixed 4,096-row chunks rather than a shared byte-derived plan
(`src/fisheye/segmentation/infer_unet_subject_masks.py:873-909`).

Composite publication is also outside the shared planner. Delta probabilities
inherit the primary base's chunk/shard layout and truncate it only for small
delta row counts. Mapping and lineage arrays use fixed 131,072-row shards,
while metrics use fixed 16,384-row inner chunks with the same shard-row target
(`src/fisheye/utils/compact_subject_mask_deltas.py:689-708`,
`src/fisheye/utils/compact_subject_mask_deltas.py:1005-1126`). The future policy
must plan composite delta, mapping, lineage, and metric surfaces by bytes rather
than preserve these inherited row-count contracts.

### Refined dense and compact writers

The centralized mask-specific constants currently select:

```text
dense masks_roi:       [128, 1, H, W]
bitpacked mask cache:  [512, 4, H, ceil(W/8)]
metrics:               256 rows
```

See `src/fisheye/shared/subject_mask_chunks.py:48-90`. At 512x512, this means:

| Payload | Logical inner-chunk bytes |
| --- | ---: |
| Dense `uint8 [128,1,512,512]` | 32 MiB |
| Bitpacked `[512,4,512,64]` | 64 MiB |

Both are much larger than the shared approximately 1 MiB starting budget. The
bitpacked layout also combines 512 observations and all four components inside
one independently decoded chunk, which is poorly aligned with a per-frame or
per-component display/edit request.

The refined finalizer calls Zarr creation directly for dense masks, compact
caches, metrics, and contours (`src/fisheye/refinement/finalize_subject_masks.py:2424-2558`).
Flat contour `points_xy` uses unsharded `float32[4096,2]` chunks, only 32 KiB
logical each (`src/fisheye/refinement/finalize_subject_masks.py:3413-3418`).

`MaskStore` also creates bitpacked and RLE arrays directly. RLE count chunks
target about 4 MiB, while row indexes use fixed 256-row chunks and remain
unsharded (`src/fisheye/shared/mask_store.py:596-663`,
`src/fisheye/shared/mask_store.py:890-951`).

The current process-shard finalizer does demonstrate the correct minimum
ownership rule for dense writes: it rounds process row ranges to the physical
dense chunk grid, assigns contiguous whole-chunk regions per process, and lets
the driver own shared full-array metric writes
(`src/fisheye/refinement/finalize_subject_masks.py:2024-2087`,
`src/fisheye/refinement/finalize_subject_masks.py:4479-4499`,
`src/fisheye/refinement/finalize_subject_masks.py:4641-4742`). This is not yet a
general multi-array plan. Every parallel writer must own complete physical
chunks or shards for every array it writes; variable-length RLE and contour
values must be serialized or merged deterministically by the driver.

### Promotion and publication

The existing refined promotion utility copies the selected run tree and then
requires the source and destination to have identical chunks
(`src/fisheye/utils/promote_refined_subject_mask_run.py:184-216`). It therefore
does not implement the desired transition:

```text
editable dense draft
  -> validation and cache regeneration
  -> immutable, physically replanned published snapshot
```

Completion and activation machinery exists, but an accepted editable draft to
new physical publication lifecycle does not. The legacy promotion path simply
preserves physical layout across that boundary.

### Training writer

The training exporter creates regular, unsharded arrays. Dense ROI images use
256-row chunks, dense masks use the refined fixed-row helper, and vector arrays
use fixed 8,192-row chunks
(`src/fisheye/utils/export_subject_mask_training_zarr.py:780-919`). A shared
`training_immutable_v1` storage profile exists, but the subject-mask exporter
does not use it (`src/fisheye/shared/zarr/storage_profiles.py:220-231`).

The training dataset is chunk-aware: it reads an ROI chunk and the matching
mask/valid range and groups samples by ROI chunk
(`src/fisheye/training/zarr_subject_mask_dataset.py:298-412`). Its final storage
profile should therefore be benchmarked against actual shuffled batch access,
not borrowed from Crimson playback.

## Existing Test And Benchmark Coverage

Useful evidence already exists and should be adapted to exact `StoragePlan`
profiles rather than discarded:

- raw probability benchmarks compare regular and indexed-sharded values,
  decoded digests, write time, peak RSS, object/byte inventories, sequential
  component scans, and random rows
  (`src/fisheye/diagnostics/benchmark_subject_mask_probability_sharding.py:219-392`);
- a separate harness repeats cold/warm probability reads, but currently uses
  local unconsolidated opens
  (`src/fisheye/diagnostics/benchmark_subject_mask_probability_sharding_reads.py:104-256`);
- refined benchmarks compare dense/bitpacked/RLE encoding size and
  encode/decode behavior
  (`src/fisheye/diagnostics/benchmark_mask_storage_encodings.py:177-388`);
- the full-finalizer harness exercises production-shaped dense masks,
  contours, and eye geometry
  (`src/fisheye/diagnostics/benchmark_subject_mask_full_finalizer.py:836-1043`);
- exporter tests cover dense authority, partial validity, compact-source to
  dense-output equality, compact-output rejection, splits, and codebooks
  (`tests/unit/fisheye/test_export_subject_mask_training_zarr.py:181-430`);
  and
- dataset tests cover split bounds/disjointness and returned
  image/mask/valid shapes
  (`tests/unit/fisheye/test_zarr_subject_mask_dataset.py:126-367`).

There is no maintained subject-mask storage benchmark for real shuffled
training minibatches, worker/prefetch scaling, aggregate RSS, or GPU starvation.
The shared benchmark contract already records this gap
(`docs/shared_zarr_storage_benchmark_contract.md:263-270`). There is likewise no
single gate spanning editable write amplification, draft-to-publication
replanning, cache regeneration, consolidated metadata, Crimson playback, and
training access. Those should remain separate workloads under one evidence
schema, not one blended score.

## Current Sleepyfish Layout

### Selected raw run

The archive selects:

```text
subject_masks_sleepyfish_composite_20260718_01
```

This is a depth-one composite over 22 base `subject_mask_shard_runs`, with no
active delta layer. The composite's top level is a logical resolver and does
not duplicate the full probability payload.

The 22 base probability stores contain approximately 1,169,010 rows in total.
Their `mask_probs_roi` arrays are Zarr v3, `uint8`, unsharded, and use regular
`[32,1,512,512]` chunks with a bytes/Zstd chain. The metadata-grid census found:

| Surface | Estimated outer payload objects |
| --- | ---: |
| All arrays in the 22 base runs | 143,432 |
| `mask_probs_roi` alone | 109,632 |

This agrees with the earlier measured archive audit in
`docs/archive/sleepyfish_subject_mask_storage_sharding_strategy_2026-07-10.md:84-146`.
The maintained writer has since gained indexed sharding defaults, but existing
published base stores retain their original unsharded physical layout.

### Selected refined run

The archive selects:

```text
refined_subject_masks_sleepyfish_full_collection_canary_20260711_02
```

This older complete run has 1,169,010 observations and four components. Its
dense authority is:

```text
shape:   [1169010, 4, 512, 512]
dtype:   uint8
chunks:  [256, 1, 512, 512]
shards:  none
codec:   bytes -> Zstd(level 0)
logical bytes per chunk: 64 MiB
```

It contains no bitpacked cache. Most historical row arrays are regular
1,024-row or 16,384-row chunks, although `instance_key` was later backfilled
with indexed sharding.

The complete metadata grid estimates 378,234 payload objects:

| Surface | Estimated payload objects |
| --- | ---: |
| Variable contour points | 151,218 |
| Other duplicated per-component state | 110,472 |
| Contour pointer/length indexes | 36,536 |
| Top-level lineage/state | 33,142 |
| Dense `masks_roi` | 18,268 |
| Sampled contours | 13,704 |
| Relations | 9,134 |
| Finalization metrics | 3,024 |
| Component metrics | 2,304 |
| Run metrics | 432 |

Dense masks are therefore important for read/write amplification, but they are
not the dominant source of object fanout. Full-resolution variable contours
and duplicated component state must be included in the storage redesign.

### Metadata

The archive root is Zarr v3 and has inline consolidated metadata for 6,534
nodes. That demonstrates that consolidated metadata can exist after archive
publication. The mask writers themselves do not freeze or validate direct
versus consolidated declaration equivalence, so consolidation is not yet an
enforceable per-run publication contract.

## Consumer Access Patterns

The storage plan must cover multiple distinct workloads:

| Consumer | Access pattern | Consequence |
| --- | --- | --- |
| Review/editor | Random single observation and component; writes dense authority | Minimize row/component write amplification; editable chunks generally unsharded |
| Crimson playback | Random frame after seeks and short forward windows | Independently decode only the requested frame/component or a small aligned window |
| Raw finalization | Sequential/block probability scans | Immutable large shards with byte-budgeted inner chunks |
| Subject-shape/derived analysis | Block row scans, currently commonly 1,024 rows | Sharded immutable source with efficient consecutive chunk ranges |
| Contour consumer | Pointer/length lookup followed by indexed flat-point slice | Eager/index columns plus separately sharded flat values |
| Training | Shuffled batches grouped by ROI-image chunk | Dense immutable chunks optimized for batch locality and bounded RSS |
| Full validation/export | Complete logical scans | Large shards reduce object and metadata operations |

The strict scientific reader already requires dense masks on the canonical
path (`src/fisheye/shared/refined_subject_masks_io.py:590-665`). The review path
reads and writes individual dense rows/components. Subject-shape materializes
blocks (`src/fisheye/analysis_workflows/materializers/subject_shape.py:157-278`).
These are different access classes and should not be represented by one
hard-coded row count.

## Documentation Discrepancies

Current code and repository policy supersede several older statements:

1. `docs/mask_rle_storage_design_and_benchmark_plan.md:19-30` says dense masks
   need not remain canonical and describes compact-only refined modes. Current
   finalization explicitly rejects compact-only modern publication
   (`src/fisheye/refinement/finalize_subject_masks.py:4873-4881`,
   `src/fisheye/refinement/finalize_subject_masks.py:6390-6399`). This document is
   stale for future-facing refined masks.
2. `docs/refined_subject_masks_runs_contract.md:178-191` proposes a custom
   `frame_index/{frame_numbers,row_start,row_count,row_indices}` lookup. The
   newer cross-stage direction is the simpler exact `F+1 int64
   frame_row_offsets` contract.
3. The same refined contract documents `[128,1,H,W]` dense chunks and
   `[512,4,H,ceil(W/8)]` bitpacked chunks
   (`docs/refined_subject_masks_runs_contract.md:306-347`). Those are current
   constants, not evidence-backed universal storage contracts.
4. `docs/subject_mask_runs_contract.md:140-152` permits broader dtypes and makes
   some lineage arrays merely recommended. The future schema must freeze exact
   dtypes and required fields per version.
5. `docs/refined_subject_masks_runs_contract.md:135-148`,
   `docs/refined_subject_masks_runs_contract.md:630-647`, and
   `docs/refined_subject_masks_runs_contract.md:813-821` treat
   `frame_indices`, `frame_counts`, `detection_indices`, and
   `detection_source` as canonical surfaces. Strict future publication forbids
   these aliases in favor of exact crop identity and acquisition-frame lineage
   (`src/fisheye/shared/refined_subject_mask_coordinate_publication.py:724-764`).
6. `docs/mask_rle_storage_design_and_benchmark_plan.md:76-80` says edit
   writeback refreshes touched bitpacked rows immediately. Current policy marks
   bitpacked and other derivatives stale after dense edits
   (`src/fisheye/shared/mask_store.py:384-390`).

## Proposed Contract Families

### 1. Raw probability snapshot

Properties to freeze:

- immutable and selector-controlled;
- exact `instance_key` and crop-snapshot/row binding;
- exact component order and probability encoding;
- standard frame lookup;
- byte-derived `WINDOWED`/block chunks;
- large immutable indexed shards;
- exact codec/checksum chain; and
- consolidated/direct metadata equivalence.

The canonical probability representation needs an explicit decision. `uint8`
is compact and matches the current primary writer. `float16` retains more
probability precision but doubles bytes and changes scientific semantics. If
both remain valid, they should have distinct versioned profile IDs rather than
dtype probing.

### 2. Refined editable draft

Properties to freeze:

- dense `uint8 masks_roi` is physically present and authoritative;
- random single-row/component mutation is supported;
- no outer sharding unless the writer owns complete shards;
- derived caches carry explicit source digest and stale state;
- accepted edits preserve `instance_key` and row lineage; and
- variable ROI sizes are described by the bound crop contract rather than a
  global 512x512 assumption.

At 512x512, one dense component is 256 KiB. Candidate editable chunks should
therefore include one row/component and a small approximately 1 MiB alternative
such as four rows by one component. The choice should be benchmarked using
actual edit, validation, and rollback behavior.

### 3. Refined immutable publication

Properties to freeze:

- exact logical equality with the accepted editable draft;
- physical storage replanned during publication;
- byte-derived inner chunks and large outer shards;
- optional bitpacked/RLE/display caches regenerated from the dense authority;
- cache manifests bind the dense logical-content digest;
- stale caches cannot be promoted;
- contours and measurements are either sharded derived surfaces or separate
  lazy derived runs; and
- the published run is immutable.

This profile must not be implemented as a byte-for-byte tree copy of the draft.

### 4. Training artifact

Properties to freeze:

- immutable dense `uint8` labels;
- exact source refined-run and dense-mask digest;
- crop pixels may be self-contained in the training artifact;
- byte-derived sharded storage;
- batch-aware read layout; and
- source compact encoding is provenance only, never training authority.

## Open Contract Decisions

- [x] Require `source_crop_xywh : float32[N,4]` in the new canonical raw,
  refined, and training source-index schemas, matching crop-v2. Preserve
  arbitrary source dtypes only behind a legacy conversion boundary.
- [ ] Freeze exact raw and refined schema IDs and exact array sets.
- [ ] Decide whether every mask run owns an exact
  `frame_row_offsets : int64[F+1]`, or binds to and validates the crop
  snapshot's offset index. Do not retain `frame_counts` as the lookup
  authority.
- [ ] For merged training artifacts, preserve source dataset plus acquisition
  frame and bind each source run's offset-index digest; do not synthesize one
  cross-recording frame axis.
- [ ] Freeze stable observation ordering and allow zero, one, or multiple
  observations per frame through repeated CSR offsets.
- [ ] Freeze `instance_key` as observation/edit-lineage identity, not subject or
  longitudinal track identity.
- [ ] Freeze component order, component IDs, empty-mask semantics, fill values,
  and reason/validity registries.
- [ ] Decide whether raw `uint8` and `float16` probabilities are separate
  schema/storage profiles.
- [ ] Freeze one homogeneous ROI extent per run or define exact padding and
  per-row valid-window arrays for mixed extents.
- [ ] Decide whether full variable contours remain in the core refined run or
  become a separately opened derived run/cache.
- [ ] Freeze dense, bitpacked, RLE, metric, sampled-contour, and full-contour
  freshness/digest relationships.
- [ ] Freeze direct/consolidated metadata normalization and equivalence.
- [ ] Define selector and compatibility boundaries for historical compact-only
  refined runs.
- [ ] Freeze the refined state machine so drafts never update normal selectors
  and publications always receive a new immutable run ID.
- [ ] Freeze one root-level atomic publication contract for crop pixels, dense
  labels, splits, and source indexes in training artifacts.

## Implementation Checklist

This checklist is intentionally not authorization to mutate production.

### Phase A: exact logical schemas

- [x] Add exact raw probability authority/core array contracts and separate
  `uint8`/`float16` logical schema variants, without writer adoption.
- [x] Add the exact refined dense scientific-core array contracts and logical
  validator, without yet freezing draft audit or optional cache layouts.
- [ ] Add exact editable-draft audit/revision arrays and published derived-cache
  extension schemas.
- [ ] Add exact component/reason registries and decoded-array validation.
- [ ] Add standard frame lookup and multi-observation tests.
- [ ] Bind both schemas to exact crop identity, manifest, coordinate contract,
  and source-pixel authority.
- [ ] Put all legacy aliases and compact-only inputs behind explicit adapters.

### Phase B: storage intents

- [x] Classify raw probability and refined dense scientific-core arrays as
  `EAGER`, `WINDOWED`, or `PER_ROW` and generate immutable byte-derived plans.
- [ ] Extend the classification to editable audit state, compact caches,
  variable contours/RLE, and training-artifact arrays.
- [ ] Classify each output as editable, immutable, or whole-shard-owned append.
- [ ] Route every mask-family array through `plan_storage`.
- [ ] Include composite delta, mapping, lineage, and metric arrays in that
  planning boundary; do not inherit base chunks or retain fixed
  131,072/16,384-row contracts.
- [ ] Remove mask-specific row-count constants from contract ownership.
- [ ] Give flat RLE/contour values and their pointer/index columns separate
  access intents.
- [ ] Estimate and gate inner bytes, shard bytes, and object counts per run.

### Phase C: publication lifecycle

- [ ] Add exact raw and refined run-manifest envelopes.
- [ ] Route standalone and composite raw snapshots through one fail-closed
  activation transaction with eligibility as the final mutation.
- [ ] Implement draft-to-published logical copy with physical replanning.
- [ ] Regenerate derived caches from dense authority during explicit
  publication or maintenance.
- [ ] Make dense compatibility materialization preserve all derivative stale
  flags until each surface has a successful regeneration/validation receipt.
- [ ] Verify decoded equality, cache digests, CRCs, and stale-state closure.
- [ ] Consolidate metadata and prove normalized direct/consolidated equality
  before selector eligibility.
- [ ] Preserve failed candidates as selector-ineligible evidence.

### Phase D: benchmark gates

- [ ] Require every candidate to pass exact decoded logical digests, exact
  schema/dtype/shape, source-manifest binding, direct/consolidated equivalence,
  and its declared object budget before comparing speed.
- [ ] Benchmark raw probability write, publish, full scan, and block scan.
- [ ] Measure raw inference/write overlap, shard-buffer RSS, component scans,
  and random row/component reads with whole-shard ownership asserted.
- [ ] Benchmark refined single-row/component edits and write amplification.
- [ ] Start the editable gate with at most 2 MiB decoded/write amplification
  per single-row/component edit; test rollback, stale-cache invalidation, and
  concurrent-edit serialization rather than treating that number as final.
- [ ] Benchmark Crimson random-frame, seek, playback-window, and resident-cache
  behavior for dense and derived display surfaces.
- [ ] Benchmark contour pointer lookup and flat-point slice access.
- [ ] Benchmark full validation and derived-cache regeneration time.
- [ ] Benchmark training shuffled-batch throughput and peak aggregate RSS at
  `num_workers={0,1,4,8}` with the actual batch size, prefetch, and persistent
  worker settings.
- [ ] Record samples/s, p50/p95 batch latency, request counts, cache reuse, GPU
  starvation, and whether per-worker caches grow to hundreds of MiB.
- [ ] Compare regular and access-aware stores using identical decoded values.
- [ ] Record local-scratch compute, publish-back time, object counts, apparent
  bytes, physical I/O, and peak RSS.
- [ ] Run immutable candidates on node-local NVMe and shared storage; require a
  mounted Mac/VPN Crimson gate for published playback surfaces.

### Phase E: canary and adoption

- [ ] Publish selector-ineligible raw and refined shadow snapshots.
- [ ] Validate them in Palette and Crimson with exact dtypes and no probing.
- [ ] Confirm historical compatibility adapters remain isolated.
- [ ] Run one full-duration selector-ineligible canary.
- [ ] Freeze a versioned promotion gate and rollback profile.
- [ ] Change production defaults only after correctness and workload gates pass.

## Recommended First Implementation Slice

After this census is reviewed, the smallest coherent implementation slice is:

1. define exact raw/refined logical schemas and frame-index semantics;
2. inventory every mask-family array into an explicit access and mutability
   class;
3. produce storage plans without yet changing writers;
4. validate those plans against synthetic multi-observation, empty-frame, and
   variable-ROI cases; and
5. add a selector-ineligible shadow writer only after the logical and storage
   manifests are frozen.

This keeps scientific identity, editing semantics, and physical optimization
separate while giving all of them one enforceable owner.
