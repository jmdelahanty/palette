# Canonical Detection Producer/Consumer Census

Status: accepted Phase 1 design input

Palette checkpoint: `eb94f885`

Crimson source reviewed: `acce592`

Date: 2026-07-23

## Decisions

| Question | Decision |
| --- | --- |
| Is `bbox_norm_coords` authoritative? | Yes. It is the only independently produced geometry surface. It is source-camera-normalized `cx,cy,w,h`. |
| What are `bbox_img_xyxy` and `centers_img_xy`? | Required materialized, integrity-checked projections in the canonical run. They are derived, but they are not optional caches: crop publication consumes them and the coordinate contract verifies exact equality to the normalized authority. |
| Are `frame_counts` and `n_detections` aliases? | Yes. Current writers write the same values and validators require equality. `n_detections` is the legacy name. |
| Should the future contract add `frame_row_offsets`? | Yes. Exact `int64`, shape `(n_frames + 1,)`, required and authoritative for frame-to-row slicing. |
| Is `frame_counts` still required? | No in the target logical contract. It is `diff(frame_row_offsets)` and is an optional transition cache while Palette consumers migrate. |
| Is `n_detections` still written? | Only by an explicit legacy-compatibility publication profile until Crimson's legacy reader is replaced. It is not part of the new canonical core. |
| What row order is guaranteed? | Rows are contiguous in nondecreasing `frame_indices` order. Within one frame, persisted producer order is stable but has no ranking semantics. Consumers identify rows by `instance_key`, not row position. |
| How is an empty frame represented? | By `frame_row_offsets[f] == frame_row_offsets[f + 1]`. There is no sentinel observation row. |
| How is an empty run represented? | Every row array has zero rows; `frame_row_offsets` has `n_frames + 1` zero values; any transition `frame_counts`/`n_detections` array has `n_frames` zero values. |
| Are null/sentinel rows allowed in raw canonical detections? | No. Raw canonical row arrays are dense and fully materialized. Physical Zarr fill values have no missing-value meaning. |

The new physical name is deliberately `frame_row_offsets`. Existing refined
detection stores use `frame_offsets`; that path is compatibility evidence, not a
reason to give a less explicit name to the shared contract. The Phase 2 adapter
must bind existing `frame_offsets` to the same logical offset contract without
writing both names into one run.

## Current Producer Ground Truth

`detect_yolo` is the current canonical `detect_runs/<run>` producer. It writes
four model/result row arrays plus the stable identity and—only as an all-or-none
canonical set—the acquisition-frame index plus pixel bbox and center arrays
(`src/fisheye/detection/detect_yolo.py:1384-1435`). Both count arrays receive the
same `frame_counts` value (`src/fisheye/detection/detect_yolo.py:1433-1436`).

The model output is accumulated one frame at a time. A frame contributes one
contiguous tuple of box, score, and class arrays, and final arrays are simple
concatenations of those tuples (`src/fisheye/detection/detect_yolo.py:2746-2784`,
`3174-3186`). All three decoder paths pass monotonically increasing batch frame
indices to that accumulator (`src/fisheye/detection/detect_yolo.py:2851-2884`,
`3035-3050`, `3112-3128`). This produces frame-major order today, but the current
immutable-storage validator checks cardinality and count equality without
checking sort order (`src/fisheye/shared/immutable_yolo_storage.py:218-225`,
`290-310`). The new schema must make the order an enforced invariant.

Canonical coordinate publication is stricter than the older `StageSpec`:

- canonical cardinality requires exact current dtypes and all ten arrays
  (`src/fisheye/shared/observation_coordinate_publication.py:2738-2776`);
- `frame_indices` and `source_acquisition_frame_index` must be the exact full-video
  identity mapping and stay within the acquisition frame domain
  (`src/fisheye/shared/observation_coordinate_publication.py:2778-2789`);
- both count arrays must be exact `int32`, must equal `bincount(frame_indices)`,
  and must sum to the row count
  (`src/fisheye/shared/observation_coordinate_publication.py:2792-2820`);
- normalized boxes must be finite, positive-area, and contained in `[0,1]`, and
  source-camera boxes and centers are derived without changing dtype
  (`src/fisheye/shared/observation_coordinate_publication.py:809-873`);
- publication re-derives and exactly compares both materialized projections
  (`src/fisheye/shared/observation_coordinate_publication.py:1100-1158`).

The currently declared `DETECT_SPEC` disagrees with the writer: it says
`bbox_norm_coords` is `float32`, marks `instance_key` and `n_detections`
optional, and does not declare the canonical acquisition/pixel arrays
(`src/fisheye/shared/zarr/stage_arrays.py:1019-1050`). The current writer and
canonical publication validator are ground truth for old archives; the Phase 2
schema replaces this stale declaration rather than expanding its unions.

Traditional detection and acquisition CSV import are not additional canonical
producers. They publish quarantined, selector-ineligible artifacts: the import
module explicitly refuses normal `detect_runs` publication
(`src/fisheye/utils/import_acquisition_detections_to_detect_run.py:1-6`,
`1043-1062`), and traditional output is stamped as artifact semantics
(`src/fisheye/detection/detect_traditional.py:862-875`).

Two maintenance paths can physically write below `detect_runs` but are not
semantic producers:

- `audit_yolo_detection_sharding` creates a benchmark-only replay. Its current
  seven-array list omits the three canonical acquisition/pixel arrays, so it
  must not be reused for vNext benchmarking unchanged
  (`src/fisheye/diagnostics/audit_yolo_detection_sharding.py:16-24`, `59-101`).
- `migrate_immutable_yolo_sharding` stages physical copies while preserving the
  existing array's dtype, chunks, codecs, fill, and attributes; it does not
  define array meaning (`src/fisheye/utils/migrate_immutable_yolo_sharding.py:
  1-10`, `285-315`).

## Target Canonical Array Roles

`N` is the detection-row count and `F` is the complete acquisition-frame count.
Continuous geometry changes from the current `float64` representation to the
already accepted `float32` vNext representation.

| Path | Target dtype and shape | Role | Required | Intended read shape | Producer | Principal consumers |
| --- | --- | --- | --- | --- | --- | --- |
| `frame_indices` | `int32 (N,)` | Row-to-run-frame membership; must agree with offsets | yes | row windows / joins | canonical detector | quality, refine, crop, training, Crimson adapter |
| `source_acquisition_frame_index` | `int64 (N,)` | Sealed acquisition temporal authority | yes | row windows / joins | canonical detector | coordinate validation, crop lineage |
| `bbox_norm_coords` | `float32 (N,4)` | Independent geometry authority, source-camera normalized `cx,cy,w,h` | yes | per-frame row range / row windows | canonical detector | refine, crop, training, Crimson |
| `bbox_img_xyxy` | `float32 (N,4)` | Required exact source-camera half-open pixel-edge projection | yes | per-frame row range / row windows | canonical detector projection | coordinate validation, crop publication |
| `centers_img_xy` | `float32 (N,2)` | Required exact midpoint projection of `bbox_img_xyxy` | yes | per-frame row range / row windows | canonical detector projection | crop placement and validation |
| `scores` | `float32 (N,)` | Model confidence evidence | yes | row-aligned with boxes | canonical detector | refine, training, Crimson |
| `class_ids` | `int32 (N,)` | Model taxonomy index | yes | row-aligned with boxes | canonical detector | identity, refine, training, Crimson |
| `instance_key` | `uint64 (N,)` | Stable row identity derived from recording, frame, bbox, and class | yes | row windows / keyed joins | canonical detector identity minting | all modern downstream lineage |
| `frame_row_offsets` | `int64 (F+1,)` | Authoritative CSR-style frame-to-row index | yes | eager once or two-value frame lookup | canonical detector index builder | Crimson and any per-frame reader |
| `frame_counts` | `int32 (F,)` | `diff(frame_row_offsets)` transition cache | no | eager / frame windows | compatibility profile | current Palette compatibility |
| `n_detections` | `int32 (F,)` | Exact legacy alias of `frame_counts` | legacy profile only | eager | legacy compatibility profile | current Crimson compatibility |

`bbox_img_xyxy` and `centers_img_xy` are intentionally redundant. The coordinate
module describes all three geometry surfaces as deliberately persisted and
digest-bound (`src/fisheye/shared/observation_coordinate_publication.py:9-19`).
Ordinary and incremental crop publication compare normalized source rows, use
the persisted centers to determine crop placement, and use persisted pixel
boxes to derive ROI-space boxes (`src/fisheye/tracking/crop.py:508-596`,
`src/fisheye/tracking/incremental_crop.py:768-846`). Removing either projection
would change the current fail-closed coordinate-publication design.

## Palette Consumer Census

| Consumer class | Current access | Arrays and assumptions | Evidence |
| --- | --- | --- | --- |
| Canonical completion and coordinate validation | eager whole-array validation | all ten current arrays; exact dtype, row alignment, projection equality, and count equality | `src/fisheye/shared/observation_coordinate_publication.py:2738-2820`, `2965-3027` |
| Detection quality | eager whole-array | `frame_indices`, `bbox_norm_coords`; prefers `frame_counts`, derives it with `bincount` when absent | `src/fisheye/refinement/detect_quality.py:608-620`, `842-869` |
| Detection refinement | eager whole-array | boxes and frames required; count, score, class, and key compatibility fallbacks remain | `src/fisheye/refinement/refine_detect.py:1299-1323`, `1392-1417` |
| Collection quality snapshot | row-window/shard reads | slices aligned ranges of frame, bbox, and key; explicitly rejects descending frame order inside a range | `src/fisheye/refinement/detect_quality_collection.py:351-364`, `718-730` |
| Crop and incremental crop | eager whole-array followed by per-frame video processing | modern source requires key, frame, and normalized bbox; canonical path verifies and consumes both pixel projections; crop requires sorted frames when requested | `src/fisheye/tracking/incremental_crop.py:491-530`, `src/fisheye/tracking/crop.py:508-596`, `1296-1309` |
| Arena assignment | eager whole-array | frames and boxes; prefers stored `frame_counts`, otherwise derives counts | `src/fisheye/tracking/arena_assignment.py:782-820` |
| Registry and metadata | eager small/full count vector | prefers `frame_counts`, falls back to `n_detections` | `src/fisheye/registry/extractors/detect_performance.py:272-280`, `src/fisheye/shared/metadata.py:43-50` |
| Training export | eager whole-array copy | can resolve raw `detect_runs` bbox/frame paths, casts bbox to `float32`, and loads complete bbox and frame arrays | `src/fisheye/utils/export_detect_training_zarr.py:839-852`, `1323-1329`, `1680-1698` |
| Analysis-to-training promotion | no direct raw-detect read | promotion selects editable refined rows and their manual flags; raw detections enter only through previously established refined/crop lineage | `src/fisheye/utils/promote_analysis_detect_to_training.py:182-195`, `225-245` |
| Detection web editing | not a raw-detect writer; editable refined dense surface | uses NaN bbox/score and class `-1` for a manually cleared refined row | `src/fisheye/tune/detect_review_backend.py:204-220`, `324-364` |
| Diagnostics | eager metadata/full-array inspection | the old checker inherits stale `DETECT_SPEC`; comparison reads `n_detections` only; the sharding audit hashes full arrays in row windows | `src/fisheye/diagnostics/check_detection_runs.py:19-31`, `57-75`; `src/fisheye/diagnostics/compare_detection_runs.py:84-112`; `src/fisheye/diagnostics/audit_yolo_detection_sharding.py:27-34`, `138-175` |

The last row matters for sentinel scope: NaN and `-1` are valid state markers on
the editable refined surface, not on immutable raw canonical detection rows.

## Crimson Consumer Census

The current Palette-runs loader eagerly reads complete `frame_indices` and
`bbox_norm_coords`, and it reads only the physical name `n_detections`, not
`frame_counts` (`crimson/src/zarr_loader.cpp:1328-1368`). Scores and class IDs
are optional in this compatibility path (`crimson/src/zarr_loader.cpp:1375-1391`).

Crimson then scans `frame_indices`, constructs in-memory offsets, and stably
copies every row into frame-contiguous buffers (`crimson/src/zarr_loader.cpp:
1465-1578`). If frame indices are absent, it instead builds offsets from
`n_detections` (`crimson/src/zarr_loader.cpp:1579-1608`). Per-frame display uses
those in-memory offsets to select `[start,end)` and then walks only that frame's
rows (`crimson/src/zarr_loader_detections.cpp:1203-1249`, `1282-1306`). A
persisted `frame_row_offsets` therefore removes an O(N) initialization scan and
copy and gives a future remote reader the exact two-value lookup it needs.

The current loader probes eight integer dtypes before giving up and probes
`float32` then `float64` for boxes (`crimson/src/zarr_loader.cpp:318-383`,
`462-507`, `515-569`). Those probes are compatibility behavior, not the future
schema. Crimson's new schema-version adapter should open the manifest-declared
exact dtype once.

Two current fallback semantics must remain confined to the legacy adapter:

- missing scores render as `1.0` and missing classes render as `0`
  (`crimson/src/zarr_loader_detections.cpp:1296-1306`);
- loaded `int32` class IDs are cast to `uint16` without range validation
  (`crimson/src/zarr_loader_detections.cpp:760-785`, `808-828`).

The older Crimson read-contract document also calls scores/classes/counts
optional and accepts multiple dtypes (`crimson/docs/crimson_detect_bbox_read_contract.md:
54-71`, `141-147`). That describes compatibility, not the new exact contract.

## Exact Target Invariants

### Cardinality and ordering

- `frame_row_offsets[0] == 0`.
- Offsets are nondecreasing, have length `F + 1`, and end at `N`.
- `frame_row_offsets[f + 1] - frame_row_offsets[f]` equals the number of rows
  whose `frame_indices` value is `f`.
- `frame_indices` is nondecreasing, so each frame owns one contiguous row range.
- Within a frame, producer order is persisted but carries no semantic rank.
- `source_acquisition_frame_index` is row-aligned and, for the current complete
  full-video detect contract, exactly equals `frame_indices` after widening to
  `int64`.

This is already proven workable by refined detections: their publisher sorts
rows by frame plus stable row ID and materializes cumulative `frame_offsets`
(`src/fisheye/shared/refined_detect_curation.py:1685-1718`); their validator
requires offset/count agreement and the corresponding row order
(`src/fisheye/shared/refined_detect_identity.py:228-285`).

### Values, missingness, and fill

- Bounding boxes are finite source-camera-normalized `cx,cy,w,h`, have positive
  width and height, and remain inside `[0,1]`.
- Scores are finite and in `[0,1]`.
- Class IDs are required `int32`, have no sentinel, and are in `[0,65535]` while
  Crimson's public box type remains `uint16`; when a bound model taxonomy is
  available they must also be less than its class count.
- `instance_key` is required and unique. Row positions are not identity.
- No raw canonical row uses NaN, `-1`, an empty string, or a fill value to mean
  absent. Absence means no row in the frame's offset interval.
- Arrays are completely written before immutable publication. Zarr fill values
  are physical initialization values only and must never be interpreted as
  missing data. Completion validation checks every value-level invariant and
  cross-array derivation before selector visibility.

Current code enforces finite, contained normalized bboxes and exact geometry
projections (`src/fisheye/shared/observation_coordinate_publication.py:809-873`,
`1144-1157`), but it currently checks only the exact `int32` shape of class IDs,
not their range (`src/fisheye/shared/observation_coordinate_publication.py:
2993-3027`). Class range and score range are therefore new vNext validation,
not claims about every old archive.

### Empty data

Current canonical publication already requires all zero-row arrays with exact
dtypes and full-length zero count vectors for a valid empty observation run
(`src/fisheye/shared/observation_coordinate_publication.py:3144-3228`). The new
contract preserves those row shapes, changes continuous geometry to `float32`,
adds the all-zero `(F+1,)` offset array, and treats count arrays according to the
selected transition/legacy profile.

## Phase 2 Consequences

1. Add exact contracts for every target row above, including
   `frame_row_offsets`; do not reuse the current incomplete `DETECT_SPEC`.
2. Make the canonical core require offsets and treat counts as derived profile
   bindings: optional `frame_counts`, legacy-only `n_detections`.
3. Add a compatibility adapter from existing raw count arrays and refined
   `frame_offsets` to the one logical frame-row-offset contract.
4. Enforce frame-contiguous ordering in the writer and validator before offsets
   are published.
5. Give Crimson one exact schema-version path that reads consolidated manifest
   metadata and persisted offsets; retain current dtype/name/default probing only
   for archives without that schema version.
