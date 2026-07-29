# Keypoint v2 Crimson Fixture Contract

Date: 2026-07-29

Status: Palette fixture complete; Crimson reader and mounted-read gate pending

## Purpose

This handoff freezes the first cross-language fixture for Palette's future
keypoint boundary. It gives Crimson exact typed inputs for:

- raw keypoint observations v2;
- observation-local keypoint quality v1; and
- derived body frame v1, including heading outside the keypoint snapshot.

The fixture is immutable, benchmark-only, selector-ineligible, and
registry-unregistered. It is correctness and consumer-integration evidence. It
is not a production selector, a refined-keypoint fixture, or a long-recording
physical-profile promotion test.

## Immutable Handoff

The server package is:

```text
/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/
keypoint_storage/integration/20260128_cropv2_keypoint_v2_20260729_v4/
```

On the mounted macOS workstation, replace `/groups/johnson/johnsonlab` with
`/Volumes/johnsonlab`:

```text
/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/
keypoint_storage/integration/20260128_cropv2_keypoint_v2_20260729_v4/
```

| Artifact | Store and explicit run | Logical schema | Manifest digest |
| --- | --- | --- | --- |
| Raw keypoints | `raw.zarr/keypoints_runs/raw_keypoints_crop_v2_yolo_v2` | `palette.stage.keypoint_observations` v2 | `227f0c80065a38d77604b0638bb16a22cd513b383609d364b4481a4fb0cf8db6` |
| Quality | `quality.zarr/keypoint_quality_runs/keypoint_quality_crop_v2_v1` | `palette.stage.keypoint_quality` v1 | `3d0af6dab6ca0ddc478c80755c040c2af2381e00166ee8a4cab7f8d9cb920e81` |
| Body frame | `body_frame.zarr/analysis/body_frame_runs/body_frame_crop_v2_keypoints_v1` | `palette.analysis.body_frame` v1 | `a8b12539669174bf20ebaf181b0c341148903588fc5ae27af46f94e24a2ab1af` |

The package handoff is `handoff_manifest.json`; its file SHA-256 is
`cd33ac60e2f72f614a0ea5f2583d08229b9dee22d2ddb2692a56a284f4f2d8c2`.
The producing Palette revision was the clean commit
`79e8108f4705e9627888d21b1e4192b345b47722`, executed as LSF job
`153230652` on `h08u18.int.janelia.org`.

The package contains 23,287 frames, 22,926 observation rows, three ordered
landmarks, and a 4512x4512 source-camera extent. Every store has an exact
23,288-element `frame_row_offsets`, 361 empty frames, no multi-row frame in
this particular fixture, and unique `instance_key` values. The schemas and
consumer must still support multiple observations per frame; that invariant is
covered by Palette's synthetic contract tests rather than this real recording.

## Exact Array Surface

All arrays are Zarr v3. At this bounded size each is one unsharded physical
chunk with `bytes` followed by `zstd(level=0, checksum=false)`. This is the
intentional single-object case. It does not select an unsharded policy for
longer recordings.

### Raw keypoints: 15 arrays

| Array | Exact dtype and shape |
| --- | --- |
| `instance_key` | `uint64[22926]` |
| `source_crop_row_ids` | `int64[22926]` |
| `source_acquisition_frame_index` | `int64[22926]` |
| `frame_indices` | `int64[22926]` |
| `frame_row_offsets` | `int64[23288]` |
| `source_crop_row_signature` | `uint8[22926,32]` |
| `keypoint_row_signature` | `uint8[22926,32]` |
| `keypoints_roi` | `float32[22926,3,2]` |
| `keypoints_img` | `float32[22926,3,2]` |
| `keypoint_confidences` | `float32[22926,3]` |
| `keypoint_valid` | `bool[22926,3]` |
| `pose_confidence` | `float32[22926]` |
| `pose_bbox_xyxy_roi` | `float32[22926,4]` |
| `pose_bbox_xyxy_img` | `float32[22926,4]` |
| `pose_success` | `bool[22926]` |

`keypoints_roi` is the landmark authority. `keypoints_img` is the required
source-camera-pixel projection cache. Raw v2 deliberately has no heading,
embedded quality, normalized keypoints, or count aliases.

### Keypoint quality: 13 arrays

| Array | Exact dtype and shape |
| --- | --- |
| `instance_key` | `uint64[22926]` |
| `source_keypoint_row_ids` | `int64[22926]` |
| `source_keypoint_row_signature` | `uint8[22926,32]` |
| `frame_indices` | `int64[22926]` |
| `frame_row_offsets` | `int64[23288]` |
| `keypoint_metric_values` | `float32[22926,3,1]` |
| `keypoint_metric_valid` | `bool[22926,3,1]` |
| `pose_metric_values` | `float32[22926,1]` |
| `pose_metric_valid` | `bool[22926,1]` |
| `keypoint_quality_flags` | `uint16[22926,3]` |
| `pose_quality_flags` | `uint16[22926]` |
| `proposed_keypoint_valid` | `bool[22926,3]` |
| `proposed_pose_usable` | `bool[22926]` |

This run is diagnostic. It cannot replace the raw coordinates or act as an
accepted review result. Its exact source-manifest binding is the raw manifest
digest above.

### Body frame: 10 arrays

| Array | Exact dtype and shape |
| --- | --- |
| `instance_key` | `uint64[22926]` |
| `source_keypoint_row_ids` | `int64[22926]` |
| `source_keypoint_row_signature` | `uint8[22926,32]` |
| `frame_indices` | `int64[22926]` |
| `frame_row_offsets` | `int64[23288]` |
| `origin_xy` | `float32[22926,2]` |
| `forward_axis_xy` | `float32[22926,2]` |
| `left_axis_xy` | `float32[22926,2]` |
| `axis_valid` | `bool[22926]` |
| `heading_deg` | `float32[22926]` |

Body-frame values use continuous source-camera pixels. Heading uses
`atan2(-y, x)` in degrees and is derived from the exact digest-bound skeleton
recipe. It is not part of keypoint authority.

## Metadata And Object Receipt

Palette's final reopen gates passed for all three stores. A separate read-only
handoff check confirmed direct/consolidated metadata equivalence for the run
group and every array: 16 declarations for raw, 14 for quality, and 11 for body
frame. Group equivalence permits only the standard omitted versus exact empty
inline consolidated-metadata envelope; array declarations are exact.

The stores contain 38 array payload objects. Including array and stage-group
metadata, they represent 82 incremental stage objects:

| Stage | Payload objects | Array metadata | Stage-group metadata | Incremental total |
| --- | ---: | ---: | ---: | ---: |
| Raw keypoints | 15 | 15 | 2 | 32 |
| Quality | 13 | 13 | 2 | 28 |
| Body frame | 10 | 10 | 2 | 22 |
| **Combined** | **38** | **38** | **6** | **82** |

Standalone filesystem counts, including each store root, are 33 files for raw,
29 for quality, and 24 for body frame, plus the package handoff. Apparent store
sizes are 3,925,803, 1,475,797, and 1,800,087 bytes, respectively.

At the representative 1,188,000-frame/1,187,087-row planning scale, the same
byte-derived policy estimates 17, 14, and 11 payload objects: 42 payload and 86
incremental stage objects total. That estimate is planning evidence only; this
fixture does not exercise those indexed shards.

## Crimson Consumer Boundary

Crimson should implement the new reader in shared backend-independent C++.
Platform code should provide only path/session wiring. The v2/v1 reader must:

1. require the explicit store and run paths above and fail closed;
2. validate the exact manifest schema/version/digest and all 38 consolidated
   declarations without dtype probing or legacy aliases;
3. open only exact compile-time TensorStore dtypes;
4. validate each `frame_row_offsets` as shape `F+1`, monotone, starting at zero,
   ending at `N`, and grouping the corresponding `frame_indices` exactly;
5. preserve all rows in `[offsets[f], offsets[f+1])`, including empty and
   future multi-observation frames;
6. retain `instance_key` as observation/edit-lineage identity rather than a
   within-frame ordinal or subject/track ID;
7. validate the quality and body-frame source manifest, row-signature, row-ID,
   frame-index, and instance-key bindings before combining stages; and
8. keep legacy keypoint/embedded-heading readers behind a separate explicit
   compatibility adapter.

For normal keypoint/body-frame presentation, the hot raw surface is
`frame_row_offsets`, `instance_key`, `keypoints_img`, `keypoint_valid`, and
`pose_success`. The hot body-frame surface is its offsets and identity plus
`origin_xy`, both axes, `axis_valid`, and `heading_deg`. `keypoints_roi`,
confidence/bbox columns, and source lineage may remain pageable or lazy for
crop inspection and editing.

All 13 quality arrays are optional diagnostic payload. Crimson may validate
their consolidated declarations and source binding at open, but ordinary
playback must perform zero quality-array payload reads. Opening the quality
panel may create its own bounded reader and retain its offsets once.

## Required Crimson Gate

The first gate should be a headless exact-schema integration benchmark against
the mounted package, followed by one GUI smoke. It should report:

- exact declaration, dtype, rank, codec, manifest, and source-binding results;
- one retained offset read per opened repository and zero later offset reads;
- cold open and first usable overlay/body-frame readiness;
- warm deterministic random-frame latency;
- 70-frame window latency at a simulated 700 FPS demand rate;
- full traversal digest equality with the manifest's logical array digests;
- concurrent keypoint/body-frame reads, cancellation, and zero stale results;
- payload opens/reads proving quality remains lazy during ordinary playback;
- file reads, transferred bytes, cache behavior, deadline misses, and peak RSS;
  and
- identical observation identities and row ranges across paged and any
  resident representation.

Negative tests must reject a missing or extra array, wrong dtype/rank, invalid
manifest digest, forbidden legacy alias, malformed offsets, duplicate
`instance_key`, mismatched source signature/row identity, embedded raw heading,
and silent fallback from an invalid explicit v2 run.

This bounded fixture can accept the logical contracts and consumer behavior.
It cannot by itself promote the long-recording keypoint physical profile. A
full-duration physical-layout fixture should be requested only after the exact
reader passes and only if the byte-derived object estimate or mounted access
measurements reveal a real scaling question.

## Palette Production Boundary

The canary was computed from a durable NRS pixel cache, but that cache is not a
Crimson input and is not stored inside an analysis archive. The source crop
archive, production selectors, registry, training artifacts, and production
stores were unchanged.

Palette must not activate production keypoint/quality/body-frame selectors on
the strength of this handoff. Remaining gates include the exact
refined-keypoint-v2 snapshot/compaction contract, production DAG adoption,
numerical comparison against legacy float64 values, representative
long-recording publication/read measurements, and Crimson acceptance.
