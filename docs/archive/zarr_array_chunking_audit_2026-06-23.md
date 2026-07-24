# Zarr Array Chunking Audit

Date: 2026-06-23

Status: diagnostic note and recommendation draft.

Related utility:

```bash
scripts/py -m fisheye.utils.audit_zarr_array_sizes <analysis.zarr> --top 40 --sort logical
scripts/py -m fisheye.utils.audit_zarr_array_sizes <analysis.zarr> --physical --top 40 --sort chunk-files
```

## Why This Audit Exists

Crimson does random seeks, frame jumps, review navigation, and selective display.
For small geometry-like arrays, full startup preload is often reasonable and
improves random seeking. But users also edit boxes, keypoints, and masks. That
means read strategy and write strategy should not be collapsed into one rule:

- Small arrays can be loaded into memory for reads.
- Editable arrays still need row-granular writes, overlay/review layers, or
  deliberately chunked storage.
- Dense masks and pixels should remain lazy/chunked, even if compression makes
  their physical size small on disk.

The practical goal is to reduce tiny-file overhead where it is safe, while not
making mask painting, keypoint edits, or bounding-box edits expensive.

## Recording Copies Audited

### Segmented Example Recording

Copied earlier to:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_segmented_recording
```

Representative analysis Zarr:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_segmented_recording/zarr/2026-01-28T21-47-47Z_arena_1_DefaultScreen_analysis.zarr
```

Metadata-only audit result:

```text
arrays=2760
logical=74.09 GiB
```

Focused physical audit of refined subject masks and contours:

```text
arrays=5
logical=22.77 GiB
physical=30.66 MiB
```

Largest relevant surfaces:

| Surface | Logical | Physical | Chunk | Chunk files | Read/write interpretation |
| --- | ---: | ---: | --- | ---: | --- |
| `refined_subject_masks_runs/.../masks_roi` | 22.67 GiB | 16.06 MiB | `16x1x512x512` | 5808 | Huge logical dense mask surface; lazy reads only. |
| `components/subject_body/contours/points_xy` | 73.49 MiB | 11.41 MiB | `4096x2` | 2352 | Small enough to preload in some contexts, but ragged edits require care. |
| `components/swim_bladder/contours/points_xy` | 9.62 MiB | 1.23 MiB | `4096x2` | 308 | Preloadable geometry/index surface. |
| `components/eye_left/contours/points_xy` | 7.92 MiB | 1008.70 KiB | `4096x2` | 254 | Preloadable geometry/index surface. |
| `components/eye_right/contours/points_xy` | 7.82 MiB | 988.88 KiB | `4096x2` | 251 | Preloadable geometry/index surface. |

Important distinction:

- `masks_roi` is physically tiny in this example because the masks compress
  aggressively.
- It is still logically huge and random frame access still crosses dense mask
  chunks.
- Physical size alone is not enough to decide Crimson startup preload policy.

### Heartrate Example Recording

Copied on 2026-06-23 from:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop
```

Destination:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording
```

Copy method:

```bash
scripts/py -m fisheye.utils.copy_recording \
  /groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop \
  /groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording \
  --destination-is-recording-dir \
  --apply
```

Copy result:

```text
copy_status=ok
```

Analysis Zarr:

```text
/groups/johnson/johnsonlab/jeremy/jlcrsi/example_heartrate_recording/zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr
```

Metadata-only audit result:

```text
arrays=916
logical=424.41 MiB
```

Physical audit result:

```text
arrays=916
logical=424.41 MiB
physical=141.60 MiB
```

Largest surfaces in the analysis Zarr are small enough to preload for Crimson
reads:

| Surface family | Example arrays | Logical size | Current chunking | Interpretation |
| --- | --- | ---: | --- | --- |
| Keypoints | `keypoints_img`, `keypoints_norm`, `keypoints_roi` | 10.04 MiB each | `1024x5x2` | Preload for reads; write edits row-wise or in review overlay. |
| Refined keypoints | `refined_keypoints_runs/.../keypoints_img` | 10.04 MiB | `1024x5x2` | Preload for reads; row/overlay writes. |
| Detection/refined boxes | `bbox_img_xyxy`, `bbox_norm_coords` | 4.02 MiB each | `65536x4` or `1024x4` | Preload for reads; moderate/large chunks are fine. |
| Reason bytes | refined detect/keypoint `reason_bytes` | 6-8 MiB | varies | Preloadable, but editable/review surfaces should preserve row semantics. |
| Stimulus calibration images | PNG byte buffers | 6-10 MiB | one chunk | Read-only preload is fine. |

The main file-count smell in the heartrate example is not large arrays. It is
small arrays split into many tiny chunks:

```text
arena_assignment_runs/.../n_detections_per_arena
logical=547.01 KiB
chunk=100x1
chunks/files ~= 1401/1322
```

This is the clearest evidence that some small table-like arrays should be
chunked much larger.

## Recommended Surface Policies

### 1. Detection Boxes And Row Geometry

Examples:

- `detect_runs/*/bbox_norm_coords`
- `refined_detect_runs/*/instances/bbox_img_xyxy`
- `refined_detect_runs/*/instances/bbox_norm_coords`
- `frame_indices`
- `scores`
- `class_ids`
- `source_*_row_ids`
- `roi_coordinates_full`

Recommendation:

- Full preload in Crimson is reasonable for typical recordings.
- Prefer row chunks of at least `4096` rows.
- For larger read-mostly arrays, `16384` to `65536` rows is reasonable.
- Avoid very small chunks like `100` rows unless the array is actively edited
  with frequent tiny writes.

Write policy:

- Do not rely on rewriting one giant base array for manual edits.
- Prefer explicit review/manual overlay surfaces, or row-granular writes into
  chunked editable outputs.

### 2. Keypoints And Refined Keypoints

Examples:

- `keypoints_img`
- `keypoints_roi`
- `keypoints_norm`
- `keypoint_confidences`
- `derived metrics` such as edge distances or triangle angles

Recommendation:

- Full preload is reasonable for Crimson random seek and display.
- Current `1024` row chunks are safe but likely conservative.
- Candidate future chunk range: `4096` to `16384` rows.

Write policy:

- Keep manual keypoint edits row-addressable.
- For review workflows, consider immutable model outputs plus separate manual
  correction layers instead of in-place base-array mutation.

### 3. Crop Geometry

Examples:

- `crop_runs/*/roi_coordinates_full`
- `source_refined_row_ids`
- `frame_indices`
- `bbox_norm_coords`

Recommendation:

- Preload for Crimson is usually safe.
- Prefer larger row chunks than `1000` where the arrays are read-mostly.
- `4096` to `16384` rows is a reasonable next candidate.

Write policy:

- Crop geometry should remain stable once emitted because downstream keypoints
  and masks depend on exact row lineage.
- If corrections are needed, write a new crop/refined run rather than mutating
  the old one in place.

### 4. Dense Pixels And Dense Masks

Examples:

- `raw_video/images_full`
- `raw_video/images_ds`
- `crop_runs/*/roi_images`
- `subject_mask_runs/*/mask_probs_roi`
- `refined_subject_masks_runs/*/masks_roi`
- component `source_seed_masks_roi`

Recommendation:

- Do not preload into Crimson by default.
- Keep lazy/chunked access.
- Chunk by the dominant access pattern:
  - frame/ROI review wants reasonably small row chunks
  - bulk training/export can tolerate larger chunks
  - cluster publication may prefer bitpacked/RLE/packed artifacts rather than
    bigger dense chunks

Write policy:

- Dense masks are expensive to edit if stored as giant chunks.
- For paintable masks, prefer an editable component-separated dense or
  bitpacked representation.
- Treat RLE as compact final/archive/export storage unless tooling explicitly
  supports efficient component-level patching.

### 5. Refined Subject Mask Contours And RLE-Like Ragged Geometry

Examples:

- `components/*/contours/points_xy`
- `components/*/contours/ptr`
- `components/*/contours/len`
- future `mask_rle/*`

Recommendation:

- Preloading pointer/index arrays is reasonable.
- Preloading `points_xy` may be reasonable when the total is small enough.
- Keep in mind that ragged edits are structurally different from dense row
  edits: changing one row can change downstream offsets.

Write policy:

- Prefer component-level rewrite or append-only/manual-layer strategies.
- Do not assume single-mask edits are cheap for packed ragged arrays.

### 6. Stimulus, Analytics, And Kinematic Time Series

Examples:

- `analysis/stimulus_runs/*/tracking_data/chaser_states/*`
- `analysis/track_kinematics_runs/*`
- summary metrics

Recommendation:

- These arrays are usually small and read-mostly.
- Full preload for visualization is often fine.
- Increase tiny row chunks where arrays are split into hundreds or thousands of
  files.
- Candidate chunk range: `4096` to `65536` rows, depending on array width.

Write policy:

- Prefer append/new-run semantics for regenerated analytics.
- These are good candidates for larger chunks because they are usually not
  edited interactively.

## Proposed Chunking Direction

| Surface type | Current issue seen | Suggested next default |
| --- | --- | --- |
| Small scalar/table arrays | Some arrays use `100` or `1000` row chunks and create many tiny files. | Use `4096-65536` row chunks when read-mostly. |
| Detection/refined box arrays | Already small enough to preload; chunking varies. | Use `16384-65536` rows for model/refined outputs. |
| Keypoint arrays | `1024` row chunks are safe but conservative. | Test `4096` or `8192` rows first. |
| Crop geometry arrays | `1000` row chunks produce avoidable file count. | Test `4096-16384` rows. |
| Dense masks/pixels | Logical size can be huge even when physical size is small. | Keep lazy; do not optimize for preload. |
| Paintable masks | RLE is compact but awkward for single-mask edits. | Prefer component-separated dense/bitpacked editable surfaces. |
| Final/archive masks | Dense logical shape is expensive to publish and move. | RLE or packed artifacts once review is complete. |

## Crimson Access-Pattern Follow-Up

Read-only Crimson inspection confirmed that the first candidate arrays are not
on the synchronous seek path. Crimson generally reads these small lineage/count
arrays fully at archive/run load, caches the resulting vectors/maps, and then
uses cached memory during seek/render.

### Candidate Families

| Candidate family | Crimson access pattern | Cached? | Written by Crimson? | Chunking interpretation |
| --- | --- | --- | --- | --- |
| `crop_runs/*/frame_indices` | Read fully by crop metadata paths. Modern mask placement also reads `roi_coordinates_full`. Other listed crop lineage arrays are not consumed. | Yes | No | Safe to chunk larger; preload is appropriate. |
| `crop_runs/*/{bbox_norm_coords,detection_indices,source_detect_row_index,source_refined_row_ids}` | Not currently consumed by Crimson crop/mask placement paths. | Not applicable | No | Safe to chunk larger from Crimson's side. |
| `keypoints_runs/*/frame_indices` | Read fully when raw keypoints are selected. Counts are inferred from shapes. | Yes | No | Safe to chunk larger or preload whole. |
| `keypoints_runs/*/{detection_indices,source_detect_row_index,source_refined_row_ids,n_keypoints,n_rois}` | Not consumed for rendering. | Not applicable | No | Safe to chunk larger from Crimson's side. |
| `refined_keypoints_runs/*/frame_indices` | Preferred keypoint source; read fully at archive load. | Yes | No | Safe to chunk larger or preload whole. |
| `refined_keypoints_runs/*/{detection_indices,source_detect_row_index,source_refined_row_ids,n_rois}` | Not consumed for rendering. | Not applicable | No | Safe to chunk larger from Crimson's side. |
| `detect_runs/*/{frame_counts,n_detections}` | Read fully at archive load to build frame offsets. | Yes | No for raw detect runs | Safe to chunk larger/preload whole. |
| `analysis/track_kinematics_runs/*/tracks/*/*` | Discovered at startup; full arrays loaded only after user requests track kinematics. | Yes after deferred load | No | Safe to chunk larger; improves deferred load more than startup. |

Crimson overlay rendering needs cached keypoint `frame_indices`, detection
`frame_offsets`, and refined-mask `frame_indices` plus `source_crop_row_ids` and
`crop_runs/<source>/{frame_indices,roi_coordinates_full}`. It does not need
`source_detect_row_index` or `source_refined_row_ids` from crop/keypoint runs
for rendering.

### Crimson-Safe First Batch

Based on the access-pattern review, the next canary batch can safely target
lineage/count arrays whose current chunks are around `1000` rows:

- `crop_runs/*/detection_indices`
- `crop_runs/*/frame_indices`
- `crop_runs/*/source_detect_row_index`
- `crop_runs/*/source_refined_row_ids`
- `keypoints_runs/*/detection_indices`
- `keypoints_runs/*/frame_indices`
- `keypoints_runs/*/source_detect_row_index`
- `keypoints_runs/*/source_refined_row_ids`
- `refined_keypoints_runs/*/detection_indices`
- `refined_keypoints_runs/*/frame_indices`
- `refined_keypoints_runs/*/source_detect_row_index`
- `refined_keypoints_runs/*/source_refined_row_ids`
- `detect_runs/*/frame_counts`
- `detect_runs/*/n_detections`

Recommended canary target chunk: first-axis chunk `16384`, preserving trailing
dimensions. This should reduce each affected array from roughly `129-132`
physical chunk files to about `8-9` files for the heartrate canary.

### Canary Batch Result

Applied on 2026-06-24 to the local canary only:

```text
/nvme1/recordings/chunking_canary_2026-06-24_heartrate/zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr
```

Command:

```bash
scripts/py -m fisheye.utils.rechunk_zarr_array_batch \
  /nvme1/recordings/chunking_canary_2026-06-24_heartrate/zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr \
  --preset crimson-lineage-v1 \
  --row-chunk 16384 \
  --reason 'canary batch rechunk Crimson-cached lineage/count arrays' \
  --apply
```

Result:

```text
matched=14
updated=14
old chunk count total=1858
new chunk count total=126
```

Affected arrays:

- `crop_runs/crop_2026-06-17_19-37-50/detection_indices`: `132 -> 9` chunks
- `crop_runs/crop_2026-06-17_19-37-50/frame_indices`: `132 -> 9` chunks
- `crop_runs/crop_2026-06-17_19-37-50/source_detect_row_index`: `132 -> 9` chunks
- `crop_runs/crop_2026-06-17_19-37-50/source_refined_row_ids`: `132 -> 9` chunks
- `detect_runs/detect_goodcopbadcop_detect_artifact_refine_20260617T025729Z_0004/frame_counts`: `137 -> 9` chunks
- `detect_runs/detect_goodcopbadcop_detect_artifact_refine_20260617T025729Z_0004/n_detections`: `137 -> 9` chunks
- `keypoints_runs/keypoints_goodcopbadcop_kpt5_traditional_v2_flat_cache_20260617/detection_indices`: `132 -> 9` chunks
- `keypoints_runs/keypoints_goodcopbadcop_kpt5_traditional_v2_flat_cache_20260617/frame_indices`: `132 -> 9` chunks
- `keypoints_runs/keypoints_goodcopbadcop_kpt5_traditional_v2_flat_cache_20260617/source_detect_row_index`: `132 -> 9` chunks
- `keypoints_runs/keypoints_goodcopbadcop_kpt5_traditional_v2_flat_cache_20260617/source_refined_row_ids`: `132 -> 9` chunks
- `refined_keypoints_runs/refined_keypoints_2026-06-18_15-00-03/detection_indices`: `132 -> 9` chunks
- `refined_keypoints_runs/refined_keypoints_2026-06-18_15-00-03/frame_indices`: `132 -> 9` chunks
- `refined_keypoints_runs/refined_keypoints_2026-06-18_15-00-03/source_detect_row_index`: `132 -> 9` chunks
- `refined_keypoints_runs/refined_keypoints_2026-06-18_15-00-03/source_refined_row_ids`: `132 -> 9` chunks

Post-batch audit shows these arrays are no longer the top chunk-file offenders.
The next remaining class is track-kinematics arrays with `1024` row chunks and
roughly `129` files each, plus keypoint count arrays (`n_keypoints`, `n_rois`)
that were intentionally not included in the first preset.

### Track-Kinematics Canary Batch Result

Applied on 2026-06-24 to the same local canary only:

```bash
scripts/py -m fisheye.utils.rechunk_zarr_array_batch \
  /nvme1/recordings/chunking_canary_2026-06-24_heartrate/zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr \
  --preset track-kinematics-v1 \
  --row-chunk 16384 \
  --reason 'canary batch rechunk deferred-load track kinematics arrays' \
  --apply
```

Dry-run showed:

```text
matched=100
old chunk count total=12525
new chunk count total=868
```

Apply result:

```text
matched=100
updated=100
old chunk count total=12525
new chunk count total=868
```

This preset targets only:

```text
analysis/track_kinematics_runs/*/*/tracks/*/*
```

Most arrays changed from `1024` row chunks to `16384` row chunks, reducing
`129 -> 9` chunks per full-length array. Small per-second arrays changed from
three chunks to one chunk. Crimson track-kinematics load/display was validated
after this rewrite; see the next section.

Post-track-batch audit shows the next remaining chunk-file classes are:

- keypoint/refined-keypoint primary and derived arrays around `129` files each
- `crop_runs/*/bbox_norm_coords`
- raw detect bbox/frame/scores arrays
- stimulus `tracking_data/chaser_states` arrays around `81` files each

Those were intentionally not included in the track preset.

### Crimson Track-Kinematics Smoke Result

Crimson GUI validation passed against the local canary after the
`track-kinematics-v1` batch rechunk.

Command:

```bash
env DISPLAY=:1 XAUTHORITY=/run/user/64406/.mutter-Xwaylandauth.0U8EP3 \
  release/redgui \
  --zarr /nvme1/recordings/chunking_canary_2026-06-24_heartrate/zarr/2026-06-14T21-12-08Z_arena_4_GoodCopBadCop_analysis.zarr \
  --swap-interval 0 \
  --frame-cap-fps 120 \
  --no-mask-perf-log
```

The tester then used `xdotool` to click **Load Track Kinematics** in Crimson's
Analysis Timeline panel and to seek/play.

Main log:

```text
/tmp/crimson_track_kinematics_gui_load_20260624_163856_managed.log
```

Screenshots:

- `/tmp/crimson_track_kinematics_loaded.png`
- `/tmp/crimson_track_kinematics_middle_seek.png`
- `/tmp/crimson_track_kinematics_late_seek.png`
- `/tmp/crimson_track_kinematics_after_playback.png`

Validated behavior:

- Track kinematics were discovered at startup as deferred movement analysis
  data.
- User-triggered lazy load succeeded.
- Loaded run:
  `goodcopbadcop_tk_hyst4_low2_latch_s005`, track `id_0`.
- Loaded movement datasets:
  speed `filtered`, `smoothed`, `raw`, and `averaged`.
- Each loaded dataset reported `131641` samples.
- Lazy load completed in `187.468 ms` in the managed smoke; a separate
  interactive run reported `142.833 ms`.
- Analysis Timeline switched from "available but not loaded" to populated track
  controls.
- Speed trace, time axis, track position overlay, filtered speed overlay, and
  smoothed heading overlay displayed.
- Seeking after track load worked at early, middle, and late positions:
  - early: frame `0`
  - middle: frame `48944`
  - late: frame `122360`
- Playback from the late range advanced to frame `124023` while movement
  overlay remained populated.
- No TensorStore/Zarr loader errors appeared from the rechunked
  track-kinematics arrays.

Known non-blocking warnings:

- The root `zarr.json` still has non-standard
  `imageio_metadata.nframes: Infinity`, causing clipped resolver probing to log
  a JSON parse warning.
- Stimulus corrected-frame metadata is missing and Crimson falls back to legacy
  mapping.

Scope caveat:

- This canary still does not contain `refined_subject_masks_runs`, so this smoke
  does not validate refined subject-mask placement.

### Crimson Canary Smoke Result

Crimson GUI/playback smoke passed against the local canary after the
`crimson-lineage-v1` batch rechunk.

Validated playback ranges:

| Range | Result |
| --- | --- |
| `0:300` | pass, `presented_frame=300`, `elapsed_s=2.99339` |
| `50000:50300` | pass, `presented_frame=50300`, `elapsed_s=2.9929` |
| `120000:120300` | pass, `presented_frame=120300`, `elapsed_s=2.99309` |

Crimson logs:

- `/tmp/crimson_playback_smoke_20260624_161748.log`
- `/tmp/crimson_playback_smoke_20260624_161836.log`
- `/tmp/crimson_playback_smoke_20260624_161853.log`

Validated from logs:

- Raw detections loaded:
  `detect_goodcopbadcop_detect_artifact_refine_20260617T025729Z_0004`,
  `131650` detections over `140035` frames.
- Refined detections loaded and used as primary:
  `refined_detect_2026-06-16_23-15-05/refined`, `131641` detections.
- Refined keypoints loaded:
  `refined_keypoints_2026-06-18_15-00-03`, review state `needs_review`.
- Keypoint overlay path loaded from the refined keypoint run.
- Crop metadata loaded: `crop_2026-06-17_19-37-50`, `131641` ROIs.
- Nonzero seek startup/playback worked for middle and late ranges.
- No Crimson loader errors appeared from the rechunked candidate arrays.

Scope caveats:

- This canary does not contain `refined_subject_masks_runs`, so this smoke does
  not validate refined subject-mask pixel placement.
- A pre-existing root metadata issue still logs during clipped collection
  probing: root `zarr.json` has `imageio_metadata.nframes: Infinity`, which is
  non-standard JSON. Crimson tolerates it for normal load/playback.

### Keep Conservative

This Crimson review does not justify rechunking heavier or more editable
surfaces yet. Keep the following conservative until separately benchmarked:

- refined keypoint coordinate/quality arrays that may be edited
- refined subject-mask `masks_roi`
- subject-mask `mask_rle` payloads
- crop `roi_images`
- dense video/pixel arrays

## Practical Next Steps

### Writer Defaults Implemented

The first writer-default slice is now implemented through
`fisheye.shared.zarr.chunk_profiles.geometry_preload_v1`.

Policy:

- Row chunk: `16384`.
- Storage profile id: `geometry_preload_v1`.
- Scope: small read-mostly metadata/count/lineage arrays that Crimson validated
  as full-preload or deferred-full-load surfaces.
- Excluded: dense masks, dense pixels, crop `roi_images`, refined keypoint
  coordinate/quality arrays, editable authoritative mask payloads.

New writes covered:

- `detect_runs/*/{frame_counts,n_detections}`
- `crop_runs/*/{frame_indices,detection_indices,source_detect_row_index,source_refined_row_ids}`
- copied keypoint/refined-keypoint lineage arrays for the same lineage names
- `analysis/track_kinematics_runs/*/*/tracks/*/*` deferred-load arrays

The row-lineage copy helper keeps its previous behavior by default. Writers
must opt into `use_geometry_preload_profile=True`, so legacy or unvalidated
surfaces do not silently inherit the profile.

### Remaining Steps

1. Run one true writer canary to verify newly produced arrays are born with the
   profile, not just rechunked after the fact.
2. Keep the audit utility and run it on each new representative recording before
   changing additional writer defaults.
3. Start with low-risk read-mostly arrays not yet covered:
   - arena assignment scalar arrays
   - selected analytics time series outside track kinematics
   - crop geometry arrays after a separate Crimson/random-seek check
   - keypoint metrics after edit/review behavior is confirmed
4. Do not change dense mask/pixel chunking based only on physical compressed
   size; use random seek benchmarks and edit behavior.
5. For Crimson, implement two explicit policies:
   - preload small immutable/read-mostly arrays
   - keep editable authoritative surfaces row-addressable even when also cached
     in memory for display
6. Later, promote stable chunking recommendations into
   `docs/zarr_storage_lifecycle_policy.md` after writer defaults and benchmarks
   agree.

## Current Conclusion

The strongest immediate recommendation is to chunk small read-mostly arrays
larger. Many geometry, keypoint, and analytics arrays are only kilobytes to
megabytes but can produce hundreds or thousands of physical chunk files. Larger
chunks there should improve startup scans, transfer behavior, and random-seek
cache efficiency without harming review workflows.

Dense masks and dense pixels need a different policy. They should stay lazy and
chunked for review, with bitpacked/RLE/packed-artifact representations used for
publication or finalized storage. Physical compressed size should not be used
as the only argument for preloading or giant chunks, because Crimson random
seeks and mask editing are governed by logical chunk access and mutation cost.
