# Realtime vs Offline Detection Comparison Design

Date: 2026-06-17

Purpose: record the GoodCopBadCop inspection around acquisition-time TensorRT
detections, offline/refined detections, and the question of whether acquisition
crop videos can substitute for regenerated ROI caches in some workflows.

## Current Conclusion

Palette already has the right shape for this as an optional diagnostic, not as a
required pipeline stage. The existing module
`src/fisheye/diagnostics/compare_realtime_offline_detections.py` writes
`analysis/detection_comparison_runs/<run>/` and stores numeric arrays plus a PNG
QC artifact. That should be extended rather than creating a second comparison
surface.

The primary realtime source for external-IPC recordings should be
`derived/external_crop_recorder/*_crop_meta.csv`, not the imported H5
`analysis/stimulus_runs/<run>/tracking_data/bounding_boxes` table. The crop-meta
file is frame-continuous and records every crop recorder outcome, including blank
frames. The H5 bounding-box table is still useful as a compatibility source, but
it is shorter because Orange uses a grab-latest/drain policy for that payload.

This diagnostic should answer two separate questions:

1. How close were acquisition-time TensorRT boxes to offline/refined boxes?
2. For frames accepted by offline/refined detection, did the acquisition crop
   video contain the same fish/bbox geometry well enough to avoid re-decoding the
   full-frame source video?

## Evidence

Relevant code and contracts:

- `src/fisheye/diagnostics/compare_realtime_offline_detections.py:1` already
  defines the optional zarr run output pattern.
- `src/fisheye/diagnostics/compare_realtime_offline_detections.py:332`
  currently loads realtime boxes only from imported stimulus H5
  `tracking_data/bounding_boxes`.
- `src/fisheye/diagnostics/compare_realtime_offline_detections.py:713` writes
  arrays and visualization metadata into `analysis/detection_comparison_runs`.
- `src/fisheye/utils/organize_recordings.py:661` records the external crop
  stream in `recording_manifest.json`, including crop metadata, frame clock,
  coordinate spaces, geometry columns, blank-frame policy, and selection policy.
- `docs/orange_runtime_video_artifact_contract.md:44` documents the crop stream
  as a first-class acquisition-time derived input.

Representative organized recording:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop
```

External crop-recorder artifacts include:

```text
derived/external_crop_recorder/Cam2010093_2026-06-14T21-12-08Z_arena_1_crop_external.mp4
derived/external_crop_recorder/Cam2010093_2026-06-14T21-12-08Z_arena_1_crop_meta.csv
derived/external_crop_recorder/Cam2010093_2026-06-14T21-12-08Z_arena_1_yolo_events.jsonl
derived/external_crop_recorder/Cam2010093_2026-06-14T21-12-08Z_arena_1_yolo_perf.csv
```

The crop metadata columns are:

```text
recording_frame_id,local_frame_id,camera_frame_id,timestamp,timestamp_sys,
has_detection,blank_frame,detection_confidence,crop_x,crop_y,crop_w,crop_h,
detection_x,detection_y,detection_w,detection_h
```

`recording_frame_id` is 1-based in the acquisition metadata. Palette offline
`frame_indices` are 0-based. The comparison must therefore use:

```text
offline_frame_index = recording_frame_id - 1
```

`detection_x`, `detection_y`, `detection_w`, and `detection_h` are full-frame
pixel-space detection boxes. `crop_x`, `crop_y`, `crop_w`, and `crop_h` are the
full-frame source geometry for the crop-video frame. `blank_frame=1` means the
crop video encoded a blank placeholder for that recording frame.

## GoodCopBadCop Coverage

Read-only inspection of the 12 organized GoodCopBadCop external-IPC recordings
showed continuous crop metadata coverage. Each crop-meta file starts at
`recording_frame_id=1` and ends at the stream frame count.

| Recording | crop rows | detections | blank rows | H5 bbox rows | H5 first payload frame | H5 last payload frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-06-14T21-12-08Z_arena_1_GoodCopBadCop` | 140035 | 138498 | 1537 | 136531 | 963 | 139028 |
| `2026-06-14T21-12-08Z_arena_2_GoodCopBadCop` | 140035 | 135943 | 4092 | 134173 | 963 | 139028 |
| `2026-06-14T21-12-08Z_arena_3_GoodCopBadCop` | 140035 | 139016 | 1019 | 137043 | 963 | 139028 |
| `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop` | 140035 | 139946 | 89 | 137972 | 964 | 139028 |
| `2026-06-14T21-50-10Z_arena_1_GoodCopBadCop` | 140198 | 139804 | 394 | 137672 | 1122 | 139186 |
| `2026-06-14T21-50-10Z_arena_2_GoodCopBadCop` | 140198 | 139840 | 358 | 137714 | 1122 | 139186 |
| `2026-06-14T21-50-10Z_arena_3_GoodCopBadCop` | 140198 | 140198 | 0 | 138065 | 1122 | 139186 |
| `2026-06-14T21-50-10Z_arena_4_GoodCopBadCop` | 140198 | 138213 | 1985 | 136081 | 1122 | 139186 |
| `2026-06-14T22-33-50Z_arena_1_GoodCopBadCop` | 139693 | 139632 | 61 | 138009 | 620 | 138688 |
| `2026-06-14T22-33-50Z_arena_2_GoodCopBadCop` | 139693 | 139085 | 608 | 137460 | 620 | 138688 |
| `2026-06-14T22-33-50Z_arena_3_GoodCopBadCop` | 139693 | 138595 | 1098 | 136972 | 620 | 138688 |
| `2026-06-14T22-33-50Z_arena_4_GoodCopBadCop` | 139693 | 139684 | 9 | 138059 | 621 | 138687 |

This is the key practical finding: crop metadata is the acquisition-derived
surface with frame-continuous outcomes. H5 `bounding_boxes` should not be used
as the default source when measuring crop-recorder sufficiency.

## Exploratory Comparison Results

The current committed diagnostic was run on one recording using H5
`bounding_boxes` as the realtime source:

```text
offline present: 120221
realtime present: 136531
both present: 118004
offline-only: 2217
realtime-only: 18527
centroid delta p50: 5.36 px
centroid delta p99: 29.96 px
median IoU: 0.8966
```

A prototype crop-meta loader was then used for the same recording. Because it
uses the frame-continuous crop recorder metadata instead of the H5 table, the
offline-only count dropped sharply:

```text
crop-meta rows: 140035
crop-meta detection rows: 138498
offline present: 120221
realtime present: 138498
both present: 119919
offline-only: 302
realtime-only: 18579
centroid delta p50: 4.95 px
centroid delta p99: 15.04 px
median IoU: 0.9039
```

The same prototype computed whether each offline/refined bbox was contained by
the acquisition crop window:

```text
offline rows: 120221
offline center inside crop: 119823 / 120221 = 99.67%
offline full bbox inside crop: 119823 / 120221 = 99.67%
blank/no-detection crop rows for offline frames: 302
```

Across all 12 GoodCopBadCop recordings, most recordings were above 99.6%
full-bbox containment. One recording was a real exception and should be reviewed
visually or with a focused outlier report:

| Recording | offline rows | both | offline-only | realtime-only | delta p50 px | delta p99 px | median IoU | offline bbox inside crop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2026-06-14T21-12-08Z_arena_1` | 120221 | 119919 | 302 | 18579 | 4.95 | 15.04 | 0.9039 | 99.67% |
| `2026-06-14T21-12-08Z_arena_2` | 104424 | 103540 | 884 | 32403 | 4.99 | 3830.29 | 0.8995 | 92.90% |
| `2026-06-14T21-12-08Z_arena_3` | 135297 | 135243 | 54 | 3773 | 4.69 | 11.46 | 0.9064 | 99.93% |
| `2026-06-14T21-12-08Z_arena_4` | 131641 | 131598 | 43 | 8348 | 4.60 | 11.82 | 0.9086 | 99.97% |
| `2026-06-14T21-50-10Z_arena_1` | 139536 | 139490 | 46 | 314 | 4.49 | 9.71 | 0.9142 | 99.96% |
| `2026-06-14T21-50-10Z_arena_2` | 139379 | 139250 | 129 | 590 | 4.68 | 16.59 | 0.9055 | 99.90% |
| `2026-06-14T21-50-10Z_arena_3` | 140193 | 140193 | 0 | 5 | 4.74 | 12.22 | 0.9065 | 99.99% |
| `2026-06-14T21-50-10Z_arena_4` | 132407 | 131921 | 486 | 6292 | 4.74 | 14.59 | 0.9063 | 99.63% |
| `2026-06-14T22-33-50Z_arena_1` | 136467 | 136438 | 29 | 3194 | 4.41 | 14.65 | 0.9042 | 99.25% |
| `2026-06-14T22-33-50Z_arena_2` | 137904 | 137838 | 66 | 1247 | 4.49 | 11.72 | 0.9104 | 99.94% |
| `2026-06-14T22-33-50Z_arena_3` | 136777 | 136673 | 104 | 1922 | 4.61 | 11.49 | 0.9078 | 99.92% |
| `2026-06-14T22-33-50Z_arena_4` | 139432 | 139426 | 6 | 258 | 4.47 | 10.61 | 0.9082 | 100.00% |

Interpretation:

- Acquisition TensorRT boxes are generally very close to offline/refined boxes
  when both are present.
- Realtime-only frames are common and should not be treated as failures by
  themselves; many are boxes that offline refinement/quality gates do not accept.
- Offline-only frames are the important failure mode for crop-video reuse,
  because those are the offline-accepted frames where the crop recorder may have
  encoded a blank frame or cropped elsewhere.
- `2026-06-14T21-12-08Z_arena_2` needs focused review before claiming crop-video
  sufficiency for that session. Its p99 delta and crop containment are outliers.

## Proposed Scriptable Module

Extend `fisheye.diagnostics.compare_realtime_offline_detections` rather than
creating a new diagnostic family.

Recommended CLI:

```bash
scripts/py -m fisheye.diagnostics.compare_realtime_offline_detections \
  /path/to/recording/zarr/recording_analysis.zarr \
  --offline-source refined \
  --realtime-source crop-meta \
  --apply
```

## Acquisition Boxes as a Nonselector Detection Artifact

Implemented module:

```bash
scripts/py -m fisheye.utils.import_acquisition_detections_to_detect_run \
  /path/to/recording/zarr/recording_analysis.zarr \
  --run-name detect_acquisition_crop_meta_<label> \
  --artifact-only \
  --apply
```

This retains `derived/external_crop_recorder/*_crop_meta.csv` as explicit
unbound numeric evidence. It does not publish or select a standard raw detect
run:

```text
detection_artifact_runs/<run>/
  artifact_row_id
  frame_indices
  bbox_norm_coords
  bbox_img_xyxy
  scores
  class_ids
  frame_counts
  n_detections
```

It also writes acquisition-specific provenance arrays. These remain artifact
semantics and are not a downstream canonical detection contract:

```text
source_crop_xywh
source_crop_meta_row_indices
source_recording_frame_ids
```

The intended chain is:

```text
detection_artifact_runs/<acquisition_import>
  -> explicit canonical identity/coordinate binding and promotion (not yet implemented)
  -> new detect_runs/<canonical_import>
  -> detect_runs/<canonical_import>/quality_reports/<quality_run>
  -> refined_detect_runs/<runtime_refined>/instances
  -> crop/keypoint/mask consumers
```

Until that promotion boundary exists, the chain stops at the artifact. Missing
acquisition detections are represented in its exact full-domain proof and count
arrays as `frame_counts == 0`. Blank crop-recorder frames and no-detection
frames are preserved in run attrs, while row-level crop provenance remains
available through `source_crop_xywh` and `source_crop_meta_row_indices`.

This deliberately does not make keypoints or segmentation consume
`crop_meta.csv` directly. Future optimized model-input paths can read crop-video
pixels, but they should still key geometry and row identity off the selected
refined-detect surface.

Recommended source options:

```text
--realtime-source auto        # default; prefer crop-meta if manifest exposes it
--realtime-source crop-meta   # external_crop_recorder CSV
--realtime-source stimulus-h5 # analysis/stimulus_runs/.../tracking_data/bounding_boxes
--crop-meta PATH              # explicit override
--recording-dir PATH          # explicit override when zarr path cannot infer root
```

Recommended batch wrapper:

```bash
scripts/py -m fisheye.diagnostics.batch_compare_realtime_offline_detections \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains GoodCopBadCop \
  --offline-source refined \
  --realtime-source crop-meta \
  --apply
```

Batch output should include JSONL so cross-recording summaries can be reviewed
without reopening every zarr:

```text
recording_id
zarr_path
recording_dir
comparison_run_path
offline_source_path
realtime_source_path
offline_present_count
realtime_present_count
both_present_count
offline_only_count
realtime_only_count
centroid_delta_p50_px
centroid_delta_p99_px
bbox_iou_p50
offline_full_bbox_inside_crop_pct
blank_crop_rows_for_offline
crop_elsewhere_rows_for_offline
status
error
```

## Implementation Checklist

1. Add a crop-meta resolver.

   - Infer recording root from `.../zarr/<recording>_analysis.zarr`.
   - Prefer `recording_manifest.json` `video_streams.streams.crop.metadata`.
   - Fallback to exactly one
     `derived/external_crop_recorder/*_crop_meta.csv`.
   - Require an explicit `--crop-meta` when multiple candidates exist.

2. Add `load_crop_meta_realtime_detection_rows`.

   - Read `recording_frame_id`, `has_detection`, `blank_frame`,
     `detection_confidence`, `detection_x`, `detection_y`, `detection_w`, and
     `detection_h`.
   - Convert `recording_frame_id` to 0-based Palette frame indices.
   - Include rows only when `has_detection != 0` and `blank_frame == 0`.
   - Convert xywh to `bbox_img_xyxy`.
   - Use CSV row number as `row_indices`.
   - Return `source_kind="external_crop_recorder_crop_meta"`.

3. Keep H5 loading as a compatibility source.

   - Rename the current `load_realtime_detection_rows` path conceptually to
     `load_stimulus_h5_realtime_detection_rows`.
   - Preserve the current CLI behavior for existing callers, but make `auto`
     prefer crop-meta when available.

4. Add crop-geometry arrays.

   When the realtime source is crop-meta, persist:

   ```text
   realtime_crop_xywh
   realtime_has_detection
   realtime_blank_frame
   offline_center_inside_realtime_crop
   offline_bbox_inside_realtime_crop
   offline_crop_margin_left
   offline_crop_margin_top
   offline_crop_margin_right
   offline_crop_margin_bottom
   crop_sufficiency_reason_code
   ```

   Suggested reason labels:

   ```text
   0 = unassigned
   1 = offline_absent
   2 = inside_crop
   3 = missing_crop_row
   4 = blank_frame
   5 = no_realtime_detection
   6 = crop_elsewhere
   ```

5. Add summary fields.

   ```text
   crop_meta_row_count
   crop_meta_detection_row_count
   crop_meta_blank_row_count
   offline_center_inside_crop_count
   offline_center_inside_crop_pct
   offline_full_bbox_inside_crop_count
   offline_full_bbox_inside_crop_pct
   blank_crop_rows_for_offline_count
   no_detection_crop_rows_for_offline_count
   missing_crop_rows_for_offline_count
   crop_elsewhere_rows_for_offline_count
   ```

6. Update the PNG diagnostic.

   - Keep the existing centroid overlay, delta trace, presence barcode, and
     distribution panels.
   - Add crop-sufficiency text to the figure footer.
   - Optionally add a second PNG or extra panel showing crop-sufficiency reason
     barcode over time.

7. Add tests.

   - Unit-test crop-meta frame conversion from 1-based to 0-based.
   - Unit-test blank/no-detection filtering.
   - Unit-test manifest-based crop-meta resolution.
   - Unit-test fallback glob resolution.
   - Unit-test crop containment metrics and reason codes.
   - Keep the existing H5-based comparison tests passing.

8. Add a batch wrapper.

   The batch wrapper should be read-mostly and tolerate partial failures:

   - Resolve candidate zarrs from registry or explicit roots.
   - Run one comparison per zarr.
   - Write one JSONL status row per recording.
   - Optionally `--apply` zarr runs.
   - Support `--summary-jsonl`.
   - Do not make this a required pipeline step.

9. Add documentation.

   - Update `docs/orange_runtime_video_artifact_contract.md` to mention that
     crop-meta is the preferred realtime detection source for offline comparison.
   - Update `docs/recording_manifest_contract.md` with the same source priority.
   - Add operator examples for a single recording and registry batch run.

## Guardrails

- Do not promote acquisition boxes into the canonical refined detection surface
  by default. They are a comparison/provenance source unless a later explicit
  import workflow defines a separate accepted contract.
- Do not infer that realtime-only rows are bad. The important crop-video reuse
  failure mode is offline/refined-present frames where the acquisition crop is
  blank, missing, or elsewhere.
- Do not seek randomly inside the crop video for the first implementation. The
  crop metadata is the geometry source of truth; video-frame pixel validation can
  be a separate diagnostic because MP4 keyframe/index details can complicate
  random access.
- Persist numeric arrays as the source of truth. PNGs are QC snapshots only.
- Keep this optional. Recordings without external crop-recorder artifacts should
  still run the normal offline pipeline.

## Recommended Next Slice

Implement the crop-meta source in
`src/fisheye/diagnostics/compare_realtime_offline_detections.py`, add focused
unit tests, and run it on the single outlier recording:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_2_GoodCopBadCop
```

If the outlier is real, add a bad-example JSONL or table of the worst frames so
Crimson/web review can inspect whether the acquisition crop was blank, cropped
elsewhere, or the offline/refined model accepted a different object.
