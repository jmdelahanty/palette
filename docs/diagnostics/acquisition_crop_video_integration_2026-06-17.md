# Acquisition Crop Video Integration State
<!-- contract-meta
version: 1
status: current
last_verified: 2026-06-17
-->

## Summary

Orange external-IPC recordings can include two acquisition-time video streams:

- full-frame ingest-authoritative camera video under `cams/`
- runtime crop video under `derived/external_crop_recorder/`

Palette now treats those crop videos as first-class recording-level acquisition
media. They are not Palette `crop_runs`, and they are not automatically promoted
to detector output. Instead, import mirrors their manifest-declared availability
into the analysis zarr and registry scans expose that availability through
SQLite views.

## Current Surfaces

### Recording Manifest

`recording_manifest.json.video_streams` is the source manifest surface written
by the organizer/manifest refresh path. For external-IPC recordings it uses:

```text
video_streams.schema_id = orange_runtime_video_streams_v1
video_streams.frame_clock = recording_frame_id
video_streams.streams.full
video_streams.streams.crop
```

The crop stream declares the crop video path, crop metadata CSV, frame clock,
crop-pixel coordinate space, full-frame geometry coordinate space, blank-frame
policy, selection policy, dimensions, codec, and pixel-source metadata when
available.

### Analysis Zarr

Analysis-zarr import now mirrors manifest stream metadata into:

```text
analysis/acquisition_video_streams/
  streams/full/
  streams/crop/
```

Schema:

```text
palette.acquisition_video_streams.v1
```

The import/backfill only performs cheap checks:

- required file existence
- sidecar JSON readability
- CSV data row counts
- frame-count mismatch warnings

It does not decode crop video and does not verify pixel parity.

Root attrs also expose coarse availability:

```text
acquisition_video_streams_available
acquisition_video_streams_path
acquisition_video_stream_count
acquisition_crop_video_available
acquisition_video_stream_inventory_status
```

### Registry

Registry schema migration `056` adds:

```text
acquisition_video_streams
```

One row is stored per `(dataset_id, stream_key)`. `Registry.scan_zarr()` extracts
`analysis/acquisition_video_streams` and replaces rows for that dataset, so
rescans are idempotent and stale stream rows are removed.

Views:

```text
dataset_acquisition_video_streams_current
recording_acquisition_video_streams_current
recording_crop_video_available_current
```

The most useful query is:

```sql
SELECT recording_id, zarr_path, frame_count, metadata_row_count
FROM recording_crop_video_available_current
WHERE crop_stream_available = 1
  AND availability_status = 'ok';
```

## GoodCopBadCop Backfill

The zarr backfill command was:

```bash
scripts/py -m fisheye.utils.backfill_acquisition_video_stream_inventory \
  /groups/johnson/johnsonlab/jeremy/recordings \
  --recursive \
  --path-contains GoodCopBadCop \
  --apply \
  --output-jsonl /tmp/goodcopbadcop_acquisition_video_stream_backfill.jsonl
```

Result:

```text
applied: 12
missing: 0
skipped: 0
failed: 0
```

Each backfilled recording had `crop` and `full` streams with
`inventory_status=ok` and `crop_stream_available=true`.

The registry was then rescanned for the same 12 analysis zarrs after applying
schema migration `056`.

Registry verification:

```text
acquisition_video_streams rows for GoodCopBadCop: 24
recording_crop_video_available_current rows: 12
crop_stream_available rows: 12
```

The crop view reported matching `frame_count` and `metadata_row_count` for all
12 recordings:

```text
2026-06-14T21-12-08Z arenas 1-4: 140035 / 140035
2026-06-14T21-50-10Z arenas 1-4: 140198 / 140198
2026-06-14T22-33-50Z arenas 1-4: 139693 / 139693
```

## Current Boundaries

This integration deliberately does not do these things yet:

- It does not create `crop_runs/<run>` from acquisition crop videos.
- It does not import acquisition boxes into `detect_runs/<run>` automatically.
- It does not make keypoint or mask stages consume crop videos directly.
- It does not validate crop-video pixels against full-frame source crops.

Those are separate consumers of this recording-level media inventory.

## Future Recommendation

The next clean step is to make acquisition boxes optionally enter Palette as a
normal raw detection run:

```text
detect_runs/<acquisition_crop_meta_import>
  -> detect_quality
  -> refined_detect_runs/<runtime_refined>/instances
  -> downstream crop/keypoint/mask consumers
```

That keeps downstream consumers on existing detection/refined-detection
contracts while preserving the acquisition source in provenance.

After that, a crop-video-backed model-input path can be added for keypoints and
segmentation:

- use `recording_crop_video_available_current` to discover eligible recordings
- verify crop metadata row alignment to `recording_frame_id`
- define a crop-video pixel contract including decoder, color range, blank-frame
  value, coordinate transform, and source geometry
- add a parity diagnostic comparing decoded crop-video pixels to full-frame
  crops for sampled rows
- let model runners choose `crop_video` as an input source when parity and
  contract checks pass

This would let crop videos replace expensive persistent ROI caches for
compatible acquisition recordings without teaching downstream consumers a new
detection geometry contract.
