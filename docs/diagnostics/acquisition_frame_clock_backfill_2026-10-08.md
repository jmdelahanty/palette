# Acquisition frame-clock backfill and single-authority fix (2026-10-08)

## What was wrong

The acquisition frame clock is published into each analysis Zarr at import
(`analysis/acquisition_frame_clock_runs`). The provider timing authority reads
only that publication. The validated-behavior frame-clock export (PR #178)
did not: `bind_validated_behavior_frame_clock` re-read raw clock files from the
recording folder, never compared them with the publication, and passed the
analysis Zarr path as the video path, so its file choice could differ from the
importer's.

Store census on 2026-10-08, before the backfill (264 analysis Zarrs):

| Clock status | Count |
|---|---|
| Published | 96 |
| Checked at import, no source | 12 |
| Never published | 156 |

Causes of "never published":

- `fisheye.utils.create_clipped_analysis_zarr` never published a clock.
  The four `2026_08_06_19_13_35_cam20100{93..96}` shells came from it.
- 148 older archives predate the importer's clock step (added 2026-07-26).
  This includes the four `sleepyfish_2026_05_05_17_45_30` cams.

## Backfill (live store, 2026-10-08)

Eight archives with a raw clock source were published using Palette's own
loaders and `publish_acquisition_frame_clock`. Each was then re-consolidated,
and each clock was read back through both direct and consolidated metadata.
No cluster jobs were touching these recordings.

| Archive | Source | Rows | Clock record sha256 (prefix) |
|---|---|---|---|
| 2026_08_06_19_13_35_cam2010093 | recording_frame_index.parquet | 2937604 | d176a5d8b1ec |
| 2026_08_06_19_13_35_cam2010094 | recording_frame_index.parquet | 2937604 | 68a27a471733 |
| 2026_08_06_19_13_35_cam2010095 | recording_frame_index.parquet | 2937604 | 68d4d1b7f27e |
| 2026_08_06_19_13_35_cam2010096 | recording_frame_index.parquet | 2937604 | 174ff656829e |
| sleepyfish_2026_05_05_17_45_30_cam2010093 | Cam…_meta.csv | 1188000 | a1a52d7e7ba6 |
| sleepyfish_2026_05_05_17_45_30_cam2010094 | Cam…_meta.csv | 1188000 | ee5765e89ffd |
| sleepyfish_2026_05_05_17_45_30_cam2010095 | Cam…_meta.csv | 1188000 | f46f8f8efef9 |
| sleepyfish_2026_05_05_17_45_30_cam2010096 | Cam…_meta.csv | 1188000 | cf418cf50e58 |

After the backfill: 104 published, 12 no source, 148 never published.

Verification:

- **August cams.** The published digests equal the
  `acquisition_frame_clock_source_sha256` sealed in the 2026-09-22 export
  (`sleepyfish_core_behavior_bout_kinematics_frame_clock_v1_20260922_v001`).
  Re-binding those four with the fixed code reproduces the export's sealed
  source bindings byte-for-byte, so the export needs no rework.
- **May cams.** The old export path would have bound
  `recording_frame_index.parquet`, while import binds the CSV. Timestamp values
  and semantics are identical in both files, so only provenance diverged.

## Code fix (same PR)

- The export reads timestamps only through
  `load_published_acquisition_frame_clock`. It refuses a recording that has no
  published clock.
- `create_clipped_analysis_zarr` publishes the clock through the same
  `publish_clipped_acquisition_frame_clock` helper the importer uses.
- The Parquet clock loader orders a camera's rows by `parent_frame_index`
  instead of by file order. Already-ordered files give identical bytes and
  digests.

## Still open

148 legacy archives have no published clock, and most have a CSV source. They
can be backfilled the same way; until then, the frame-clock export refuses them.
