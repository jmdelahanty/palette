# Orange Rolling Clip Recording Contract
<!-- contract-meta
status: design
last_verified: 2026-05-16
purpose: Capture Orange's rolling-clip recording layout and the Palette import/indexing shape needed to process long recordings as clip-local units.
-->

## Purpose

Long Orange recordings should be handled as rolling clips rather than one giant
MP4 when possible. Palette's cluster and local workflows should process these
clips as independent video units, then build experiment-level indexes that
preserve a continuous recording frame clock.

This document records the Orange read-only audit results that drive Palette's
clipped-recording import design. Palette now has prototype utilities for
retroactive clip materialization, clip verification, recording-level frame
index construction, and creating a metadata-only clipped analysis-Zarr shell.
Model writers, run-group importers, finalized clip collections, and registry
projection support still need clip-aware implementations.

## Orange Producer Behavior

Orange already supports native headless rolling clips when
`fixed.recording_control.clip_seconds > 0`. Supervised headless `external_ipc`
rolling clips are also implemented. GUI rolling is still future work according
to the Orange audit.

Relevant Orange audit references:

- `docs/recording_session_manifest_contract.md:14`
- `src/orange_headless_client.cpp:8060`
- `src/session/recording_session.cpp:707`

Native headless rolling layout:

```text
<recording_folder>/
  recording_session.json
  recording_clip_index.json
  recording_clip_index.csv
  recording_snapshot.json
  ptp_sync_summary.json
  Cam<serial>_pipeline_perf.csv
  Cam<serial>_acquisition_cadence_probe.csv

  clips/clip_000000/
    clip_manifest.json
    Cam<serial>.mp4
    Cam<serial>_meta.csv
    Cam<serial>_keyframe.json
```

External IPC rolling uses the same `clips/clip_000000/` directory shape but
uses external artifact names:

```text
clips/clip_000000/
  Cam<serial>_external.mp4
  Cam<serial>_external_meta.csv
  Cam<serial>_external_keyframe.json
```

Native naming is created in Orange `src/shared_recording_output.cpp:24` and
keyframe sidecars are normalized in `src/FFmpegWriter.cpp:334`. External names
are created in `tools/external_recorder_ipc_probe.cpp:1417`.

## Clip Identity

Orange identifies clips by:

- `clip_index`: integer index.
- `clip_id`: formatted as `clip_%06d`.
- `clip_directory`: `clips/clip_%06d`.
- `recording_clip_index.json` / `.csv`: one row per clip-camera artifact.

Palette should preserve these identities rather than inventing a separate clip
namespace.

Recommended Palette clip table granularity:

```text
(recording_id, camera_serial, clip_index)
```

Core clip table fields:

- `recording_id`
- `session_id`
- `producer`
- `recording_backend_mode`
- `camera_serial`
- `clip_index`
- `clip_id`
- `clip_directory`
- `video_path`
- `metadata_path`
- `keyframe_path`
- `clip_manifest_path`
- `first_recording_frame_id`
- `last_recording_frame_id`
- `frame_count`
- `recording_frame_id_gaps`
- `packet_count`
- `packet_count_source`
- `rollover_at_recording_frame_id`
- `status`
- `start_reason`
- `stop_reason`
- `final_clip`

## Frame Mapping

For native Orange `Cam*_meta.csv`, the header is:

```text
frame_id,timestamp,timestamp_sys
```

Despite the column name, `frame_id` is the session-continuous
`recording_frame_id`. It is not clip-local and is not the camera/vendor frame
counter. The MP4 timeline is clip-local and starts at zero per clip.

Palette should derive:

- `clip_local_frame_index`: zero-based row number in the per-clip metadata CSV.
- `recording_frame_id`: native Orange metadata `frame_id`.
- `timestamp`: camera/PTP timestamp from metadata.
- `timestamp_sys`: host/system timestamp from metadata.

External IPC metadata is richer and may include:

- `recording_frame_id`
- `local_frame_id`
- `gop_index`
- `frame_index_within_gop`
- `timestamp`
- `timestamp_sys`

For native rolling clips, the authoritative local-to-global map is the per-clip
metadata CSV row order plus `frame_id`.

## Clip Boundary Semantics

Orange native rolling:

- preopens the next writer;
- switches at a GOP first-frame boundary;
- forces IDR/SPS/PPS on the first frame of the new clip;
- keeps `recording_frame_id` continuous;
- ends the previous clip at `rollover_at_recording_frame_id - 1`;
- starts the next clip at `rollover_at_recording_frame_id`.

The MP4 writer resets sample indexing per clip. Therefore
`Cam*_keyframe.json.total_frames` is clip-local for rolling clips.

For native multi-camera rolling, Orange uses one shared clip schedule and
shared clip folder names. Cameras are intended to roll on the same
`recording_frame_id` boundary, but Palette should still treat per-camera ranges
in `camera_artifacts` / `recording_clip_index` as authoritative.

## Ingest Authority

Palette should treat these as ingest-authoritative:

- `recording_session.json`: session identity, mode, producer, backend, clips.
- `recording_clip_index.json` / `.csv`: per `(clip, camera)` ranges and artifact
  paths.
- `clips/<clip_id>/clip_manifest.json`.
- per-clip `Cam*_meta.csv`: row order is clip-local frame index; `frame_id` is
  the session-continuous recording frame id.
- per-clip `Cam*.mp4` path and keyframe sidecar path from camera artifacts.

Palette should treat these as diagnostics/telemetry, not primary frame maps:

- `ptp_sync_summary.json`
- `Cam*_pipeline_perf.csv`
- `Cam*_acquisition_cadence_probe.csv`
- durations and FPS summaries
- packet counts

This is consistent with Orange documentation that PTP summary encoded counts
are not authoritative for encoded frame counts.

## Derived Parent Frame Index

Orange does not currently write a parent/session-level per-frame index for
rolling clips. The current Orange parent-level index is
`recording_clip_index.json` / `.csv`, and its granularity is one row per
clip-camera artifact.

Palette should add a derived convenience sidecar at the recording root:

```text
<recording_folder>/
  recording_frame_index.parquet
  recording_frame_index.csv
  recording_frame_index_manifest.json
```

`recording_frame_index.parquet` is the primary machine-readable artifact.
`recording_frame_index.csv` is optional and intended for inspection/debugging,
not high-volume production reads. `recording_frame_index_manifest.json` records
how the table was generated and which source artifacts were used.

This table is derived, not authoritative. The authoritative source remains:

- top-level `recording_clip_index.json` / `.csv`;
- per-clip `Cam*_meta.csv` row order and `frame_id`;
- per-clip clip manifests and camera artifact paths.

Diagnostics such as packet counts, ffprobe counts, PTP summaries, cadence
probes, and FPS estimates are validation/sanity signals only. They must not be
used as the primary local-to-global frame map.

Use Parquet for this sidecar because the frame index is a row-oriented table,
not an n-dimensional tensor. It has one row per camera frame and mixed column
types: ids, frame indices, timestamps, clip labels, and paths. Zarr attrs are
not appropriate for millions of rows, and Zarr arrays make string/path columns
and filtered table scans unnecessarily awkward. Zarr remains the analysis array
store; Parquet is the table/index sidecar. Palette may additionally publish the
compact numeric clock subset (`recording_frame_id`, `parent_frame_index`,
camera/PTP time, system time, and validity) as a digest-bound
`analysis/acquisition_frame_clock_runs` authority. That subset supports exact
array alignment and does not replace the Parquet table's clip and source-path
columns.

The two timestamps are not interchangeable. `timestamp_sys` is
producer-declared `clock_gettime(CLOCK_REALTIME)` in POSIX nanoseconds since the
UTC 1970 epoch, excluding leap seconds. `timestamp` is an unchanged
`Emergent::CEmergentFrame.timestamp` in the camera hardware clock domain. A
recording may classify the latter as inferred IEEE-1588/TAI only from combined
recording evidence: PTP synchronization configuration, valid PTP-offset
samples, camera-latch/frame agreement, and a stable camera-minus-host offset
near the applicable TAI-UTC difference. PTP enablement by itself is
insufficient. Without that evidence the camera clock retains an unspecified,
device-defined epoch. Neither field means time since Orange process, stream,
or recording start.

The analysis Zarr should point to the frame-index sidecar through small attrs or
a manifest, not duplicate the full table in root metadata. Recommended root or
manifest fields:

- `recording_frame_index_path`
- `recording_frame_index_schema`
- `recording_frame_index_manifest_path`
- `recording_frame_index_row_count`
- `recording_frame_index_source_authority`

Recommended row granularity:

```text
(camera_serial, recording_frame_id)
```

This row shape works for both current one-camera recordings and future
multi-camera recordings. Do not use one nested row per session frame with
camera sub-objects. Each camera artifact has its own authoritative range, so
the per-camera row is the safe unit.

Required columns:

- `session_id`
- `producer`
- `recording_folder`
- `camera_serial`
- `recording_frame_id`
- `parent_frame_index`
- `clip_index`
- `clip_id`
- `clip_local_frame_index`
- `timestamp`
- `timestamp_sys`
- `video_path`
- `metadata_path`
- `keyframe_path`
- `clip_manifest_path`
- `clip_directory`
- `clip_recording_folder`

Required derivations:

- `parent_frame_index = recording_frame_id - 1` for native Orange recordings
  whose recording frame IDs are one-based and continuous.
- `clip_local_frame_index` is the zero-based row number in the per-clip
  metadata CSV. After continuity checks pass, this should also equal
  `recording_frame_id - first_recording_frame_id` within that clip.

Recommended manifest fields:

- `schema_version`
- `generated_by`
- `generated_at_utc`
- `artifact_role`
- `source_authority`
- `recording_folder`
- `recording_clip_index_json`
- `recording_clip_index_csv`
- `row_count`
- `camera_serials`
- `recording_frame_id_min`
- `recording_frame_id_max`
- `checks`
- optional source file size, mtime, and content hash fields

Recommended manifest values:

```json
{
  "generated_by": "fisheye.utils.build_recording_frame_index",
  "artifact_role": "palette_derived_convenience_index",
  "source_authority": "recording_clip_index + per_clip_metadata_csv"
}
```

Training-Zarr import from clipped recordings should read
`recording_frame_index.parquet`, choose sample rows on the parent recording
clock, then materialize each frame from `(video_path, clip_local_frame_index)`.
For compatibility with current training readers, the import should write:

- `raw_video/original_frame_indices` as a compatibility alias for
  `parent_frame_index`;
- `raw_video/source_recording_frame_ids`;
- `raw_video/source_clip_index`;
- `raw_video/source_clip_local_frame_indices`;
- clip source metadata in attrs or a small sidecar group.

This keeps old frame-index consumers working while preserving the exact clip
source needed to reopen the correct MP4.

Important distinction:

- `recording_frame_index.parquet` is a recording-level sidecar and can exist
  regardless of whether an analysis Zarr or training Zarr has been created.
- `raw_video/original_frame_indices` is a sampled-Zarr-local array. In clipped
  training Zarrs it should point to `parent_frame_index`, so it maps imported
  samples back to the parent recording timeline.
- Clip lookup then comes from joining that parent frame to
  `recording_frame_index.parquet`, or from a sampled
  `source_frame_index.parquet` copied into the training Zarr.

Consumer-facing semantics are documented separately in
`docs/clipped_recording_consumer_mapping_contract.md`. In short: clipped
training Zarrs should keep stage `frame_indices` sample-local, while clipped
analysis run groups should keep clip-local run outputs under
`clips/<clip_id>/cameras/<camera_serial>/...` until a finalize stage exposes a
parent-level collection.

## Single-Video Recordings

The same table-vs-array boundary applies to current single-video recordings.
They do not need clip-local paths, but they still have a parent recording frame
clock and source-video metadata.

Current training imports often store `raw_video/original_frame_indices` as a
small Zarr array. That remains useful for sampled training archives because it
is compact and directly aligned to imported `raw_video` frames. It is not a
complete provenance table.

For full single-video analysis recordings, Palette may also create a
`recording_frame_index.parquet` sidecar with `clip_id` omitted or set to a
sentinel such as `full_video`, and `clip_local_frame_index == parent_frame_index`.
This gives single-video and clipped recordings the same query surface:

- sample a parent-frame range;
- map a parent frame to a source video and local frame index;
- preserve timestamps and camera ids;
- avoid putting large tabular metadata into Zarr attrs.

The compatibility rule is:

- `raw_video/original_frame_indices` is an array-level sampled-import mapping.
- `recording_frame_index.parquet` is a recording-level frame provenance table.

When both exist, `raw_video/original_frame_indices` should reference
`parent_frame_index` values from the recording frame index.

## Editing And Stale-State Boundaries

The recording frame index must not become a mutable review or workflow ledger.
It is an immutable or regenerable source map. It answers:

```text
Given (camera_serial, recording_frame_id), which source video and local frame
should be decoded?
```

It should not store:

- edited detections;
- manual additions/deletions;
- review approvals;
- downstream stale flags;
- keypoint or mask quality;
- stage completion status;
- latest run choices.

Those mutable states remain in Zarr run groups and registry/finalize
projections:

- Refined detection edits live in `refined_detect_runs/...`.
- Crop/keypoint/mask outputs live in their stage run groups.
- Review status lives on the reviewed run group or its review-status sidecar.
- Stale status is derived by comparing recorded upstream run names,
  fingerprints, and stable row identities.
- Registry rows and `experiment_index` tables are rebuildable projections from
  canonical Zarr state.

Stable identity should use explicit frame and row ids:

- Frame identity: `(camera_serial, recording_frame_id)` or
  `parent_frame_index`.
- Clip location: `(clip_id, clip_local_frame_index)` from the frame index.
- Refined detection identity: `source_refined_row_ids` /
  `refined_row_ids` inside the refined-detection run.
- Downstream lineage: source run path, source fingerprint, `clip_id`, and
  source refined row ids when available.

If an operator edits detections in
`clips/clip_000017/cameras/2010093/refined_detect_runs/...`, the frame index
does not change. Downstream keypoint or mask runs become stale only if their
recorded upstream refined-detection fingerprint or row lineage no longer
matches. A finalize/status pass should report that staleness at clip, camera,
stage, and workflow level.

This separation keeps the clipped layout manageable:

- Parquet frame index: source-location map.
- Zarr run groups: mutable scientific/editing state.
- Finalize/registry views: derived status and query surfaces.

## Recommended Analysis Zarr Shape

Represent clips as children of one recording/session, not as independent
recording rows. The recommended physical layout is one parent analysis Zarr per
recording, with clip-local run groups inside that parent store.

Clip-local groups are the primary compute and import target. A clip job can
own `clips/<clip_id>/cameras/<camera_serial>/<run_family>/<run_name>` without
coordinating row offsets or physical chunks with jobs processing other clips.
The parent recording clock is preserved by `recording_frame_id`,
`parent_frame_index`, and the derived `recording_frame_index.parquet` sidecar.

Preferred future shape:

```text
experiment.zarr/
  recording_frame_index.parquet
  recording_frame_index_manifest.json

  clips/
    clip_000000/
      cameras/
        2010093/
          source/
            clip_manifest
            video_path
            metadata_path
            keyframe_path
            frame_map/
              clip_local_frame_index
              recording_frame_id
              parent_frame_index
              timestamp
              timestamp_sys
          detect_runs/
          refined_detect_runs/
          crop_runs/
          keypoints_runs/
          refined_keypoints_runs/
          subject_mask_runs/
          refined_subject_masks_runs/
    clip_000001/
      ...
  experiment_index/
    clip_table
    workflow_manifests/
    finalized_runs/
```

Readers should treat an experiment as a logical concatenation of clip-local
outputs using explicit fields:

- `clip_id`
- `clip_local_frame_index`
- `recording_frame_id`
- optional generated `global_frame_index` alias when it is equal to
  `recording_frame_id`
- source video path or source clip id

Do not make each clip a separate `recording_id` in the registry. A clip is a
child artifact of one recording/session. Registry projections may add clip
tables or views, but the biological/experimental recording identity remains the
parent recording.

Do not make one giant global array the primary active-write target for long
recordings. Global arrays can be generated later as exports or compatibility
views, but broad cluster compute should write disjoint clip namespaces first.

## Clip-Local Processing And Finalize Stage

The preferred production workflow is:

```text
recording_clip_index + per-clip MP4s
  -> create parent analysis Zarr shell and recording_frame_index
  -> submit independent clip-local jobs
  -> import validated clip-local run groups
  -> run a serialized finalize stage
  -> refresh registry/query projections
```

During active compute, each job should write only complete clip-local run
groups. Examples:

```text
clips/clip_000017/cameras/2010093/detect_runs/detect_...
clips/clip_000017/cameras/2010093/refined_detect_runs/refined_detect_...
clips/clip_000017/cameras/2010093/crop_runs/crop_...
```

The finalize stage is the only owner of shared experiment-level metadata. It
should run after the relevant clip-local imports complete and should:

- verify every expected clip-camera artifact has the required stage output;
- verify per-clip outputs reference the expected upstream run names and source
  fingerprints;
- build or refresh `experiment_index/clip_table`;
- build or refresh `experiment_index/workflow_manifests/<workflow_id>`;
- choose logical latest run collections for stages only after coverage checks
  pass;
- refresh consolidated metadata when policy requires it;
- update registry projections from the canonical Zarr state.

The finalize stage should not recompute expensive per-frame model outputs. It
should operate on compact manifests, frame indexes, and already-imported
clip-local run groups.

For image-local stages, clip-local processing is the natural execution unit:

- detection;
- detect quality;
- refined detection when it only filters/curates frame-local detections;
- crop geometry and temporary ROI cache materialization;
- keypoint inference;
- subject-mask or eye-mask inference.

For temporal stages, clip boundaries are a real algorithmic concern:

- track kinematics;
- bout detection;
- bout kinematics when it depends on bout windows crossing clip edges;
- smoothing, hysteresis, derivatives, and temporal state machines.

Those stages should either run in the finalize phase over compact parent-level
inputs, or use explicit overlap/state handoff between clips. Do not silently
run them per clip and concatenate results unless the boundary policy is part of
the run provenance.

Compatibility with existing readers should be layered on top of the clip-local
layout. A finalized workflow may expose a manifest such as:

```text
experiment_index/finalized_runs/<workflow_id>/
  detect_run_collection
  refined_detect_run_collection
  crop_run_collection
  keypoint_run_collection
  subject_mask_run_collection
```

Each collection maps `(clip_id, camera_serial)` to the concrete clip-local run
path. Older reader compatibility can then be implemented by a resolver that
understands either a traditional top-level run group or a finalized clip
collection. Materializing global concatenated arrays should remain an explicit
export/compatibility step, not the default write path.

## Cluster Processing Implications

Clip-local processing is the preferred scaling unit for future long recordings.
Detection, crop, pose, segmentation, and refinement jobs should process one
clip or a small independent clip set and write clip-local run groups.

Avoid appending all clips into one giant run group during active compute. That
requires global row allocation and chunk-safe concurrent writes. A later
serialized finalize step can build experiment-level indexes, query views,
`latest` pointers, consolidated metadata, and registry projections.

This layout should also avoid repeated random seeking into 11-hour MP4s:
Palette can decode each short clip sequentially and preserve continuity through
the `recording_frame_id` map.

## Implementation Gaps

Palette implements only the setup pieces of this contract. Remaining slices:

- Organizer/import support for Orange rolling folder layout.
- Manifest/backfill schema for `recording_clip_index.{json,csv}`.
- Registry tables or projections for clip children of a recording.
- Analysis/training Zarr writers that can write clip-local run groups.
- Run-group artifact/import support beyond the current detection artifact
  slice for `clips/<clip_id>/cameras/<camera_serial>/<family>/<run>`.
- Finalize-stage utility for clip coverage checks, collection manifests,
  logical latest aliases, consolidated metadata, and registry projection.
- Readers that can resolve finalized clip collections by `recording_frame_id`.
- Cluster planners that emit `(recording_id, camera_serial, clip_id, stage)`
  work items.
- Validation that per-clip metadata row count, keyframe `total_frames`, and
  decodable MP4 frames agree within the per-clip artifact.

## Current Code Audit Notes

Read-only audit on 2026-05-16 found these expected stale assumptions:

- `src/fisheye/shared/zarr/schema.py` still describes top-level run families
  for traditional single-video archives. Clipped shell creation currently lives
  in `fisheye.utils.create_clipped_analysis_zarr`.
- Current LSF batch submitters discover whole `*_analysis.zarr` archives, not
  `(recording_id, camera_serial, clip_id, stage)` work items.
- Most crop, keypoint, and mask writers still write top-level groups such as
  `crop_runs/<run>` and update parent `latest` attrs directly.
- `fisheye.utils.import_run_group_artifact` currently supports the detection
  artifact slice. For rolling clips, pass `--use-intended-target` to import to
  `clips/<clip_id>/cameras/<camera_serial>/detect_runs/<run_name>`.
- `fisheye.refinement.detect_quality` and
  `fisheye.refinement.refine_detect` accept explicit detect/refined family
  paths for clip-local quality/refinement smoke runs.
- Existing reader/status tools generally resolve `latest` from top-level run
  families and need a shared resolver for finalized clip collections.
- Temporal analysis tools consume parent-level latest keypoint/detection
  surfaces today. They need explicit clip-boundary policy before clipped
  execution is safe.
- Training-Zarr creation from clipped recordings is not implemented; it should
  use `recording_frame_index.parquet` to map parent sample frames to
  `(video_path, clip_local_frame_index)`.

These notes are not defects in current single-video workflows. They define the
compatibility layer needed before clipped analysis archives can become a normal
production path.

## Frame Index Builder Utility

Palette has an initial builder for the recording-level frame index sidecar:

```bash
scripts/py -m fisheye.utils.build_recording_frame_index \
  /path/to/recording_folder
```

The builder supports two layouts:

- rolling clips: reads `recording_clip_index.json` plus per-clip
  `Cam*_meta.csv`;
- single-video camera layout: reads `cams/Cam*_meta.csv` and
  `cams/Cam*.mp4` when no clip index is present.

The single-video `cams/` layout is not legacy. It is still a first-class
recording representation for short or moderate recordings where one MP4 per
camera remains operationally simple. The rolling `clips/` layout is for long
recordings or workflows that benefit from clip-parallel processing and smaller
video units.

Default outputs:

- `recording_frame_index.parquet`;
- `recording_frame_index_manifest.json`.

Optional CSV inspection output:

```bash
scripts/py -m fisheye.utils.build_recording_frame_index \
  /path/to/recording_folder \
  --write-csv
```

The command refuses to overwrite existing sidecars unless `--overwrite` is
provided. Use `--dry-run --json` to inspect table shape and validation checks
without writing outputs.

Smoke result from 2026-05-16:

```text
recording: /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093
source_layout: rolling_clips
rows: 1188000
camera_serials: ['2010093']
recording_frame_id_range: 1..1188000
first mapping: clip_000000 local 0 -> parent_frame_index 0
last mapping: clip_000021 local 53999 -> parent_frame_index 1187999
manifest_status: ok
failure_count: 0
parquet_size: 22M
```

## Analysis-Zarr Shell Creator Utility

Palette has an initial metadata-only shell creator for clipped recordings:

```bash
scripts/py -m fisheye.utils.create_clipped_analysis_zarr \
  /path/to/recording_folder \
  --output-zarr /path/to/<recording>_analysis.zarr
```

The shell creator requires:

- top-level `recording_clip_index.json`;
- top-level `recording_frame_index_manifest.json`;
- the manifest-referenced `recording_frame_index.parquet`.

It writes structure and provenance only. It does not run detection, import model
outputs, set stage `latest` aliases to real runs, update a registry, or write
finalized workflow collections.

Current shell layout:

```text
<recording>_analysis.zarr/
  raw_video/                  # attrs declare external_clips storage
  detect_runs/                # parent finalized/aggregated placeholder
  refined_detect_runs/
  crop_runs/
  keypoints_runs/
  refined_keypoints_runs/
  eye_masks_runs/
  refined_eye_masks_runs/
  subject_mask_runs/
  refined_subject_masks_runs/
  clips/
    clip_000000/
      cameras/
        2010093/
          source/
            frame_map/
          detect_runs/
          refined_detect_runs/
          crop_runs/
          keypoints_runs/
          refined_keypoints_runs/
          eye_masks_runs/
          refined_eye_masks_runs/
          subject_mask_runs/
          refined_subject_masks_runs/
  experiment_index/
    clip_table/
    workflow_manifests/
    finalized_runs/
```

The camera hierarchy is intentional even for current one-camera recordings. It
preserves the Orange clip-camera artifact granularity and avoids a future schema
break if multi-camera rolling recordings are introduced.

Dish-mask geometry is recording/camera scoped. Orange acquisition guarantees
that dish locations and camera geometry do not move within a recording, so the
analysis shell records `dish_mask_scope="recording_camera"` and
`dish_mask_clip_policy="single_camera_mask_applies_to_all_clips"` in
`analysis_metadata.attrs`. Downstream clipped training and analysis workflows
should derive or copy one dish mask per `(recording_id, camera_serial)` and
reuse it across all clips for that camera.

For single-camera shells, `fisheye.utils.create_clipped_analysis_zarr`
automatically discovers an unambiguous sibling training Zarr and copies
`analysis_metadata.attrs["dish_mask"]` plus root
`attrs["experiment_setup"]`, recording the source Zarr and copy timestamp.
Production creation should pass `--require-dish-mask` so a missing or
conflicting geometry fails before the shell is written. Use
`--copy-analysis-metadata-from` when discovery is ambiguous; multi-camera
shells require that explicit choice until a camera-keyed mask contract exists.

Smoke result from 2026-05-16:

```text
recording: /nvme1/recordings/sleepyfish_2026_05_05_17_45_30_cam2010093
output_zarr: /tmp/palette_clipped_analysis_shell_smoke/sleepyfish_2026_05_05_17_45_30_cam2010093_analysis.zarr
source_layout: rolling_clips
clip_count: 22
clip_camera_row_count: 22
camera_serials: ['2010093']
recording_frame_index_row_count: 1188000
recording_frame_id_range: 1..1188000
```

## Planning Utility

Palette has an initial dry-run planner for retroactively designing
Orange-style clip boundaries from a long camera MP4:

```bash
scripts/py -m fisheye.utils.plan_orange_style_clips \
  /path/to/Cam2010093_recording.mp4 \
  --target-duration-minutes 30 \
  --snap-direction next \
  --output-dir /tmp/palette_clip_plan \
  --summary
```

The planner is read-only with respect to the source recording. It expects the
camera stream bundle next to the video unless explicit paths are supplied:

- `Cam*.mp4`
- `Cam*_meta.csv`
- `Cam*_keyframe.json`

It writes plan artifacts only:

- `recording_clip_index.json`
- `recording_clip_index.csv`

The planner uses `Cam*_keyframe.json` instead of probing every video frame, so
it avoids a full `ffprobe` scan of long MP4s. It snaps target boundaries to
safe keyframe starts, maps each proposed clip to metadata rows, and verifies:

- metadata row count equals keyframe `total_frames`;
- `recording_frame_id` is continuous globally;
- each clip start is a keyframe;
- each clip-local metadata slice has continuous `recording_frame_id`;
- each clip row records both clip-local frame range and session-continuous
  `recording_frame_id` range.

## Materialization Utility

Palette also has an explicit apply-mode materializer for retroactively writing
the planned clip layout:

```bash
scripts/py -m fisheye.utils.materialize_orange_style_clips \
  /path/to/Cam2010093_recording.mp4 \
  --target-duration-minutes 30 \
  --snap-direction next
```

The command is dry-run by default. It prints the stream-copy commands and target
paths without writing files. Pass `--apply` only after reviewing the plan:

```bash
scripts/py -m fisheye.utils.materialize_orange_style_clips \
  /path/to/Cam2010093_recording.mp4 \
  --target-duration-minutes 30 \
  --snap-direction next \
  --apply
```

By default, the output recording directory is inferred as the parent of the
`cams/` directory. Existing `cams/` source videos are left untouched. The
materializer writes:

- `clips/clip_000000/<source-video-name>.mp4`
- `clips/clip_000000/<source-video-stem>_meta.csv`
- `clips/clip_000000/<source-video-stem>_keyframe.json`
- `clips/clip_000000/clip_manifest.json`
- top-level `recording_clip_index.json`
- top-level `recording_clip_index.csv`

The materializer refuses to overwrite existing artifacts unless `--overwrite`
is provided. It writes the root clip index last so a failed clip command does
not advertise a partially materialized recording as complete.

Current implementation limits:

- It materializes one camera stream at a time.
- It does not yet update registry tables.
- It uses keyframe-aligned `ffmpeg -c:v copy -frames:v <clip_frame_count>` and
  assumes the source `Cam*_keyframe.json` accurately describes safe clip starts.

## Verification Utility

After materialization, verify the resulting clipped recording before downstream
import or processing:

```bash
scripts/py -m fisheye.utils.verify_orange_style_clips \
  /path/to/recording_folder
```

The default verifier performs cheap structural checks:

- top-level `recording_clip_index.json` exists;
- each clip video, metadata CSV, keyframe JSON, and `clip_manifest.json` exists;
- per-clip metadata row count equals `recording_clip_index` `frame_count`;
- per-clip metadata first/last `frame_id` match the indexed
  `recording_frame_id` range;
- per-clip metadata frame IDs are continuous;
- per-clip keyframe `total_frames` equals `frame_count`;
- per-clip keyframe list starts at clip-local frame `0` and stays within clip
  bounds;
- per-clip manifest `clip_id` matches the index row.

For a stronger post-split gate, add `--probe-video`:

```bash
scripts/py -m fisheye.utils.verify_orange_style_clips \
  /path/to/recording_folder \
  --probe-video
```

This uses `ffprobe -count_packets` for each clip video and verifies the encoded
packet count against the clip index `frame_count`. This can scan each clip, so
it is intentionally opt-in. While probing, the CLI prints per-clip start/done
progress to stderr. Use `--no-progress` when machine-readable stdout/stderr is
more important than operator feedback.
