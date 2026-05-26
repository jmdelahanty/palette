# Recording Store Relocation Components
<!-- contract-meta
status: active_design
last_verified: 2026-05-19
purpose: Define the migratory surfaces that must be inspected or rewritten when moving Palette recordings and Zarrs between storage roots.
-->

## Purpose

Palette recordings are moving from workstation-local storage such as
`/nvme1/recordings` toward durable cluster storage such as
`/groups/johnson/johnsonlab/jeremy/recordings` or another PRFS-backed root.

For simple consumers, relocation can look like changing one registry pointer.
For clipped recordings and training Zarrs, that is not sufficient: active path
references also live in Zarr attrs, Parquet sidecars, finalized-run manifests,
and review proxy artifacts.

This document defines the components that are migratory and the policy for
rewriting them.

## Path Classes

### Active Location Pointers

Active pointers define where current tools should read the recording, videos,
frame indexes, and Zarrs after relocation. These should be rewritten when a
store is promoted to a new canonical location.

Examples:

- registry `datasets.zarr_path`
- registry `datasets.source_recording_frame_index_path`
- Zarr root attrs such as `recording_path`
- Zarr root attr `source_recording_frame_index_path`
- sampled training `source_frame_index.parquet` video and metadata paths
- clipped analysis finalized-collection `source.video_path` entries
- review proxy manifests that point to source or proxy videos

### Historical Provenance

Historical provenance records how an artifact was originally produced. It
should not be silently rewritten during relocation. If a historical field points
to `/nvme1`, that may still be true.

Examples:

- `copied_detection_runs_from`
- `copy_existing_detections_from`
- `copy_analysis_metadata_from`
- original command-line arguments in manifests
- git/environment/provenance snapshots captured at artifact creation

When active paths are rewritten, add an explicit relocation note instead of
destroying historical provenance, for example:

```json
{
  "clipped_training_relocation": {
    "relocated_at_utc": "2026-05-20T01:27:03Z",
    "relocated_by": "fisheye.utils.relocate_recording_store",
    "from_recording_path": "/nvme1/recordings/example",
    "to_recording_path": "/groups/.../recordings/example",
    "historical_copied_detection_runs_from_preserved": "/nvme1/..."
  }
}
```

### Frozen Derived Artifacts

Merged/exported training datasets and trained model artifacts are immutable
derived products. They may contain source paths in manifests or `source_index`
arrays, but those paths describe the source set used at export time.

Do not rewrite old merged datasets just because source Zarrs move. Instead,
build a new exported dataset version from the migrated source registry state.

## Migratory Components

### Physical Recording Root

The physical recording directory should move as a unit when possible.

Expected first-class child artifacts:

- `cams/` for single-video recordings
- `clips/` for rolling-clipped recordings
- `zarr/`
- `recording_frame_index.parquet` for clipped recordings
- `recording_clip_index.json` / `.csv` for clipped recordings
- `recording_manifest.json` or source acquisition manifests when present
- derived review proxy artifacts under `derived/` when intended to remain valid

For single-video recordings, `cams/` remains first-class and should not be
treated as legacy.

### Registry Rows

The registry is the main discovery surface. A relocation must update the
canonical dataset row, not create a second active row for the same recording.

Fields to inspect or update in `datasets`:

- `dataset_id`
- `zarr_path`
- `path_hash`
- `status`
- `artifact_kind`
- `zarr_origin`
- `zarr_use`
- `source_layout`
- `source_frame_index_path`
- `source_recording_frame_index_path`
- `source_frame_index_schema`

`path_hash` is based on the resolved filesystem path. Moving a Zarr changes it.

Potential dependent tables:

- `recording_step_status`
- `recording_step_status_history`
- data profile tables
- quality/performance tables
- `dataset_lineage`
- `training_sets.dataset_ids_json`

Before deleting duplicate rows, search every registry table for the old
`dataset_id`. If the duplicate is not in a training set and only has stale
status rows, deleting the duplicate with foreign keys enabled is acceptable
after a registry backup.

### Recording Table

If the registry has a `recordings` row for the recording, its `recording_path`
should be updated to the canonical storage root as part of the same migration.

This is separate from `datasets.zarr_path`: one recording can have analysis,
training, and derived Zarrs.

### Zarr Root Attributes

Zarr root attrs carry active identity and source references used by tools that
open a Zarr directly without querying the registry.

Common active attrs to update:

- `recording_path`
- `source_recording_frame_index_path`
- `source_video_path` for single-video layouts, if present
- `source_video_paths` for inspection-only lists, if present
- `current_keypoint_group_path` only when copying latest keypoint state from a
  more complete peer Zarr

Common historical attrs to preserve:

- `copied_detection_runs_from`
- captured `command_line_args`
- creation-time `git_info`
- creation-time `platform_info`

### Clipped Recording Frame Index

`recording_frame_index.parquet` is the parent recording frame map. For a
relocated clipped recording, path columns should point to the new recording
root.

Columns to inspect or rewrite:

- `recording_folder`
- `video_path`
- `metadata_path`
- `keyframe_path`
- `clip_manifest_path`
- `clip_recording_folder`

The semantic frame keys should not change:

- `camera_serial`
- `recording_frame_id`
- `parent_frame_index`
- `clip_index`
- `clip_id`
- `clip_local_frame_index`

### Clipped Training Source Frame Index

`<training>.zarr/source_frame_index.parquet` is a sampled snapshot of the
recording frame index. It must remain row-aligned with
`raw_video/original_frame_indices`.

Columns to inspect or rewrite:

- `recording_folder`
- `video_path`
- `metadata_path`
- `keyframe_path`
- `clip_manifest_path`
- `clip_recording_folder`
- `source_recording_frame_index_path`

Columns that must not be semantically changed:

- `sample_index`
- `parent_frame_index`
- `recording_frame_id`
- `clip_index`
- `clip_id`
- `clip_local_frame_index`

Validation should confirm:

- no active path column still points to the old root;
- every active video/metadata/keyframe/manifest path exists;
- `raw_video/original_frame_indices == source_frame_index.parent_frame_index`;
- sampled row count equals `raw_video/images_ds.shape[0]`.

### Clipped Analysis Finalized Collections

Clipped analysis shells may store finalized collection manifests under paths
such as:

```text
experiment_index/finalized_runs/<collection_id>
```

Those manifests are active resolver surfaces for viewers and importers.

Fields to inspect:

- selected run `source.video_path`
- selected run `source.metadata_path`
- selected run `source.keyframe_path`
- `recording_clip_index`
- any run-dir or artifact paths under `derived/cluster_artifacts`

Internal Zarr group paths such as
`clips/clip_000017/cameras/2010093/refined_detect_runs/<run>` should remain
relative and usually do not require relocation edits.

### Stage Run Families

When replacing a source training Zarr with a copied peer, parent latest attrs
must move with the run families.

Important parent groups:

- `crop_runs`
- `detect_runs`
- `refined_detect_runs`
- `keypoints_runs`
- `refined_keypoints_runs`
- `subject_mask_runs`
- `refined_subject_masks_runs`
- `eye_masks_runs`
- `refined_eye_masks_runs`

For each copied family:

- copy the child run directories;
- copy the parent `zarr.json` attrs;
- verify `latest`, `latest_materialized`, `latest_any`, and review-status
  pointers still point at existing child groups.

### Sidecar Manifests

Sidecar manifests are often read by humans and tooling.

Examples:

- `*_clipped_training_manifest.json`
- `*_analysis.zarr_shell_manifest.json`
- `recording_frame_index_manifest.json`
- review proxy `manifest.json`
- cluster submission/finalization manifests under `derived/cluster_artifacts`

Rewrite active output/source pointers when the sidecar is meant to describe the
relocated artifact. Preserve historical command-line arguments unless the
manifest explicitly models a current command template.

### Review Proxy Artifacts

Review proxy videos are derived for UI performance. They are optional but
active when a web reviewer is launched with `--review-proxy-manifest`.

If proxy videos move with the recording:

- rewrite proxy manifest paths;
- verify every proxy video exists;
- verify proxy scale metadata still matches source-video dimensions;
- leave source detection coordinates unchanged.

If proxy videos are not copied, rebuild them from the relocated source videos
instead of preserving stale proxy manifests. On LSF clusters, use
`scripts/submit_review_proxy_videos_bsub.sh` for full-recording rebuilds so the
sequential transcode runs on a compute node rather than a login node.

## Recommended Relocation Workflow

1. Back up the registry.
2. Copy the physical recording root to the destination.
3. Compare source and destination Zarr metadata nodes.
4. Copy missing run families when parity with an existing peer is desired.
5. Rewrite active Zarr attrs and Parquet path columns.
6. Rewrite active sidecar manifest paths.
7. Preserve historical provenance and add an explicit relocation record.
8. Update the canonical registry row.
9. Remove duplicate registry rows only after checking every table for
   references.
10. Run validators and a task-specific dry run.

Minimum validation gates:

- path-column audit has zero old-root active paths;
- all active source paths exist;
- clipped-training provenance validator returns `status: ok`;
- registry query returns exactly one canonical row for the recording;
- task dry run selects the relocated Zarr, for example detection training
  preflight reports valid boxes.

## Sleepyfish Smoke Example

For
`sleepyfish_2026_05_05_17_45_30_cam2010093`, the `/groups` clipped training
Zarr was made parity-equivalent to the `/nvme1` clipped training Zarr by:

- copying missing `crop_runs`, `keypoints_runs`, and
  `refined_keypoints_runs`;
- preserving detection and refined-detection runs that already matched;
- rewriting active root attrs to `/groups`;
- rewriting sampled `source_frame_index.parquet` active path columns to
  `/groups`;
- rewriting active fields in `*_clipped_training_manifest.json`;
- preserving historical `copied_detection_runs_from=/nvme1/...`;
- updating the canonical registry dataset row to the `/groups` Zarr;
- pruning the duplicate suffixed registry row after confirming it was not in
  any training set.

Observed validation:

- source and destination Zarr metadata nodes: 373 vs 373;
- missing node count: 0;
- shape/type differences: 0;
- clipped-training provenance validator: `status: ok`;
- active sampled source paths: 0 old-root paths, 0 missing paths;
- detection training dry run: 238 refined boxes, 0 invalid boxes.

## Future Tooling

This process should become a dedicated relocation utility rather than ad hoc
SQL and one-off scripts.

Suggested command shape:

```bash
scripts/py -m fisheye.utils.relocate_recording_store \
  --source-root /nvme1/recordings/<recording_id> \
  --dest-root /groups/johnson/johnsonlab/jeremy/recordings/<recording_id> \
  --registry /nvme1/palette_registry.sqlite \
  --copy-missing-run-families \
  --repair-active-paths \
  --preserve-historical-provenance \
  --apply
```

The tool should expose a dry-run report that lists every active path rewrite,
every preserved historical field, every registry row update, and every
duplicate-row cleanup candidate before applying changes.
