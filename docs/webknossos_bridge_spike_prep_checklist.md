# webKnossos bridge spike prep checklist

<!-- contract-meta
status: proposed
created: 2026-07-05
owner: jeremy
audience: Palette maintainer + webKnossos administrator
related: docs/webknossos_palette_bridge_spike.md
-->

## Purpose

Use this worksheet before creating any webKnossos dataset from Palette data.

The goal is to prove whether webKnossos can consume a Palette-derived read-only
view without making an unmanaged copy of the training Zarr pixels.

## Non-negotiable invariant

Palette remains canonical.

webKnossos may hold:

- read-only image views,
- disposable compatibility caches,
- task metadata,
- annotation exports.

webKnossos must not become the canonical owner of:

- Palette training Zarr pixels,
- Palette row identity,
- run/revision state,
- component/keypoint/detection schema,
- approval state,
- or final training labels.

## Step 0 administrator questions

Send these to the webKnossos administrator before doing filesystem work.

```text
Hi,

I would like to run a small Palette/webKnossos bridge test using one 200-frame
canary dataset. The goal is to avoid copying canonical pixels if possible.

Questions:

1. Can the webKnossos Docker container mount this storage read-only?
   /groups/johnson/johnsonlab/jeremy

2. What is the instance's binaryData path for our organization/team?

3. Are symlinked datasets allowed under binaryData?

4. If symlinks are allowed, do targets need to remain inside Docker-mounted
   paths?

5. What webKnossos version is running?

6. Can we create a scratch dataset/team/project for RedScare canary testing?

7. Can I get or create an API token for Python-library testing?

8. What is the preferred local export path for annotations from this campus
   instance?

9. For volume annotations, do you usually export Zarr, WKW, or both?

10. For skeleton annotations, do you usually export NML, CSV, or both?

11. Are bounding-box annotations exported through the same task/annotation
    export path, or through a different API/UI mechanism?

12. Is there an existing recommended way on this campus instance to import
    OME-Zarr/Zarr data from a shared filesystem path rather than upload it?
```

## Canary source

Use this source for the first test unless a smaller scratch recording is
explicitly chosen.

```text
source_zarr_path=/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr
dataset_id=2026-06-23T16-01-09Z_arena_1:z92f469b75d66
recording_id=2026-06-23T16-01-09Z_arena_1
zarr_use=training
crop_run=crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare
source_array=crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare/roi_images
source_shape=(200,384,384)
```

## Candidate webKnossos view names

Use names that make the non-canonical status obvious.

```text
webknossos_dataset_name=palette_redscare_roi_stack_canary_readonly
view_name=redscare_roi_stack_z_y_x_symlink_view
view_axis_interpretation=z_y_x
webknossos_z_axis_semantics=palette_crop_row
canonical_owner=palette
view_role=read_only_adapter
```

## Bridge manifest template

Create this before import. Store it next to the generated view/cache.

```json
{
  "schema": "palette.webknossos_bridge_manifest.v1",
  "created_at_utc": "",
  "created_by": "",
  "canonical_owner": "palette",
  "view_role": "read_only_adapter",
  "copy_policy": "no_pixel_copy_expected",
  "dataset_id": "2026-06-23T16-01-09Z_arena_1:z92f469b75d66",
  "recording_id": "2026-06-23T16-01-09Z_arena_1",
  "zarr_use": "training",
  "source_zarr_path": "/groups/johnson/johnsonlab/jeremy/recordings/2026-06-23T16-01-09Z_arena_1_RedScare/zarr/2026-06-23T16-01-09Z_arena_1_RedScare_training.zarr",
  "source_crop_run": "crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare",
  "source_array": "crop_runs/crop_red_scare_acquisition_crop_video_training_2026-06-23T16-01-09Z_arena_1_RedScare/roi_images",
  "source_shape": [200, 384, 384],
  "source_dtype": "uint8",
  "webknossos_dataset_name": "palette_redscare_roi_stack_canary_readonly",
  "view_axis_interpretation": "z_y_x",
  "webknossos_z_axis_semantics": "palette_crop_row",
  "row_mapping_path": "row_mapping.csv",
  "annotation_import_policy": "webknossos_exports_must_import_through_palette_command",
  "palette_import_target_policy": "explicit_target_run_only"
}
```

## Row mapping CSV template

Create this before annotation work. The first row must map webKnossos slice/time
identity back to Palette row identity.

```csv
webknossos_z,palette_row,frame_index,source_crop_video_frame_index,source_crop_local_frame_id,source_crop_xywh,bbox_roi_xyxy
0,0,,,,,
1,1,,,,,
2,2,,,,,
```

Required columns:

```text
webknossos_z
palette_row
frame_index
source_crop_video_frame_index
source_crop_local_frame_id
```

Optional but useful columns:

```text
source_crop_xywh
bbox_roi_xyxy
source_recording_frame_id
```

## No-copy filesystem test checklist

Use this before any copied export.

```text
[ ] webKnossos container can read /groups read-only.
[ ] webKnossos binaryData path is known.
[ ] Scratch dataset/team/project exists.
[ ] Palette canary source path exists.
[ ] Bridge manifest exists.
[ ] Row mapping CSV exists.
[ ] OME-Zarr/webKnossos wrapper path is outside the canonical training Zarr.
[ ] Wrapper declares itself non-canonical/read-only.
[ ] Wrapper contains no copied pixel chunks.
[ ] Symlink target resolves from inside the webKnossos container.
[ ] webKnossos scan/import sees the dataset.
[ ] Dataset renders 200 slices/rows.
[ ] Slice 0 maps to Palette row 0.
[ ] Slice 199 maps to Palette row 199.
[ ] Intensity/range display is plausible.
[ ] No Palette Zarr arrays were modified.
```

## Tiny copied control checklist

Use this only if the no-copy wrapper fails.

```text
[ ] Failure from no-copy wrapper was recorded.
[ ] Copied control uses <=10 rows.
[ ] Copied control manifest says copy_policy=disposable_compatibility_control.
[ ] Copied control has source dataset_id/recording_id/source_array.
[ ] Copied control is stored outside canonical training Zarr.
[ ] Copied control imports/renders in webKnossos.
[ ] Result is used only to isolate compatibility, not as production design.
```

## First annotation test

Start with one mask component.

Recommended:

```text
component=subject_body
annotation_type=volume_segmentation
rows=0..9
```

Checklist:

```text
[ ] Create one scratch annotation/task.
[ ] Paint/edit a small subject_body mask on a few rows.
[ ] Export annotation through the campus-preferred path.
[ ] Record export path/ID in the manifest.
[ ] Read export with webKnossos Python library or documented parser.
[ ] Confirm exported voxel coordinates map back to Palette row/y/x.
[ ] Do not apply to Palette canonical run yet.
```

## Import readiness checklist

Only proceed to Palette import when all are true.

```text
[ ] Annotation export has stable ID/path.
[ ] Export format is documented.
[ ] Export row/slice coordinates map through row_mapping.csv.
[ ] Component identity is explicit.
[ ] Source dataset identity matches bridge manifest.
[ ] Import target Palette run is explicit.
[ ] Import will append audit/provenance events.
[ ] Import will leave review approval pending unless separately approved.
```

## Decision log template

Use this after the spike.

```text
date=
operator=
webknossos_instance=
webknossos_version=
palette_commit=
source_dataset_id=
source_recording_id=
source_zarr_path=
view_path_or_url=
view_strategy=symlink_wrapper|filesystem_dataset|remote_ome_zarr|copied_control
pixel_copy_performed=yes|no
imported_by_webknossos=yes|no
rendered_correctly=yes|no
annotation_type_tested=mask|bbox|keypoint
annotation_export_format=
roundtrip_to_palette_attempted=yes|no
roundtrip_to_palette_success=yes|no
blockers=
recommendation=continue|revise_adapter|do_not_adopt_for_this_surface
```

