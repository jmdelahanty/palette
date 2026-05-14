# Sampled Import

This guide covers importing video frames into Zarr archives for training data.
A sampled import grabs every Nth frame from a video, producing a compact Zarr
that's practical to work with for labeling and training without importing
hundreds of thousands of frames.

## When to use sampled imports

- You want to **build training datasets** (detection, pose, or segmentation)
  from your camera videos.
- You need labeled frames for review and correction but don't need every
  single frame.
- You want a **compact Zarr** that's fast to read during training.

If you're running the full analysis pipeline on a recording (detection through
kinematics), you don't need a sampled import — the
[pipeline workflow](pipeline_workflow.md) handles that with its own import
step.

## Single video import

```bash
scripts/py -m fisheye.capture.import_video /path/to/Cam2010093.mp4 \
  --training-data \
  --frame-step 100 \
  --config configs/fisheye/import_local.yaml \
  --zarr-path /path/to/output/training_sample.zarr
```

This imports every 100th frame (frames 0, 100, 200, ...) from the video.

### Required flags

Both `--training-data` and `--frame-step` must be used together — the script
will error if you use one without the other. This is intentional: you can't
accidentally create a truncated import by forgetting a flag.

### Arguments

| Argument | Description |
|----------|-------------|
| `video_path` | Source video file (positional, required) |
| `--training-data` | Enable sampled import mode |
| `--frame-step N` | Import every Nth frame (e.g. 100 = frames 0, 100, 200, ...) |
| `--config PATH` | Import config YAML (controls resolution, format, chunking) |
| `--zarr-path PATH` | Output Zarr path (default: `<video_stem>.zarr` next to the video) |
| `--skip-tail-frames N` | Skip the last N frames (default: 0). See [below](#tail-frame-issues) |
| `--overwrite` | Delete existing Zarr before importing |
| `--cpu-only` | Force CPU decoding (default uses GPU) |

### Choosing a frame step

The right value depends on your video length and how much data you need:

| Video length | `--frame-step` | Approximate frames imported |
|-------------|----------------|---------------------------|
| 30 min @ 30fps | 100 | ~540 |
| 30 min @ 30fps | 50 | ~1,080 |
| 1 hr @ 30fps | 100 | ~1,080 |
| 1 hr @ 30fps | 200 | ~540 |

For initial training sets, `--frame-step 100` is a reasonable starting point.
You can always import again with a smaller step if you need more frames.

---

## Recording-only / video-only imports

Use `fisheye.utils.intake_video_only_recording` when the recording directory
was organized from camera videos only and has no H5/protocol source. This
wrapper calls the sampled import path, then stamps the Zarr with recording
metadata such as `recording_id`, `dish_design`, `camera_id`, and
`experiment_context_status = "absent"`.

Example:

```bash
scripts/py -m fisheye.utils.intake_video_only_recording \
  /nvme1/recordings/<recording>/cams/Cam2010093_<recording>.mp4 \
  --recording-dir /nvme1/recordings/<recording> \
  --zarr-path /nvme1/recordings/<recording>/zarr/<recording>_training.zarr \
  --frame-step 5000 \
  --skip-tail-frames 0 \
  --session-uuid <recording> \
  --recording-id <recording> \
  --recording-name <recording> \
  --protocol-name sleepyfish \
  --dish-design palm \
  --camera-id 2010093 \
  --num-dishes 1 \
  --fish-per-dish 1 \
  --register \
  --registry /nvme1/palette_registry.sqlite
```

Run the same command with `--dry-run` first if you want to inspect the plan
without writing the Zarr or registry row.

If the organized recording already has `recording_manifest.json`, do not pass
`--write-manifest` during training-Zarr intake. The organizer-owned recording
manifest is the richer source of truth. Use `--write-manifest` only when you
are intentionally creating a minimal manifest for a standalone video-only
recording.

For very long recordings, choose `--frame-step` from a target sample count
rather than reusing `100` blindly:

```text
frame_step ~= source_frame_count / target_sample_count
```

For example, an 11-hour 30 fps video has about 1,188,000 frames. A target of
roughly 240 sampled frames gives `frame_step ~= 5000`. This keeps the training
Zarr comparable in size to short-recording sampled imports.

### Video-only validation checks

After import, check:

- root attrs include `zarr_purpose = "training"`,
  `experiment_context_status = "absent"`, and
  `stimulus_runs_available = false`;
- `raw_video` attrs include the expected `frame_step` and sampled
  `total_frames`;
- `raw_video/original_frame_indices` exists when `frame_step > 1`;
- strict JSON parsing of all `zarr.json` metadata files succeeds;
- the registry has one `datasets` row with `zarr_use = "training"` if
  `--register` was used.

For detection-label seeding, compare the trained model input dimensions against
the arrays stored in `raw_video`. Model dimensions are queryable from the
registry through `model_input_shapes`. If a sampled Zarr already has a
downsampled frame array matching the model size, use that representation for
initial detection seeding. The helper command is:

```bash
scripts/py -m fisheye.utils.predict_training_detections /path/to/training.zarr \
  --registry /nvme1/palette_registry.sqlite \
  --model-run-id <registered_detect_run_id> \
  --run-name detect_seed_<model_or_date> \
  --apply
```

Then initialize the curated refined-detect surface for review:

```bash
scripts/py -m fisheye.refinement.refine_detect /path/to/training.zarr \
  --detect-run detect_seed_<model_or_date> \
  --per-frame-top-k 1
```

For sampled training imports, `refine_detect` runs in passthrough mode: it does
not require `detect_quality`, disables artifact filters, and writes
`refined_detect_runs/<run>/instances` for manual review/approval. The normal
refinement dish-mask gate still applies when
`analysis_metadata.attrs["dish_mask"]` is present: raw out-of-dish candidates
remain in `source_detections`, but they are filtered from `instances` with
reason `outside_dish_mask`. Use `--per-frame-top-k 1` for one-fish-per-frame
training Zarrs when you want the highest-confidence in-dish seed detection as
the initial reviewed candidate while preserving lower-confidence raw candidates
as `source_detections` rows.

The GPU import path may print a warning that a buffer does not support
`__cuda_array_interface__` and is falling back to a slower copy path. Treat
that as a performance warning, not a failed import, unless the process exits
non-zero. On systems with kvikIO/GDS enabled, provenance should still record
`import_method = "kvikio_zarr"` and `gds_enabled = true` when the core write
path used GDS.

---

## Batch import across recordings

If you have organized recordings (from the
[organize step](organize_recordings.md)), you can import sampled training
Zarrs in batch:

### Dry-run first

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --dry-run
```

This shows each recording it found, the camera video it matched, and where the
output Zarr would go. Add `--rich` for a formatted table view.

### Apply

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --apply
```

### How batch discovery works

The batch importer looks for the organized recording layout:

```
recordings/<session>/
  raw/<session>.h5        # reads metadata (camera ID, session UUID)
  cams/Cam<id>.mp4        # the video to import
  zarr/                   # output goes here
    <h5_stem>.zarr        # created by the importer
```

It extracts the camera ID from the H5 metadata and matches it to a video in
`cams/`. If there's only one MP4 in `cams/`, it uses that regardless of camera
ID.

For video-only recordings without H5/protocol metadata, use
`intake_video_only_recording` directly as shown above, or add a dedicated batch
wrapper only after deciding the metadata policy for that batch.

### Batch-specific options

| Argument | Default | Description |
|----------|---------|-------------|
| `--frame-step N` | 100 | Frame sampling interval |
| `--skip-tail-frames N` | 200 | Skip last N frames (higher default than single-video) |
| `--config PATH` | `configs/fisheye/import_local.yaml` | Import config |
| `--recursive` | off | Search recordings root recursively |
| `--overwrite` | off | Re-import over existing Zarrs |
| `--no-skip-existing` | off | Attempt import even if Zarr exists |
| `--import-stimulus` | off | Also import stimulus events from the H5 |
| `--stimulus-always` | off | Re-import stimulus even if already present |
| `--no-log` | off | Disable JSONL logging |

### Importing stimulus data

If your recordings have stimulus protocols (moving gratings, looming dots,
etc.) and you want stimulus event tables available for analysis, add
`--import-stimulus`:

```bash
scripts/py -m fisheye.utils.import_recordings_training /nvme1/recordings \
  --recursive \
  --frame-step 100 \
  --import-stimulus \
  --apply
```

This reads stimulus events from the H5 and writes them to
`analysis/stimulus_runs/` in the Zarr. By default it skips recordings that
already have stimulus data — use `--stimulus-always` to re-import.

---

## Import configuration

The config YAML controls what resolutions and formats are written. Two configs
ship with the repo:

### `configs/fisheye/import_local.yaml` (default for batch)

```yaml
import:
  resolutions: both          # full + downsampled
  full: true
  downsampled:
    size: [768, 1280]        # height x width
    method: area
    preserve_aspect: true
    formats: [rgb]
    chunk_size: 64
```

Produces full-resolution grayscale frames and 768x1280 RGB downsampled frames.

### `configs/fisheye/import_nfs.yaml`

```yaml
import:
  resolutions: both
  full: true
  downsampled:
    size: [640, 640]
    method: area
    formats: [gray]
    chunk_size: 16
```

Produces full-resolution frames and 640x640 grayscale downsampled frames. Uses
sharding for better NFS performance.

### Key config options

| Option | Values | Description |
|--------|--------|-------------|
| `resolutions` | `both`, `full`, `downsampled` | Which resolution sets to write |
| `downsampled.size` | `[H, W]` | Target downsampled dimensions |
| `downsampled.method` | `area`, `bilinear`, `cubic`, `lanczos4` | Resize interpolation method |
| `downsampled.formats` | `[gray]`, `[rgb]`, `[gray, rgb]` | Output color formats |
| `downsampled.preserve_aspect` | `true`/`false` | Maintain aspect ratio when resizing |
| `use_sharding` | `true`/`false` | Enable Zarr sharding (better for NFS) |
| `chunk_size` | int | Frames per Zarr chunk |
| `gpu_fp16` | `true`/`false` | Use FP16 for GPU resize (faster) |

---

## What the output Zarr contains

After a sampled import with `--frame-step 100`, the Zarr looks like:

```
training_sample.zarr/
  raw_video/
    images_full           # (N, H, W) uint8 — full resolution grayscale
    images_ds             # (N, 640, 640) uint8 — downsampled grayscale (if config has gray)
    images_ds_rgb         # (N, 768, 1280, 3) uint8 — downsampled RGB (if config has rgb)
    original_frame_indices  # (N,) int32 — maps back to original video frames
```

Where N is the number of sampled frames (not the original video length).

### Key metadata attributes (on `raw_video/`)

| Attribute | Value | Description |
|-----------|-------|-------------|
| `import_mode` | `"sampled"` | Distinguishes from full imports |
| `frame_step` | e.g. `100` | The sampling interval used |
| `original_video_length` | e.g. `54000` | Total frames in the source video |
| `imported_frame_count` | e.g. `540` | Frames actually imported |
| `source_video` | filename | Source video filename |
| `source_path` | absolute path | Full path to source video |
| `video_width`, `video_height` | int | Original video dimensions |
| `video_fps` | float | Original video frame rate |

### The `original_frame_indices` array

This array is the key to sampled imports. It maps each frame in the Zarr back
to its position in the original video:

```
Zarr frame 0  →  Video frame 0
Zarr frame 1  →  Video frame 100
Zarr frame 2  →  Video frame 200
...
```

Downstream tools (detection, cropping) use this to translate coordinates and
labels back to the original video when needed.

---

## Tail frame issues

Some videos have corrupted or incomplete frames near the end due to how
the camera encoder finalizes the file. The `--skip-tail-frames` flag drops the
last N frames from the import to avoid decoding errors.

- **Single video default**: 0 (no frames skipped)
- **Batch import default**: 200 frames skipped

If you see decoding errors near the end of an import, try increasing
`--skip-tail-frames`:

```bash
scripts/py -m fisheye.capture.import_video /path/to/video.mp4 \
  --training-data \
  --frame-step 100 \
  --skip-tail-frames 500 \
  --zarr-path /path/to/output.zarr
```

---

## Next steps

Once you have sampled Zarrs, follow the
[training data workflow](training_data.md) to run detection, review labels,
build training configs, and train models.
