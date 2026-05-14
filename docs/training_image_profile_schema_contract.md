# Training Image Profile Schema Contract
<!-- contract-meta
version: 1
status: active
last_verified: 2026-05-13
-->

## Purpose

Training image profiles summarize image-domain differences across sampled
training Zarrs before model training. They are meant to answer questions like:

- Are new recordings darker/brighter than existing training data?
- Are contrast, sharpness, clipping, or illumination gradients different?
- Is fish-vs-background contrast comparable after detection review?
- Can a training data card aggregate these measurements without reopening every
  Zarr?

This profile complements `detection_profile_runs`. Detection profiles describe
label geometry and coverage. Training image profiles describe the sampled frame
pixels themselves.

## Storage

Per-Zarr profile runs live at:

- `analysis/training_image_profile_runs/<run_name>/`
- parent attr `latest`

Required run attrs:

- `schema_name = "training_image_profile"`
- `schema_version = "v1"`
- `created_at_utc`
- `source_dataset_id`
- `source_recording_id`
- `source_zarr_use`
- `source_frame_array`
- `source_frame_count`
- `profiled_frame_count`
- `sample_policy`
- `source_frame_content_hash`
- `source_frame_content_fingerprint_schema_id`
- `source_frame_content_fingerprint_schema_version`
- `source_frame_content_fingerprint_canonicalization`
- `profile_config`
- `profile_summary`
- standard run-lineage attrs from `fisheye.shared.run_lineage_fingerprint`

The writer also mirrors the aggregate intensity histogram into arrays:

- `intensity_histogram_counts`
- `intensity_histogram_bin_edges`

The canonical structured payload remains `attrs["profile_summary"]`. It must be
strict JSON serializable; non-finite numeric values must be written as `null`,
not bare `NaN` or `Infinity`.

## Metrics

Frame source resolution is selected by:

1. `raw_video/images_ds_rgb`
2. `raw_video/images_ds`
3. `raw_video/images_full`

Operators can override this with `--frame-source`.

Per-frame metrics include:

- intensity mean/std/min/max and p01/p05/p50/p95/p99
- contrast as `p99 - p01` and `p95 - p05`
- dark/bright clipping fractions
- sharpness as Laplacian variance and mean gradient magnitude
- illumination center-edge delta
- illumination x/y slopes

If a detection source is available, label-conditioned metrics are also computed
for bbox fish patches and surrounding context:

- fish mean intensity
- background mean intensity
- signed and absolute fish/background contrast
- fish/background standard deviation

Missing detections are not fatal. The image-only profile is still valid.

## Registry Projection

The registry table is `training_image_profile`; the query view is
`training_image_profile_latest`.

The projection stores one latest query row per dataset with:

- source frame array and frame counts
- median intensity/contrast/sharpness summaries
- clipping-rate means
- illumination summaries
- fish/background contrast when available
- recording/subject context via `dataset_context_current`
- full strict-JSON profile payload in `profile_json`

Use:

```bash
scripts/py -m fisheye.utils.training_image_profile \
  /path/to/training.zarr \
  --apply \
  --sync-registry \
  --registry /nvme1/palette_registry.sqlite
```

Dry-run without writing:

```bash
scripts/py -m fisheye.utils.training_image_profile /path/to/training.zarr
```

## Data Cards

Aggregate data cards are generated from registry rows:

```bash
scripts/py -m fisheye.utils.aggregate_training_image_data_card \
  --manifest /path/to/detect_training.manifest.json \
  --registry /nvme1/palette_registry.sqlite \
  --output /path/to/training_image_data_card.json
```

This writes a strict-JSON card with metric distributions, aggregate intensity
histograms, profile-run references, and missing-profile diagnostics. By default
it also writes plot PNGs next to the card.

## Policy

- Training image profiles are derived caches/snapshots, not source labels.
- They should be regenerated when sampled frames change or when a different
  frame source is selected.
- They should be produced after label review if fish/background contrast is
  needed, because label-conditioned metrics depend on the curated detection
  surface.
- They are training-Zarr scoped by default. Analysis-Zarr profiling is allowed
  only for diagnostics with an explicit override.
