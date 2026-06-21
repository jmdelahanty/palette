# GoodCopBadCop Group Export Design
<!-- design-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: define the first cross-recording export surface for GoodCopBadCop
chaser analyses. The per-recording analysis Zarr remains authoritative; the
group export is a disposable Parquet product generated from selected Zarr
runs and a virtual collection manifest.

## Storage Boundary

GoodCopBadCop analysis arrays stay in the existing run-local locations:

```text
analysis/detection_occupancy_runs/<run>/spatial_occupancy/<zone_set_id>/
analysis/chaser_distance_runs/<run>/epoch_summary/
analysis/chaser_distance_runs/<run>/epoch_distributions/
```

The group-analysis export should use the existing cross-recording analytics
layout:

```text
/nvme1/exports/palette_analytics/v1/<table>/export_run_id=<run>/part-*.parquet
/nvme1/exports/palette_analytics/v1/manifests/export_run_id=<run>.json
```

The export manifest records the source Zarrs, selected collection manifest,
row counts, part files, diagnostics, host, and git metadata. Rows include the
collection ID and manifest SHA when exported from a virtual collection.

## Tables

### `goodcopbadcop_spatial_occupancy_zones`

Row axis:

```text
recording x detection_occupancy_run x zone_set_id x epoch_window x zone
```

This table supports pooled quadrant and future spatial-zone summaries. Core
columns include:

- source identity: `recording_id`, `zarr_path`, `detection_occupancy_run`,
  `detection_occupancy_path`, `source_lineage_hash`
- source provenance: detection occupancy schema/method/version, source
  detection path, source stimulus epoch run/path, `source_refs_json`
- window metadata: `window_id`, `window_label`, `start_frame`, `end_frame`,
  `start_time_s`, `end_time_s`, `duration_s`
- zone metadata: `zone_set_id`, `zone_set_source`, `zone_set_source_ref`,
  coordinate frame/origin/axis directions, `zone_id`, `zone_label`,
  `display_order`, bounds as `x_min`, `y_min`, `x_max`, `y_max`
- metrics: `frame_count`, `time_s`, `fraction_of_epoch`,
  `fraction_of_detected`, `detected_frame_count`, `missing_frame_count`,
  `total_span_frames`, `coverage_pct`

### `goodcopbadcop_chaser_epoch_summary`

Row axis:

```text
recording x chaser_distance_run x epoch_window x chaser
```

This table supports pooled epoch-level chaser-distance metrics. Core columns
include source identity/provenance, window metadata, `chaser_index`,
`threshold_mm`, `valid_frame_count`, `mean_distance_mm`, `min_distance_mm`,
`p05_distance_mm`, `p50_distance_mm`, `p95_distance_mm`, and
`fraction_within_threshold`.

### `goodcopbadcop_chaser_distance_histogram`

Row axis:

```text
recording x chaser_distance_run x epoch_window x chaser x distance_bin
```

This table supports pooled distance distributions without exporting dense
framewise samples. Pooled counts can be summed across recordings, and densities
can be recomputed or weighted from `hist_count`, `valid_sample_count`, and bin
metadata.

Core metric columns are `distance_bin_index`, `bin_left_mm`, `bin_right_mm`,
`bin_center_mm`, `bin_width_mm`, `hist_count`, `hist_density`, and
`valid_sample_count`.

## Collection Profile

Use a virtual collection manifest with export profile
`goodcopbadcop_chaser`. Required run families are:

- `detection_occupancy_run`
- `chaser_distance_run`

`stimulus_run` is optional. Production exports should prefer explicit run
names when the chosen cohort is final; canaries may rely on latest resolution
if the manifest freezes the resolved run IDs before export.

## Example

```bash
scripts/py -m fisheye.utils.build_virtual_collection_manifest \
  --profile goodcopbadcop_chaser \
  --collection-id goodcopbadcop_chaser_20260621_v001 \
  --collection-name "GoodCopBadCop chaser cohort 2026-06-21" \
  --output /nvme1/exports/palette_analytics/collections/goodcopbadcop_chaser_20260621_v001.manifest.json \
  /groups/.../recording_a_analysis.zarr \
  /groups/.../recording_b_analysis.zarr
```

```bash
scripts/py -m fisheye.utils.export_cross_recording_analytics \
  --collection-manifest /nvme1/exports/palette_analytics/collections/goodcopbadcop_chaser_20260621_v001.manifest.json \
  --output-root /nvme1/exports/palette_analytics \
  --tables goodcopbadcop_spatial_occupancy_zones,goodcopbadcop_chaser_epoch_summary,goodcopbadcop_chaser_distance_histogram \
  --export-run-id run_20260621T_goodcopbadcop_chaser_v001 \
  --jobs 1 \
  --registry /nvme1/palette_registry.sqlite \
  --index-registry
```

## Deferred

Dense framewise distance sample exports are intentionally deferred. The compact
histogram table covers the first pooled distribution workflows with much lower
volume. Add a sample table only when exact unbinned cross-recording analyses
need it.
