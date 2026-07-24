# GoodCopBadCop Group Export Design
<!-- design-meta
status: draft
last_updated: 2026-06-22
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
analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/
analysis/chaser_distance_runs/<run>/egocentric_bearing/<component>/
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

### `goodcopbadcop_epoch_behavior_summary`

Row axis:

```text
recording x chaser_distance_run x epoch_window
```

This table supports pooled fish-level speed, bout, and inter-bout interval
metrics. It is exported from
`analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/per_epoch_fish`.
It deliberately has one row per epoch, not one row per chaser, so fish-level
values cannot be double-counted by object selection.

Core columns include source identity/provenance, window metadata,
`epoch_behavior_component`, `epoch_behavior_path`,
`source_track_kinematics_run`, `source_track_kinematics_scope`,
`source_track_kinematics_track_id`, `source_swim_bout_run`,
`source_swim_bout_level_path`, `source_speed_level`,
`speed_sample_count`, `mean_speed_mm_s`, `median_speed_mm_s`,
`p05_speed_mm_s`, `p95_speed_mm_s`, `max_speed_mm_s`, `total_path_mm`,
`bout_count`, `bout_rate_per_min`, `mean_bout_duration_s`,
`median_bout_duration_s`, `mean_bout_path_length_mm`,
`median_bout_path_length_mm`, `mean_bout_net_heading_change_deg`,
`median_bout_net_heading_change_deg`, `mean_abs_bout_net_heading_change_deg`,
`median_abs_bout_net_heading_change_deg`, `mean_bout_heading_path_deg`,
`median_bout_heading_path_deg`, `inter_bout_interval_count`,
`mean_inter_bout_interval_s`, `median_inter_bout_interval_s`,
`p05_inter_bout_interval_s`, `p95_inter_bout_interval_s`,
`inter_bout_interval_rate_per_min`, `mean_distance_from_arena_center_mm`,
`median_distance_from_arena_center_mm`, `wall_fraction`, `wall_time_s`, and
`tracking_dropout_fraction`.

### `goodcopbadcop_epoch_bout_distribution`

Row axis:

```text
recording x chaser_distance_run x epoch_window x swim_bout
```

This table supports per-recording and pooled distribution plots for bout-level
metrics within each epoch. It is exported from
`analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/per_epoch_bouts`.
It deliberately has one row per assigned swim bout, so inferential grouped
statistics must collapse to fish/recording level before testing.

Core columns include source identity/provenance, window metadata,
`epoch_behavior_component`, `epoch_behavior_path`,
`source_track_kinematics_run`, `source_track_kinematics_scope`,
`source_track_kinematics_track_id`, `source_swim_bout_run`,
`source_swim_bout_level_path`, `source_speed_level`, `bout_source_row`,
`bout_id`, `bout_event_frame`, `bout_event_time_s`, `bout_start_frame`,
`bout_end_frame`, `bout_start_time_s`, `bout_end_time_s`, `bout_duration_s`,
`bout_path_length_mm`, `bout_net_heading_change_deg`,
`abs_bout_net_heading_change_deg`, and `bout_heading_path_deg`.

### `goodcopbadcop_epoch_center_distance_histogram`

Row axis:

```text
recording x chaser_distance_run x epoch_window x center_distance_bin
```

This table supports pooled distance-from-arena-center and wall-hugging
diagnostics. It is exported from
`analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/center_distance_histogram`.
Counts should be pooled by summing `hist_count` across recordings and then
recomputing fractions from pooled counts.

Core columns include source identity/provenance, window metadata,
`bin_index`, `bin_left_mm`, `bin_right_mm`, `bin_center_mm`, `bin_width_mm`,
`hist_count`, `hist_fraction`, `hist_density_per_mm`, `valid_frame_count`,
`arena_radius_mm`, `wall_band_mm`, and `geometry_status`.

`goodcopbadcop_epoch_speed_summary` remains available as a legacy speed-only
export while downstream viewers migrate to this table.

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

### `goodcopbadcop_egocentric_epoch_summary`

Row axis:

```text
recording x chaser_distance_run x egocentric_component x epoch_window x chaser
```

This table supports pooled fish-centric heading-to-chaser summaries. It is
derived from `analysis/chaser_distance_runs/<run>/egocentric_bearing/<component>/epoch_summary`
and intentionally uses the chaser-distance run as the parent analysis surface
rather than creating a separate run family.

Core columns include source identity/provenance, window metadata,
`chaser_index`, `egocentric_component_name`, `egocentric_component_path`,
`source_track_kinematics_run`, `source_track_kinematics_scope`,
`source_track_kinematics_track_id`, `source_heading_array`, `heading_level`,
`angle_convention`, `valid_frame_count`, `circular_mean_bearing_deg`,
`circular_resultant_length`, `mean_alignment_cos`, `mean_lateral_sin`,
`fraction_front_45`, `fraction_lateral_45`, and `fraction_behind_45`.

### `goodcopbadcop_egocentric_distance_bearing_histogram`

Row axis:

```text
recording x chaser_distance_run x egocentric_component x epoch_window x chaser x distance_bin x bearing_bin
```

This table supports pooled polar heatmaps across recordings without exporting
dense framewise point clouds. Counts are summed across recordings for pooled
heatmaps; probabilities are kept for per-recording inspection and should not be
averaged without an explicit weighting policy.

Core columns include source identity/provenance, window metadata,
`chaser_index`, `egocentric_component_name`, `distance_bin_index`,
`bearing_bin_index`, distance-bin edges/center/width in millimeters,
bearing-bin edges/center/width in degrees, `hist_count`,
`hist_probability`, and `valid_sample_count`.

## Collection Profile

Use a virtual collection manifest with export profile
`goodcopbadcop_chaser`. Required run families are:

- `detection_occupancy_run`
- `chaser_distance_run`

`stimulus_run` is optional. The egocentric tables additionally require a
complete `egocentric_bearing` component under the selected chaser-distance run.
That component carries the required track-kinematics source references, so the
collection profile does not need a separate visualization or egocentric run
family. Production exports should prefer explicit run names when the chosen
cohort is final; canaries may rely on latest resolution if the manifest freezes
the resolved run IDs before export.

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
  --tables goodcopbadcop_spatial_occupancy_zones,goodcopbadcop_chaser_epoch_summary,goodcopbadcop_epoch_behavior_summary,goodcopbadcop_epoch_center_distance_histogram,goodcopbadcop_chaser_distance_histogram,goodcopbadcop_egocentric_epoch_summary,goodcopbadcop_egocentric_distance_bearing_histogram \
  --export-run-id run_20260621T_goodcopbadcop_chaser_v001 \
  --jobs 1 \
  --registry /nvme1/palette_registry.sqlite \
  --index-registry
```

## Deferred

Dense framewise distance sample exports are intentionally deferred. The compact
histogram table covers the first pooled distribution workflows with much lower
volume. Dense egocentric point-cloud exports are also deferred; pooled polar
heatmaps should use the distance-bearing histogram table unless exact unbinned
cross-recording analyses need framewise samples.

## Viewer

The first read-only web viewer for these exports is documented in
[`group_analytics_viewer_design.md`](group_analytics_viewer_design.md). It
uses the Parquet export and manifest as its backend surface, while leaving
per-recording source-array inspection to marimo.
