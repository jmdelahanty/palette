# GoodCopBadCop Viewer Dataset Cleanup Checklist
<!-- design-meta
status: historical-implementation-record
last_updated: 2026-07-12
-->

Purpose: document the cleanup pass needed after the first GoodCopBadCop
Marimo and group-viewer prototypes. The main correction is that viewers should
render persisted analysis outputs. They should not define scientific metrics
by recomputing them from lower-level arrays at display time.

## Superseding V2 direction

This checklist records the implementation path that produced the current
version-1 exports. Its protocol-prefixed table names, compatibility fallbacks,
and legacy vocabulary are not the target contract for the published group
analytics application.

The version-2 transition is intentionally breaking:

- table names describe an analysis family or data grain, not a protocol;
- chaser tables use a `chaser_*` family across GoodCopBadCop, RedScare, and
  other chaser experiments;
- general bout facts use protocol-neutral tables such as
  `swim_bout_events` and `inter_bout_interval_events`;
- stimulus or chaser epoch membership is represented separately from the
  general bout fact when practical;
- canonical chaser behavior vocabulary is `unknown`, `aggressive`,
  `random_non_chasing`, and `inert`; `static` remains a distinct epoch/activity
  state rather than a chaser behavior class;
- `benign` values and `*_benign` columns are not accepted by the version-2
  capability/query path;
- the published viewer does not translate version-1 tables or vocabulary at
  runtime;
- version-1 exports remain immutable provenance artifacts but are reported as
  requiring re-export rather than rendered through compatibility aliases.

Palette will update the exporter and statistics references, create new
immutable RedScare and GoodCopBadCop exports, compare row counts and scientific
results with the historical exports, and then point the capability-driven
viewer at version 2. Historical names below describe what was implemented at
the time and should not be copied into new contracts.

## Decision

Palette viewers are presentation layers over persisted artifacts.

Allowed in viewers:

- selecting rows by current UI state;
- converting stored arrays to tidy dataframes;
- downsampling dense points for browser performance;
- formatting labels, hover text, colors, legends, and table columns;
- exploratory binning for arbitrary custom windows when clearly labeled as a
  display transform.

Not allowed as the canonical path:

- computing named epoch metrics only inside Marimo;
- deriving group-level means, SEMs, p-values, or CIs only inside a viewer;
- using a viewer-only calculation as the value later exported or cited;
- creating a visualization-specific run family just to hold analysis values.

The canonical flow should be:

```text
per-recording zarr component
  -> per-recording Marimo reader/plotter
  -> cross-recording Parquet export
  -> group statistics export
  -> group viewer reader/plotter
```

## Immediate Problem

The current GoodCopBadCop Marimo epoch kinematics summary is the main violation
of this boundary.

Current behavior:

```text
apps/marimo/components/goodcopbadcop_chaser.py
  _load_epoch_summary_dataframe(...)
```

This helper reads:

- `analysis/chaser_distance_runs/<run>`;
- `analysis/track_kinematics_runs/<scope>/<run>`;
- `analysis/swim_bout_runs/<run>`;
- stimulus epoch windows.

Then it computes per-window speed, bout, and inter-bout interval summaries at
Marimo load time. That was useful to prove the visualization, but it should
not remain the contract for values such as:

- `mean_speed_mm_s`;
- `bout_count`;
- `bout_rate_per_min`;
- `inter_bout_interval_count`;
- `mean_inter_bout_interval_s`;
- `median_inter_bout_interval_s`;
- `inter_bout_interval_rate_per_min`.

Those values should be persisted before the viewer reads them.

## Current Audit

This section records the current viewer/export calculations that need cleanup
or explicit labeling.

| Surface | Current calculation | Status | Action |
| --- | --- | --- | --- |
| Palette Explorer, `GoodCopBadCop` epoch kinematics | Computes epoch speed, bout count/rate, and IBI summaries from track-kinematics and swim-bout runs in `_load_epoch_summary_dataframe(...)` | Contract violation for named metrics | Move to persisted `epoch_behavior_summary`; keep temporary `computed_in_viewer` fallback only while backfilling |
| Cross-recording export, `goodcopbadcop_epoch_speed_summary` | Computes epoch speed directly from chaser-distance positions at export time | Legacy export-derived metric | Replace with persisted `goodcopbadcop_epoch_behavior_summary`; keep existing table as backwards-compatible speed-only legacy table until consumers migrate |
| Group analytics viewer, epoch speed panel | Computes means/medians/std/SEM from exported rows at request time | Acceptable exploratory display fallback, not citable stats source | Prefer persisted group descriptive/statistics tables when present; label request-time summaries as `computed_from_export_rows` |
| Speed-vs-distance export/view | Computes speed-distance bins from framewise speed and chaser distance during export | Useful exploratory binned visualization | If it becomes a named endpoint, persist a per-recording `speed_distance_summary` component under the chaser-distance run; otherwise label as export-derived exploratory |
| Egocentric polar custom windows | Bins selected frame points interactively | Acceptable custom-window display transform | For named epochs/pre-post, render from stored egocentric histogram arrays; keep custom windows labeled as viewer-side transforms |
| Spatial occupancy chaser-zone hatching | Infers which zone contains each chaser in the viewer | Low-risk annotation but not a stored contract | Add stored `chaser_zone_membership` if group exports or figures depend on it |
| Group descriptive stats | Computes descriptive summaries inside request handlers | Acceptable temporary UI convenience | Export `goodcopbadcop_group_descriptive_summary` for citable cohort numbers |

The main rule is whether a value has a scientific name and could be exported,
reported, or compared across recordings. If yes, it belongs in zarr or a
derived export with provenance. If it is only a UI transformation of an
already-stored metric, it can stay in the viewer as long as it is labeled.

## Per-Recording Component

Add a component under the existing chaser-distance run, not a new top-level
visualization run family:

```text
analysis/chaser_distance_runs/<run>/
  epoch_behavior_summary/
    attrs:
      latest = <component_name>
      latest_complete = <component_name>
    <component_name>/
      attrs:
        schema_id = "palette.goodcopbadcop.epoch_behavior_summary.v1"
        schema_version = 1
        status = "complete"
        method = "goodcopbadcop_epoch_behavior_summary"
        method_version = 1
        created_at_utc = ...
        source_refs = {...}
        parameters = {...}
      per_epoch_fish/
      per_epoch_chaser/
      per_epoch_bouts/
      visualizations/
```

This keeps the module associated with the protocol/chaser-distance analysis
surface while still making the metric reusable by Marimo, exports, and group
statistics.

### `per_epoch_fish`

Row axis:

```text
epoch_window
```

Core columns:

- `window_id`
- `window_index`
- `window_label`
- `start_frame`
- `end_frame`
- `start_time_s`
- `end_time_s`
- `duration_s`
- `speed_sample_count`
- `mean_speed_mm_s`
- `median_speed_mm_s`
- `p05_speed_mm_s`
- `p95_speed_mm_s`
- `max_speed_mm_s`
- `total_path_mm`
- `center_distance_sample_count`
- `mean_distance_from_arena_center_mm`
- `median_distance_from_arena_center_mm`
- `p05_distance_from_arena_center_mm`
- `p95_distance_from_arena_center_mm`
- `arena_radius_mm`
- `wall_band_mm`
- `wall_fraction`
- `wall_time_s`
- `bout_count`
- `bout_rate_per_min`
- `median_bout_duration_s`
- `mean_bout_duration_s`
- `median_bout_path_length_mm`
- `mean_bout_path_length_mm`
- `bout_heading_sample_count`
- `mean_bout_net_heading_change_deg`
- `median_bout_net_heading_change_deg`
- `mean_abs_bout_net_heading_change_deg`
- `median_abs_bout_net_heading_change_deg`
- `mean_bout_heading_path_deg`
- `median_bout_heading_path_deg`
- `inter_bout_interval_count`
- `mean_inter_bout_interval_s`
- `median_inter_bout_interval_s`
- `p05_inter_bout_interval_s`
- `p95_inter_bout_interval_s`
- `inter_bout_interval_rate_per_min`
- `tracking_dropout_fraction`

Fish-level epoch metrics must appear only once per epoch. They should not be
duplicated once per chaser in stored tables, because that makes later sums and
group means easy to double-count.

### `center_distance_histogram`

Row axis:

```text
epoch_window x center_distance_bin
```

This table supports the wall-hugging / distance-from-center diagnostic. It is
computed only when circle arena geometry is available. Pooled group histograms
should sum `hist_count` and recompute fractions from the pooled count.

### `per_epoch_chaser`

Row axis:

```text
epoch_window x chaser
```

Core columns:

- all epoch window identity columns from `per_epoch_fish`;
- `chaser_column_index`
- `chaser_index`
- `distance_sample_count`
- `mean_distance_mm`
- `median_distance_mm`
- `p05_distance_mm`
- `p95_distance_mm`
- `min_distance_mm`
- optional `fraction_within_threshold`
- optional threshold parameter columns.

Chaser-distance summaries belong here because they are object-specific.
Bout, IBI, and fish-speed summaries belong in `per_epoch_fish`.

### `per_epoch_bouts`

Row axis:

```text
epoch_window x swim_bout
```

This table supports per-recording bout distribution plots without requiring the
viewer or exporter to recompute bout-to-epoch assignment. It should preserve the
source bout row and include:

- epoch window identity columns;
- `bout_source_row`;
- `bout_id`;
- `bout_event_frame` / `bout_event_time_s`;
- `bout_start_frame` / `bout_end_frame`;
- `bout_start_time_s` / `bout_end_time_s`;
- `bout_duration_s`;
- `bout_path_length_mm`;
- `bout_net_heading_change_deg`;
- `abs_bout_net_heading_change_deg`;
- `bout_heading_path_deg`.

These rows are descriptive raw-layer data. Group-level inference must collapse
to fish/recording-level summaries before testing.

### Provenance

The component must record:

- source chaser-distance run/path;
- source track-kinematics run/path/scope/track id;
- speed level used;
- source swim-bout run/path;
- swim-bout signal level;
- source stimulus epoch run/path;
- frame assignment rule for bouts;
- frame assignment rule for inter-bout intervals;
- window boundary inclusion rule;
- zarr path/source fingerprint;
- git commit/dirty state and host/environment summary.

Frame-based assignment is preferred when frame columns exist. Time-based
assignment is a fallback and must be recorded in `parameters`.

## Per-Recording Plots

The Palette Explorer should read the persisted component and render:

- table for `per_epoch_fish`;
- table for `per_epoch_chaser`;
- bar plot of `bout_rate_per_min` by epoch as the primary bout-count
  comparison;
- secondary bar plot of raw `bout_count` by epoch;
- bar plot of `inter_bout_interval_count` by epoch;
- bar or line plot of `mean_inter_bout_interval_s` by epoch;
- bar plots for mean bout duration, mean bout distance, and mean absolute bout
  heading change;
- bar plot of `wall_fraction` by epoch when arena geometry is available;
- line plot of fish distance from arena center, using
  `center_distance_histogram`, when arena geometry is available;
- optional paired `mean_speed_mm_s` plots;
- existing chaser-distance epoch plots from `per_epoch_chaser`.

The IBI plots must use `per_epoch_fish` so "All chasers" cannot duplicate the
same fish-level values.

Use `bout_rate_per_min` rather than raw `bout_count` for cross-epoch
comparisons because epoch durations can differ. Keep `bout_count` visible as a
supporting count, and show `window_duration_s` plus
`tracking_dropout_fraction` in the table/hover context.

Static PNG artifacts are optional but useful for headless review:

```text
visualizations/epoch_behavior_summary_png
visualizations/epoch_inter_bout_interval_png
```

If static PNGs are written, they should be treated as snapshots of the stored
tables, not as the canonical metric source.

## Marimo Cleanup

Implementation steps:

- [x] Add writer module
      `src/fisheye/analysis/goodcopbadcop_epoch_behavior_summary.py`.
- [x] Store fish-level rows in `per_epoch_fish`, with one row per epoch.
- [x] Store chaser-specific distance rows in `per_epoch_chaser`, with one row
      per epoch x chaser.
- [x] Record source chaser-distance, track-kinematics, and swim-bout
      provenance in component attrs.
- [x] Add a reader in `src/fisheye/visualization/goodcopbadcop_interactive.py`
      for the latest complete `epoch_behavior_summary` component.
- [x] Extend `GoodCopBadCopInteractiveData` or the loaded Marimo view model
      with optional `epoch_behavior_per_epoch_fish_df` and
      `epoch_behavior_per_epoch_chaser_df`.
- [x] Change `apps/marimo/components/goodcopbadcop_chaser.py` so the normal
      path reads the persisted component.
- [x] Keep the current `_load_epoch_summary_dataframe(...)` logic only as a
      temporary legacy fallback, clearly labeled `computed_in_viewer`.
- [x] Surface a visible warning when the fallback path is used.
- [ ] Once all `/groups` GoodCopBadCop zarrs are backfilled, remove or disable
      the fallback by default.

Important display behavior:

- Fish-level plots, including IBI count and mean/median IBI, must read
  `per_epoch_fish`. They must not duplicate values once per selected chaser.
- Chaser-specific plots, including distance summaries, should read
  `per_epoch_chaser` and should respond to the chaser picker.
- The combined table may join `per_epoch_fish` onto `per_epoch_chaser` for
  convenience, but the UI should make clear which columns are fish-level and
  which are chaser-level.
- If the persisted component is missing, the fallback output must be visually
  marked as provisional and computed in the viewer.

Acceptance criteria:

- [x] Marimo values match the stored zarr arrays in unit tests.
- [x] `mean_inter_bout_interval_s` and `inter_bout_interval_count` are visible
      in the table.
- [x] IBI count and mean IBI plots render from `per_epoch_fish`.
- [x] Selecting one chaser affects only chaser-specific tables/plots, not
      fish-level IBI totals.
- [x] Missing component shows a clear "not backfilled" message, not silent
      recomputation.

## Cross-Recording Export Cleanup

Add exported tables for the new component.

### `goodcopbadcop_epoch_behavior_summary`

Row axis:

```text
recording x chaser_distance_run x epoch_window
```

Source:

```text
analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/per_epoch_fish
```

This table should carry all fish-level speed, bout, and IBI metrics, plus
component/source provenance.

### `goodcopbadcop_chaser_epoch_summary`

The existing table should continue to represent chaser-specific distance
metrics. If any fish-level speed or IBI columns have been added there during
prototype work, migrate them to `goodcopbadcop_epoch_behavior_summary` and keep
`goodcopbadcop_chaser_epoch_summary` focused on chaser-specific rows.

Implementation steps:

- [x] Add export loader for `goodcopbadcop_epoch_behavior_summary`.
- [x] Add the table to export manifests and row-count reporting.
- [x] Include source component path, source refs, parameters, and source
      fingerprint columns.
- [x] Keep `goodcopbadcop_epoch_speed_summary` available as a legacy table
      until group viewer panels and downstream notebooks read the new behavior
      table.
- [x] Update group viewer epoch-speed panels to prefer
      `goodcopbadcop_epoch_behavior_summary` and fall back to
      `goodcopbadcop_epoch_speed_summary` only when the persisted behavior table
      is absent.
- [x] Update `docs/goodcopbadcop_group_export_design.md` after implementation.
- [x] Add tests proving exported values equal stored zarr component arrays.

## Group Statistics Cleanup

The group viewer currently computes descriptive means, medians, standard
deviations, and SEMs from exported Parquet rows at request time. This is
acceptable for exploratory display, but citable statistics should be exported
as data.

Add descriptive statistics rows to the group statistics export, or add a
companion table:

```text
goodcopbadcop_group_descriptive_summary
```

Row axis:

```text
metric_family x metric_name x condition/window x group_key
```

Core columns:

- `source_export_run_id`
- `stats_run_id`
- `metric_family`
- `metric_name`
- `condition_name`
- `group_key_json`
- `unit_count`
- `mean`
- `median`
- `std_dev`
- `sem`
- `min`
- `max`
- optional bootstrap CI for the mean or median.

Initial metric families to include:

- `epoch_behavior`: `mean_speed_mm_s`, `bout_rate_per_min`,
  `mean_bout_duration_s`, `mean_bout_path_length_mm`,
  `mean_bout_net_heading_change_deg`,
  `mean_abs_bout_net_heading_change_deg`, `mean_bout_heading_path_deg`,
  `inter_bout_interval_count`, `mean_inter_bout_interval_s`,
  `median_inter_bout_interval_s`, `wall_fraction`,
  `median_distance_from_arena_center_mm`;
- `chaser_distance`: existing distance metrics;
- `spatial_occupancy`: existing zone metrics;
- `cra_primary_endpoint`: existing summary metrics;
- `cra_near_field`: existing summary metrics;
- `egocentric_alignment`: existing epoch metrics.

The group viewer should prefer the persisted descriptive/statistics tables when
available. Request-time summaries may remain as a fallback with a label such as
`computed_from_export_rows`.

Implemented behavior:

- `scripts/py -m fisheye.utils.compute_group_statistics --apply` writes both
  `goodcopbadcop_group_statistical_summary` and, when descriptive rows are
  available, `goodcopbadcop_group_descriptive_summary`.
- The stats manifest records both output tables in `output_tables` and
  `row_counts_by_table`.
- Group viewer endpoints for spatial occupancy, chaser distance, and epoch
  behavior prefer `goodcopbadcop_group_descriptive_summary` when the selected
  stats run matches the viewed export.
- If no descriptive table or matching row exists, the viewer computes the same
  display summary from exported rows and labels it
  `computed_from_export_rows`.

## Borderline UI Calculations

These are lower priority than the epoch behavior summary but should be tracked.

### Speed-vs-Distance Bins

The current speed-vs-distance group visualization is a useful exploratory view
because it asks how locomotion changes as a function of object proximity. It is
also a derived metric: the export combines framewise fish speed with chaser
distance and bins the result. That is acceptable for a prototype, but it should
not become an implicit endpoint hidden inside the export loader.

Preferred future stored component if this becomes a named analysis:

```text
analysis/chaser_distance_runs/<run>/speed_distance_summary/<component>/
  attrs:
    schema_id = "palette.goodcopbadcop.speed_distance_summary.v1"
    source_refs = {...}
    parameters = {
      distance_bin_edges_mm = ...,
      speed_level = ...,
      frame_assignment_rule = ...
    }
  per_epoch_chaser_distance_bin/
```

Row axis:

```text
epoch_window x chaser x distance_bin
```

Core columns:

- `window_id`
- `window_label`
- `chaser_index`
- `distance_bin_left_mm`
- `distance_bin_right_mm`
- `sample_count`
- `mean_speed_mm_s`
- `median_speed_mm_s`
- `p05_speed_mm_s`
- `p95_speed_mm_s`
- `tracking_dropout_fraction`

Cleanup steps:

- [ ] Leave the current export-derived speed-distance table labeled
      exploratory.
- [ ] If this panel becomes part of the planned analysis set, add the stored
      component above.
- [ ] Make the export read the stored component before falling back to
      export-time binning.
- [ ] Include speed level, distance bin edges, and source refs in provenance.

### Egocentric Polar Heatmaps

Current custom-window heatmaps can remain viewer-side because arbitrary windows
are interactive exploration. For named epoch/pre-post views, prefer persisted
egocentric component data:

```text
analysis/chaser_distance_runs/<run>/egocentric_bearing/<component>/epoch_summary
analysis/chaser_distance_runs/<run>/egocentric_bearing/<component>/distance_bearing_histogram
```

Cleanup steps:

- [ ] For named epochs, render from persisted `distance_bearing_histogram`.
- [ ] Use viewer-side binning only for custom windows.
- [ ] Label custom-window heatmaps as computed display transforms.

### Spatial Occupancy Chaser-Zone Labels

The current Marimo component infers which zone contains each chaser so it can
add bar patterns. This is a small annotation, but it should eventually be
stored with the occupancy component or protocol object metadata.

Preferred future stored table:

```text
analysis/detection_occupancy_runs/<run>/spatial_occupancy/<zone_set_id>/chaser_zone_membership
```

Row axis:

```text
epoch_window x chaser x zone
```

Core columns:

- `window_id`
- `window_label`
- `chaser_index`
- `zone_id`
- `zone_label`
- `membership_method`
- `representative_x`
- `representative_y`
- `valid_position_count`

Cleanup steps:

- [ ] Add chaser-zone membership to the spatial occupancy writer or a
      protocol-specific occupancy companion component.
- [ ] Export it if group plots need chaser-zone annotations.
- [ ] Update Marimo to read the stored membership before falling back to
      viewer-side inference.

## Backfill Order

Recommended order for `/groups` GoodCopBadCop zarrs:

1. Implement and unit-test `epoch_behavior_summary`.
2. Run a small real-zarr canary on one recording.
3. Confirm Marimo renders the persisted table and IBI plots.
4. Backfill all GoodCopBadCop zarrs on `/groups`.
5. Refresh the interactive specs if they need source-path updates.
6. Export `goodcopbadcop_epoch_behavior_summary` to Parquet.
7. Recompute group statistics/descriptive summaries.
8. Check the group viewer against the refreshed export.
9. Remove or gate viewer-side fallback calculations.

## Step-by-Step Work Plan

Use this as the execution checklist for the cleanup.

### Phase 1: Stored Per-Recording Behavior Component

- [x] Write `goodcopbadcop_epoch_behavior_summary` builder.
- [x] Read windows from the selected chaser-distance run's `epoch_summary`.
- [x] Resolve track-kinematics source from the swim-bout run when available.
- [x] Compute speed/path summaries from track-kinematics arrays.
- [x] Compute bout counts/rates and bout-duration/path summaries from
      swim-bout tables.
- [x] Compute heading-change-per-bout summaries when track heading samples are
      available.
- [x] Compute IBI count/rate and mean/median/quantile IBI summaries from
      swim-bout interval tables.
- [x] Compute distance-from-arena-center summaries and wall fraction when
      circle arena geometry is available.
- [x] Copy chaser-distance epoch summaries into `per_epoch_chaser`.
- [x] Write center-distance histogram rows for pooled wall/center diagnostics.
- [x] Write attrs with schema, method, source refs, parameters, warnings, and
      git provenance.
- [x] Add in-memory or temp-zarr unit tests for row counts, source attrs, and
      known fixture values.

### Phase 2: Palette Explorer Reader

- [x] Add visualization reader for latest complete `epoch_behavior_summary`.
- [x] Update `GoodCopBadCopLoadedView` to carry the persisted behavior
      component.
- [x] Change the normal epoch summary path to read persisted tables.
- [x] Keep current computation as a clearly labeled fallback.
- [x] Add IBI count and mean/median IBI plots from `per_epoch_fish`.
- [x] Ensure chaser picker filters only `per_epoch_chaser` rows.
- [x] Add tests for persisted-source loading and fallback labeling.

### Phase 3: `/groups` Backfill

- [x] Run one-recording canary with `--overwrite`.
- [x] Open Palette Explorer and confirm the source path points to
      `epoch_behavior_summary/<component>`.
- [x] Backfill all current `/groups` GoodCopBadCop recordings.
- [x] Record skipped recordings and reasons.
- [x] Re-run the canary viewer after bulk backfill.

### Phase 4: Cross-Recording Export

- [x] Add `goodcopbadcop_epoch_behavior_summary` to export table set.
- [x] Export one row per recording x epoch from `per_epoch_fish`.
- [x] Add `goodcopbadcop_epoch_center_distance_histogram` to export table set.
- [x] Include source component path and provenance attrs as columns.
- [x] Keep legacy `goodcopbadcop_epoch_speed_summary` during transition.
- [x] Add export tests proving values come from stored zarr rows.

### Phase 5: Group Statistics and Viewer

- [x] Add `epoch_behavior` metric family to group statistics.
- [x] Compute descriptive summaries for speed, bout, and IBI metrics.
- [x] Make group viewer prefer persisted descriptive summaries where present.
- [x] Keep request-time summaries as `computed_from_export_rows` fallback.
- [x] Add viewer tests for std deviation/SEM visibility.

### Phase 6: Follow-On Cleanup

- [ ] Decide whether speed-vs-distance bins need their own stored component.
- [ ] Persist chaser-zone membership if hatch/pattern annotations become part
      of exported group figures.
- [ ] Confirm named egocentric epoch plots read stored histograms, while custom
      windows remain interactive display transforms.
- [ ] Remove disabled legacy fallbacks after old exports are no longer used.

## Validation Commands

Use repository Python through `scripts/py`.

Static validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m py_compile \
  src/fisheye/analysis/<new_epoch_behavior_module>.py \
  src/fisheye/visualization/goodcopbadcop_interactive.py \
  apps/marimo/components/goodcopbadcop_chaser.py
```

Focused tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/palette-pycache scripts/py -m pytest -p no:cacheprovider \
  tests/unit/fisheye/test_<new_epoch_behavior_module>.py \
  tests/unit/fisheye/test_marimo_palette_explorer_components.py \
  tests/unit/fisheye/test_export_cross_recording_analytics.py \
  tests/unit/fisheye/test_group_statistics.py \
  tests/unit/fisheye/test_group_analytics_viewer.py \
  -q
```

Marimo check should run outside the sandbox:

```bash
scripts/py -m marimo check apps/marimo/palette_explorer.py
```

Real-zarr canaries should run outside the sandbox, especially when they touch
`/groups`, `/nvme1`, or real zarr stores.

## Operational Refresh Steps

Use this sequence when refreshing the current `/groups` GoodCopBadCop cohort.

1. Backfill or refresh per-recording epoch behavior summaries:

   ```bash
   scripts/py -m fisheye.utils.run_goodcopbadcop_epoch_behavior_summary \
     --recordings-root /groups/johnson/johnsonlab/jeremy/recordings \
     --recording-like '2026-06-14%GoodCopBadCop%' \
     --apply \
     --overwrite
   ```

2. Open one recording in Palette Explorer and confirm the epoch behavior source
   path points to:

   ```text
   analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>
   ```

3. Refresh the cross-recording analytics export so
   `goodcopbadcop_epoch_behavior_summary` is written from the stored zarr
   component.

4. Recompute group statistics/descriptives from that export:

   ```bash
   scripts/py -m fisheye.utils.compute_group_statistics \
     --profile goodcopbadcop_chaser \
     --export-root /nvme1/exports/palette_analytics \
     --source-export-run-id <export_run_id> \
     --stats-run-id <stats_run_id> \
     --metrics chaser_distance,spatial_occupancy,epoch_behavior,cra_primary_endpoint,cra_near_field,egocentric_alignment \
     --apply \
     --overwrite
   ```

5. Start the group viewer with that export and `--stats-run-id auto` or the
   explicit stats run id:

   ```bash
   scripts/py -m fisheye.utils.serve_group_analytics_viewer \
     --export-root /nvme1/exports/palette_analytics \
     --export-run-id <export_run_id> \
     --stats-run-id auto \
     --host 127.0.0.1 \
     --port 8770
   ```

6. In the viewer, check that relevant responses report
   `summary_source = persisted_descriptive_summary`. If they report
   `computed_from_export_rows`, the descriptive table is missing, stale, or
   lacks that metric/group key.

## Done Definition

This cleanup is done when:

- per-recording epoch behavior metrics are stored in zarr with provenance;
- Marimo reads those persisted metrics and renders IBI plots from them;
- exports include the fish-level epoch behavior table;
- group statistics include descriptive summaries for epoch behavior metrics;
- group viewer prefers persisted statistics/descriptives over request-time
  recomputation;
- any remaining viewer-side computations are clearly marked as exploratory or
  custom-window display transforms.
