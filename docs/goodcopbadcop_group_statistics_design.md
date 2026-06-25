# GoodCopBadCop Group Statistics Design
<!-- design-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: define a general group-statistics layer for GoodCopBadCop and future
protocol exports. The first use case is testing whether cross-recording
metrics differ across epochs, chasers, and spatial zones, such as average
fish-to-chaser distance or quadrant occupancy.

## Decision

Create a general "group statistics over derived measurement tables" layer, not
a visualization-specific module and not another per-recording zarr run family.

The statistical module should read cross-recording Parquet exports with
explicit provenance, compute recording-level contrasts and uncertainty, and
write compact statistical summary tables that group viewers and notebooks can
render.

Pipeline boundary:

```text
per-recording zarrs
  -> cross-recording Parquet export
  -> group statistics module
  -> viewer plots/stat tables
```

Per-recording zarrs remain the source of truth. Parquet exports remain the
cohort query/cache surface. The group-statistics output is derived from the
export and should carry the export run id, selected source tables, cohort
manifest identity, parameters, and git/environment provenance.

## Statistical Unit

The primary statistical unit is the recording/animal, not individual frames.

Do not pool all frames across recordings and test frame-level differences as
independent observations. Frame-level pooling is useful for visualization, but
it inflates sample size and can make trivial within-recording temporal
autocorrelation look statistically decisive.

Default policy:

- aggregate or read one metric value per recording per condition;
- compute within-recording contrasts where epochs are paired;
- test the distribution of recording-level contrasts across recordings;
- report `n_recordings` explicitly for every result.

For GoodCopBadCop, common paired contrasts include:

- `training - pre`
- `post - pre`
- `post - training`

## Input Tables

The first implementation should read the existing GoodCopBadCop group export
tables documented in `docs/goodcopbadcop_group_export_design.md`:

```text
goodcopbadcop_spatial_occupancy_zones
goodcopbadcop_chaser_epoch_summary
goodcopbadcop_chaser_distance_histogram
goodcopbadcop_egocentric_epoch_summary
goodcopbadcop_egocentric_distance_bearing_histogram
```

Future GoodCopBadCop exports can add:

```text
goodcopbadcop_fish_heading_epoch_summary
```

The statistics module should prefer Polars for table loading, grouping,
joins, pivots, and aggregation. Convert to NumPy only at the statistical
calculation boundary when necessary.

## Initial Metric Families

### Chaser Distance Contrasts

Source table:

```text
goodcopbadcop_chaser_epoch_summary
```

Candidate metrics:

- `mean_distance_mm`
- `p50_distance_mm`
- `fraction_within_threshold`
- `valid_frame_count` as a QC/support metric, not usually a primary outcome

Grouping keys:

```text
chaser_index
epoch/window label
```

Initial contrasts:

```text
training - pre
post - pre
post - training
```

Expected questions:

- Does average fish-to-chaser distance change from pre to training?
- Does post distance recover toward pre?
- Are effects different for chaser 0 and chaser 1?

### Spatial Occupancy Contrasts

Source table:

```text
goodcopbadcop_spatial_occupancy_zones
```

Candidate metrics:

- `time_s`
- `fraction_of_epoch`
- `fraction_of_detected`
- `coverage_pct` as a QC/support metric, not usually a primary outcome

Grouping keys:

```text
zone_set_id
zone_id
epoch/window label
```

Initial contrasts:

```text
training - pre
post - pre
post - training
```

Expected questions:

- Does occupancy in a quadrant differ between pre and post?
- Are fish spending more time in the chaser-containing zones?
- Do predefined zones from future experimental metadata show epoch-specific
  shifts?

Important caveat: quadrant occupancy fractions are compositional because the
zone fractions sum to one within an epoch. Paired per-zone contrasts are a good
MVP if labeled clearly. If these become primary claims, add compositional or
multinomial methods.

### Distance Distribution Summaries

Source table:

```text
goodcopbadcop_chaser_distance_histogram
```

Initial use:

- pooled distribution visualization by summing `hist_count`;
- per-recording derived distribution metrics, such as near-chaser fraction
  under a distance threshold or distribution quantiles estimated from binned
  counts.

Avoid treating histogram bins as independent tests by default. Binwise tests
may be useful for exploratory visualization, but they should be marked as
exploratory and corrected for multiple comparisons.

## Initial Statistical Methods

Use robust, easy-to-explain methods first:

- paired mean difference;
- paired median difference;
- bootstrap confidence interval over recordings;
- paired sign-flip/permutation p-value;
- Benjamini-Hochberg FDR correction across related metric families.

Suggested defaults:

```text
bootstrap_iterations = 10000
permutation_iterations = exact if n <= 20 else 10000 random sign flips
confidence_level = 0.95
fdr_method = benjamini_hochberg
minimum_recordings = 3
random_seed = 0
```

The module should report when a result is underpowered or skipped because there
are too few paired recordings after filtering.

## Output Tables

Write one compact table per statistics run:

```text
goodcopbadcop_group_statistical_summary
```

Row axis:

```text
metric_family x metric_name x contrast_name x group_key
```

Core columns:

- source identity:
  - `export_run_id`
  - `collection_id`
  - `collection_manifest_sha256`
  - `source_table`
  - `source_row_count`
- metric identity:
  - `metric_family`
  - `metric_name`
  - `contrast_name`
  - `condition_a`
  - `condition_b`
  - `group_key_json`
- sample/support:
  - `unit`
  - `unit_count`
  - `paired_unit_count`
  - `excluded_unit_count`
  - `missing_policy`
- descriptive stats:
  - `mean_a`
  - `mean_b`
  - `mean_difference`
  - `median_difference`
  - `std_difference`
  - `effect_size`
- inferential stats:
  - `ci_low`
  - `ci_high`
  - `p_value`
  - `q_value`
  - `test_method`
  - `bootstrap_iterations`
  - `permutation_iterations`
- provenance:
  - `parameters_json`
  - `created_at_utc`
  - `git_commit`
  - `git_branch`
  - `git_dirty`

Write a companion descriptive table in the same stats run:

```text
goodcopbadcop_group_descriptive_summary
```

Row axis:

```text
metric_family x metric_name x condition/window x group_key
```

Core columns:

- source identity:
  - `source_export_run_id`
  - `stats_run_id`
  - `source_table`
  - `source_manifest_sha256`
- metric identity:
  - `metric_family`
  - `metric_name`
  - `condition_name`
  - `condition_label`
  - `group_key_json`
- sample/support:
  - `unit`
  - `unit_count`
- descriptive stats:
  - `sum`
  - `mean`
  - `median`
  - `std_dev`
  - `sem`
  - `min`
  - `max`
- provenance:
  - `parameters_json`
  - `created_at_utc`

This table is the preferred source for cohort-level means, medians, standard
deviations, and SEMs displayed by the group viewer. Request-time summaries in
the viewer are a fallback and must be labeled `computed_from_export_rows`.

Optional companion tables:

```text
goodcopbadcop_group_statistical_units
goodcopbadcop_group_statistical_diagnostics
```

The unit table is useful for debugging and plotting paired lines:

```text
stat_result_id x recording_id x condition x value
```

The diagnostics table can record skipped tests, missing conditions, insufficient
recordings, and multiple-comparison families.

## Storage

Use the same analytics export root style as the group export:

```text
/nvme1/exports/palette_analytics/v1/<table>/export_run_id=<stats_run>/part-*.parquet
/nvme1/exports/palette_analytics/v1/manifests/export_run_id=<stats_run>.json
```

The stats manifest should reference:

- source export run id;
- source export manifest path and SHA;
- input tables and row counts;
- output tables and row counts;
- contrast definitions;
- metric definitions;
- random seeds and iteration counts;
- code/git/environment provenance.

## CLI Shape

Proposed command:

```bash
scripts/py -m fisheye.utils.compute_group_statistics \
  --profile goodcopbadcop_chaser \
  --source-export-run-id run_20260621T_goodcopbadcop_chaser_v001 \
  --export-root /nvme1/exports/palette_analytics \
  --stats-run-id stats_20260621T_goodcopbadcop_chaser_v001 \
  --metrics chaser_distance,spatial_occupancy \
  --contrasts training-pre,post-pre,post-training \
  --bootstrap-iterations 10000 \
  --permutation-iterations 10000 \
  --minimum-recordings 3 \
  --apply
```

Default mode should be dry-run/planning, reporting which tables and contrasts
would be computed and how many paired recordings each contrast has.

## Viewer Integration

The existing group analytics viewer should read the statistics tables as an
optional overlay, not recompute statistics in the browser.

Initial viewer additions:

- show mean differences and confidence intervals beside pooled charts;
- show paired recording counts for each metric;
- prefer `goodcopbadcop_group_descriptive_summary` for descriptive cohort
  values when present;
- label request-time summaries as `computed_from_export_rows` when no matching
  descriptive row is available;
- distinguish exploratory results from primary configured contrasts;
- expose provenance for the stats run and its source export run.

## Implementation Status

First implementation slice added on 2026-06-21:

- `src/fisheye/group_statistics/paired.py` implements paired mean/median
  contrasts, bootstrap confidence intervals, paired sign-flip p-values, and
  Benjamini-Hochberg q-values.
- `src/fisheye/group_statistics/goodcopbadcop.py` maps the current
  GoodCopBadCop export tables to recording-level contrast rows.
- The same GoodCopBadCop profile writes
  `goodcopbadcop_group_descriptive_summary` for citable cohort descriptive
  values, including speed, bout, IBI, chaser-distance, spatial-occupancy, CRA,
  near-field, and egocentric metric families when those source tables are
  present in the selected export.
- `scripts/py -m fisheye.utils.compute_group_statistics` computes dry-runs by
  default and writes `goodcopbadcop_group_statistical_summary` plus
  `goodcopbadcop_group_descriptive_summary` with `--apply`.
- The group analytics viewer can auto-discover a matching statistics run by
  `source_export_run_id`, render a compact Group Statistics table, and prefer
  persisted descriptive rows for cohort summaries.

The first real artifact was written for:

```text
source_export_run_id = run_20260621T_goodcopbadcop_chaser_egocentric_v001
stats_run_id = stats_20260621T_goodcopbadcop_chaser_egocentric_v001
```

## Implementation Checklist

- [x] Add a `docs/` contract or design note for the generic statistics output
      schema once implementation starts.
- [x] Add `src/fisheye/group_statistics/` or `src/fisheye/analysis/group_statistics/`
      with pure functions for paired contrasts, bootstrap intervals,
      permutation tests, and FDR correction.
- [x] Use Polars for export table loading, filtering, grouping, pivoting, and
      joining.
- [x] Add a GoodCopBadCop profile adapter that maps source tables to metric
      families:
      - [x] `goodcopbadcop_chaser_epoch_summary` -> chaser distance metrics.
      - [x] `goodcopbadcop_spatial_occupancy_zones` -> spatial occupancy
            metrics.
      - [ ] `goodcopbadcop_chaser_distance_histogram` -> pooled distribution
            and derived per-recording distribution metrics.
      - [x] `goodcopbadcop_egocentric_epoch_summary` -> heading-to-chaser
            alignment metrics.
      - [ ] `goodcopbadcop_egocentric_distance_bearing_histogram` -> pooled
            polar heatmap distributions.
- [x] Implement condition/epoch label normalization so `pre_event`,
      `training_event`, and `post_event` can be referenced by contrast aliases.
- [x] Build recording-level paired contrast tables with one row per recording
      and condition pair.
- [x] Implement minimum-recording filtering and explicit skipped-result
      diagnostics.
- [x] Implement bootstrap confidence intervals over recording-level
      differences.
- [x] Implement paired sign-flip/permutation p-values.
- [x] Implement Benjamini-Hochberg q-values within configured comparison
      families.
- [x] Write `goodcopbadcop_group_statistical_summary` as Parquet.
- [x] Write `goodcopbadcop_group_descriptive_summary` as Parquet.
- [ ] Optionally write `goodcopbadcop_group_statistical_units` for paired-line
      plots and diagnostics.
- [x] Write a stats manifest referencing the source export manifest and source
      table row counts.
- [x] Include both statistical and descriptive output tables in the stats
      manifest when descriptive rows are written.
- [x] Add `scripts/py -m fisheye.utils.compute_group_statistics` CLI with
      dry-run default and `--apply` write mode.
- [x] Add unit tests for paired contrast construction, bootstrap intervals,
      permutation p-values, q-value correction, missing-pair diagnostics, and
      GoodCopBadCop table adapters.
- [x] Add a small fixture Parquet export for end-to-end CLI tests.
- [x] Update `docs/group_analytics_viewer_design.md` to describe optional
      stats table consumption.
- [x] Add group viewer endpoints for statistical summaries.
- [x] Add viewer UI for confidence intervals, p/q-values, and paired unit
      counts.

## Deferred

- Mixed-effects models across animals, batches, or acquisition dates.
- Compositional spatial-occupancy models.
- Binwise distribution hypothesis tests.
- Direct zarr-backed group statistics without Parquet export.
- Automatic primary-vs-exploratory contrast registration from protocol
  metadata.
