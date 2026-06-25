# Group Analytics Viewer Design
<!-- design-meta
status: draft
last_updated: 2026-06-22
-->

Purpose: provide a read-only browser surface over cross-recording Palette
analytics exports. The first supported profile is the GoodCopBadCop chaser
export.

## Boundary

The viewer reads derived Parquet exports and their manifest. It does not open
every source Zarr for cohort plots, does not mutate source data, and does not
replace per-recording marimo inspection.

Authority split:

```text
Zarr archives   = source arrays and per-recording provenance
Parquet exports = cohort query/cache surface
Web viewer      = read-only aggregated presentation
Marimo          = per-recording drilldown and source-array inspection
```

## Implementation

The MVP lives under:

```text
src/fisheye/group_analytics_viewer/
src/fisheye/utils/serve_group_analytics_viewer.py
```

It uses the same lightweight server pattern as the recording status page:

- stdlib `ThreadingHTTPServer`
- static HTML/CSS/JS
- JSON endpoints
- `pyarrow.parquet` reads over the selected export tables

The backend currently expects the GoodCopBadCop table set:

- `goodcopbadcop_spatial_occupancy_zones`
- `goodcopbadcop_chaser_epoch_summary`
- `goodcopbadcop_chaser_distance_histogram`
- `goodcopbadcop_epoch_behavior_summary`
- `goodcopbadcop_epoch_center_distance_histogram`
- `goodcopbadcop_egocentric_epoch_summary`
- `goodcopbadcop_egocentric_distance_bearing_histogram`

## Running

```bash
scripts/py -m fisheye.utils.serve_group_analytics_viewer \
  --export-root /nvme1/exports/palette_analytics \
  --export-run-id run_20260621T_goodcopbadcop_chaser_egocentric_v001 \
  --stats-run-id auto \
  --host 127.0.0.1 \
  --port 8770
```

From a laptop, tunnel the workstation port:

```bash
ssh -L 8770:127.0.0.1:8770 <workstation>
```

Then open:

```text
http://127.0.0.1:8770
```

## API

```text
GET /healthz
GET /api/export/summary
GET /api/options
GET /api/goodcopbadcop/spatial
GET /api/goodcopbadcop/chaser-summary
GET /api/goodcopbadcop/chaser-histogram
GET /api/goodcopbadcop/epoch-center-distance-histogram
GET /api/goodcopbadcop/egocentric-summary
GET /api/goodcopbadcop/egocentric-histogram
GET /api/goodcopbadcop/statistics
GET /api/goodcopbadcop/recordings
GET /api/goodcopbadcop/provenance
```

The browser receives plot-ready aggregates, not raw table dumps. Pooled
histograms are computed by summing `hist_count` first and deriving pooled
density from the pooled count and bin width. Pooled egocentric bearing-distance
bins are also computed from summed `hist_count`, with probabilities derived
after pooling.
Pooled epoch center-distance histograms follow the same count-first rule.

Grouped metric endpoints report descriptive statistics over the contributing
recording-level rows: `n`, `sum`, `mean`, `median`, sample `std_dev`, `sem`,
`min`, and `max`. These are descriptive cohort summaries only.

When a matching stats run is available, grouped metric endpoints prefer
persisted rows from `goodcopbadcop_group_descriptive_summary`. The response
then reports:

```text
summary_source = persisted_descriptive_summary
```

If the descriptive table is absent, stale, or missing a requested metric/group
key, the endpoint computes the display summary from the selected export rows
and reports:

```text
summary_source = computed_from_export_rows
```

The optional statistics endpoint reads
`goodcopbadcop_group_statistical_summary` and
`goodcopbadcop_group_descriptive_summary` from a separate stats export. With
`--stats-run-id auto`, the server discovers the latest stats manifest whose
`source_export_run_id` matches the viewed export. If `--stats-run-id` is passed
explicitly, the manifest must still point back to the viewed export. This keeps
descriptive cohort values, mean differences, confidence intervals, p-values,
q-values, and paired unit counts provenance-linked to the plotted cohort.

Current written example:

```text
source_export_run_id = run_20260621T_goodcopbadcop_chaser_egocentric_v001
stats_run_id = stats_20260621T_goodcopbadcop_chaser_egocentric_v001
```

## Future

- Use the registry to browse indexed export runs before selecting one.
- Add direct links or generated commands for opening the selected recording in
  the marimo per-recording explorer.
- Add generic profile routing once there is a second group-analysis table set.
