# GoodCopBadCop Analysis Surfaces
<!-- contract-meta
status: draft
last_updated: 2026-06-21
-->

Purpose: describe the current per-recording GoodCopBadCop analysis products for
users, consumers, and future agents. This is a workflow/read guide; the
individual schema contracts remain in the linked docs.

## Current Run Chain

GoodCopBadCop analysis is stored inside each recording's analysis zarr. It is
not a cross-recording export yet.

The current chain is:

```text
analysis/stimulus_runs/<run>
  imported external-IPC stimulus events, chaser states, calibration, alignment

analysis/stimulus_epoch_runs/goodcopbadcop_stimulus_epochs_v1_20260617
  reusable pre/training/post frame windows

refined_detect_runs/<run>/instances
  offline refined fish detections

analysis/detection_occupancy_runs/goodcopbadcop_detection_occupancy_v1_20260617
  event-aligned refined-detection heatmaps and coverage metrics; planned
  spatial occupancy zone summaries live under this same run family

analysis/chaser_distance_runs/goodcopbadcop_chaser_distance_v1_20260617
  framewise fish-to-chaser distances, epoch summaries, and distributions
```

Planned protocol-specific components under the chaser-distance run include
CRA primary endpoint metrics, egocentric bearing metrics, near-field
avoidance metrics, and epoch behavior summaries. See
[`goodcopbadcop_cra_near_field_design.md`](archive/goodcopbadcop_cra_near_field_design.md)
for the near-field implementation plan and
[`goodcopbadcop_viewer_dataset_cleanup_checklist.md`](archive/goodcopbadcop_viewer_dataset_cleanup_checklist.md)
for the viewer/dataset cleanup plan.

The key design rule is that event-window semantics live once in
`analysis/stimulus_epoch_runs`. Detection occupancy and chaser distance consume
that run instead of independently redefining what "pre", "training", and
"post" mean.

## Consumer Decision Table

| Question | Read this surface |
| --- | --- |
| What frames define pre/training/post? | `analysis/stimulus_epoch_runs/<run>/windows/*` |
| Where was the fish detected in each epoch? | `analysis/detection_occupancy_runs/<run>/heatmaps/*` |
| How much time did the fish spend in coarse spatial zones? | `analysis/detection_occupancy_runs/<run>/spatial_occupancy/<zone_set_id>/summary/*` when written |
| What was detection coverage per epoch? | `analysis/detection_occupancy_runs/<run>/coverage/*` |
| How far was the fish from each chaser over time? | `analysis/chaser_distance_runs/<run>/distances/distance_mm` |
| What is the median/quantile distance per epoch? | `analysis/chaser_distance_runs/<run>/epoch_summary/*` |
| What is the full distance distribution per epoch? | `analysis/chaser_distance_runs/<run>/epoch_distributions/*` |
| What are per-epoch speed, bout, and IBI summaries? | `analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/per_epoch_fish/*` when written |
| What are chaser-specific epoch distance summaries paired with behavior rows? | `analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/per_epoch_chaser/*` when written |
| What are lower-tail near-field avoidance metrics? | `analysis/chaser_distance_runs/<run>/cra_near_field/<component>/*` when written |
| Which refined detections were used? | `attrs["source_detection_path"]` and `attrs["source_refs"]` on each derived run |

For distance distributions, prefer the stored histogram arrays under
`epoch_distributions/` instead of recomputing bins from dense framewise
distances. The stored bins are shared across epochs and chasers within a run,
which makes distribution-shape comparisons direct and reproducible.

Distribution arrays use:

```text
hist_counts[window, chaser, distance_bin]
hist_density[window, chaser, distance_bin]
```

`hist_density` is normalized so each non-empty `(window, chaser)` distribution
integrates to 1 when multiplied by `bin_width_mm`.

## Writers

Write or refresh stimulus epochs plus detection occupancy:

```bash
scripts/py -m fisheye.utils.run_goodcopbadcop_detection_analysis \
  --coverage-min 90 \
  --limit 12 \
  --apply \
  --overwrite
```

Write or refresh chaser distance metrics and visualizations:

```bash
scripts/py -m fisheye.utils.run_goodcopbadcop_chaser_distance_analysis \
  --coverage-min 90 \
  --limit 12 \
  --distribution-bin-width-mm 2 \
  --apply \
  --overwrite
```

Write or refresh CRA near-field avoidance components for the current
GoodCopBadCop `/groups` batch:

```bash
scripts/py -m fisheye.utils.run_goodcopbadcop_cra_near_field \
  --recordings-root /groups/johnson/johnsonlab/jeremy/recordings \
  --recording-like '2026-06-14%GoodCopBadCop%' \
  --limit 12 \
  --apply \
  --overwrite
```

The detection-occupancy and chaser-distance wrappers are registry-based. The
near-field wrapper can also scan an explicit recording root, which is useful
when the current `/groups` batch already has chaser-distance and CRA endpoint
components but is not selected by the registry coverage views.

Write or refresh per-epoch behavior summaries for speed, bouts, and
inter-bout intervals:

```bash
scripts/py -m fisheye.utils.run_goodcopbadcop_epoch_behavior_summary \
  --recordings-root /groups/johnson/johnsonlab/jeremy/recordings \
  --recording-like '2026-06-14%GoodCopBadCop%' \
  --limit 12
```

The command above is a dry run. Add `--apply --overwrite` after the dry run
looks right:

```bash
scripts/py -m fisheye.utils.run_goodcopbadcop_epoch_behavior_summary \
  --recordings-root /groups/johnson/johnsonlab/jeremy/recordings \
  --recording-like '2026-06-14%GoodCopBadCop%' \
  --limit 12 \
  --apply \
  --overwrite
```

## Visualization Artifacts

Detection occupancy writes:

```text
analysis/detection_occupancy_runs/<run>/visualizations/detection_occupancy_overview_png
```

Coarse spatial occupancy zone summaries, such as image quadrants, should be
stored as arrays under:

```text
analysis/detection_occupancy_runs/<run>/spatial_occupancy/<zone_set_id>/
```

These arrays are not visualization artifacts. Marimo components and interactive
specs should point to them when rendering protocol-specific zone summaries. See
[`spatial_occupancy_zone_summary_design.md`](spatial_occupancy_zone_summary_design.md).

Chaser distance writes three separate artifacts:

```text
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_timeseries_png
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_epoch_median_png
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_epoch_distribution_png
analysis/chaser_distance_runs/<run>/visualizations/goodcopbadcop_chaser_dashboard_interactive
```

The distribution plot renders from `epoch_distributions/hist_density`, not from
freshly computed bins. This matters because means and medians can be similar
while the distribution shape changes substantially across epochs.

The interactive dashboard spec is a small JSON artifact stored beside the
chaser-distance PNGs. It points to canonical source arrays in the
chaser-distance run, the matched detection-occupancy run, and the shared
stimulus-epoch run. It does not introduce a separate visualization run family.

Epoch behavior summaries should be stored as a component under the
chaser-distance run:

```text
analysis/chaser_distance_runs/<run>/epoch_behavior_summary/<component>/
  per_epoch_fish/
  per_epoch_chaser/
```

This component is the intended source for named per-epoch speed, bout count,
bout rate, inter-bout interval count, and mean/median inter-bout interval
values. Palette Explorer may temporarily compute these values as a
`computed_in_viewer` fallback while older recordings are being backfilled, but
that fallback is not the analysis contract.

Export any single artifact:

```bash
scripts/py -m fisheye.utils.view_zarr_visualization <analysis.zarr> \
  --run-path analysis/chaser_distance_runs/goodcopbadcop_chaser_distance_v1_20260617 \
  --artifact chaser_distance_epoch_distribution_png \
  --output /tmp/chaser_distance_epoch_distribution.png
```

Export a combined detection occupancy plus chaser distance time-series panel:

```bash
scripts/py -m fisheye.utils.view_detection_chaser_overview <analysis.zarr> \
  --output /tmp/detection_chaser_overview.png \
  --print-paths
```

The combined viewer is an export/composition helper. It does not write a new
zarr artifact and does not recompute metrics.

Open the interactive dashboard:

```bash
scripts/py -m marimo run apps/marimo/goodcopbadcop_explorer.py -- \
  --zarr-path <analysis.zarr>
```

The same stored interactive spec can also be opened through the general Palette
explorer:

```bash
scripts/py -m marimo run apps/marimo/palette_explorer.py -- \
  --zarr-path <analysis.zarr>
```

See `docs/marimo_explorer_architecture.md` for the component and renderer
registry design.

## Coordinate Notes

Current external-IPC GoodCopBadCop imports use
`tracking_data/chaser_states.attrs.coordinate_frame =
"arena_relative_canvas_px"`. Chaser positions are already arena-local canvas
pixels.

Offline refined detection centroids are source-image pixels. The chaser
distance writer maps them into arena-local canvas pixels using the stored
calibration homography and `calibration/arena_geometry` origin attrs. Consumers
should not apply the legacy run-level `coordinate_transform` when group-local
coordinate attrs are present.

## Schema References

- `docs/stimulus_epoch_run_contract.md`
- `docs/detection_analysis_run_surfaces.md`
- `docs/spatial_occupancy_zone_summary_design.md`
- `docs/chaser_distance_run_contract.md`
- `docs/archive/goodcopbadcop_cra_near_field_design.md`
- `docs/artifact_storage_map.md`
