# GoodCopBadCop Analysis Surfaces
<!-- contract-meta
status: draft
last_updated: 2026-06-17
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
  event-aligned refined-detection occupancy heatmaps and coverage metrics

analysis/chaser_distance_runs/goodcopbadcop_chaser_distance_v1_20260617
  framewise fish-to-chaser distances, epoch summaries, and distributions
```

The key design rule is that event-window semantics live once in
`analysis/stimulus_epoch_runs`. Detection occupancy and chaser distance consume
that run instead of independently redefining what "pre", "training", and
"post" mean.

## Consumer Decision Table

| Question | Read this surface |
| --- | --- |
| What frames define pre/training/post? | `analysis/stimulus_epoch_runs/<run>/windows/*` |
| Where was the fish detected in each epoch? | `analysis/detection_occupancy_runs/<run>/heatmaps/*` |
| What was detection coverage per epoch? | `analysis/detection_occupancy_runs/<run>/coverage/*` |
| How far was the fish from each chaser over time? | `analysis/chaser_distance_runs/<run>/distances/distance_mm` |
| What is the median/quantile distance per epoch? | `analysis/chaser_distance_runs/<run>/epoch_summary/*` |
| What is the full distance distribution per epoch? | `analysis/chaser_distance_runs/<run>/epoch_distributions/*` |
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

Both wrappers are registry-based. They select active analysis zarrs whose
recording IDs match GoodCopBadCop and whose detection coverage passes the
threshold.

## Visualization Artifacts

Detection occupancy writes:

```text
analysis/detection_occupancy_runs/<run>/visualizations/detection_occupancy_overview_png
```

Chaser distance writes three separate artifacts:

```text
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_timeseries_png
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_epoch_median_png
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_epoch_distribution_png
```

The distribution plot renders from `epoch_distributions/hist_density`, not from
freshly computed bins. This matters because means and medians can be similar
while the distribution shape changes substantially across epochs.

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
- `docs/chaser_distance_run_contract.md`
- `docs/artifact_storage_map.md`
