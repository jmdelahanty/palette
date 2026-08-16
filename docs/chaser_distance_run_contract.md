# Chaser Distance Run Contract
<!-- contract-meta
version: 1
status: draft
implementation: specified-only
last_updated: 2026-06-17
-->

Purpose: define the per-recording analysis surface for offline fish-to-chaser
distance measurements. This lets GoodCopBadCop and related chaser protocols
compare refined offline fish positions against the acquisition-time chaser
state without mutating the imported stimulus run or the refined detection run.

For the current GoodCopBadCop workflow and user-facing review commands, see
[`goodcopbadcop_analysis_surfaces.md`](goodcopbadcop_analysis_surfaces.md).
For the detailed active coordinate-frame workflow, including the current
homography direction caveat, see
[`goodcopbadcop_coordinate_frame_workflow.md`](goodcopbadcop_coordinate_frame_workflow.md).

## Boundary

`analysis/stimulus_runs/<run>` remains the imported stimulus authority. It owns
chaser state, protocol events, calibration metadata, and frame alignment.

`refined_detect_runs/<run>/instances` remains the offline fish-position
authority for detection-derived centroid measurements.

`analysis/stimulus_epoch_runs/<run>` owns reusable event-aligned windows such as
`pre_event`, `training_event`, and `post_event`.

`analysis/chaser_distance_runs/<run>` is a derived analysis run. It consumes
those authorities and writes framewise distances plus epoch summaries. It must
not patch chaser state, refined detection rows, or stimulus epoch windows.

## Storage

Canonical location:

```text
analysis/chaser_distance_runs/<run>/
  zarr.json
  frames/
    camera_frame_id
    stimulus_frame_num
    timestamp_ns
    stimulus_epoch_window_id
  chasers/
    chaser_index
  positions/
    fish_centroid_img_xy
    fish_centroid_arena_xy
    chaser_arena_xy
    fish_valid
    chaser_valid
  distances/
    distance_px
    distance_mm
    nearest_chaser_index
    nearest_distance_mm
  epoch_summary/
    window_id
    label_bytes
    start_frame
    end_frame
    valid_frame_count
    mean_distance_mm
    min_distance_mm
    p05_distance_mm
    p50_distance_mm
    p95_distance_mm
    fraction_within_threshold
  epoch_distributions/
    window_id
    chaser_index
    bin_edges_mm
    bin_centers_mm
    hist_counts
    hist_density
    valid_sample_count
  visualizations/
    chaser_distance_timeseries_png
    chaser_distance_epoch_median_png
    chaser_distance_epoch_distribution_png
```

The parent group should carry:

```text
analysis/chaser_distance_runs.attrs["latest"] = <run>
analysis/chaser_distance_runs.attrs["latest_complete"] = <run>
```

## Required Run Attributes

- `schema_id`: `"palette.chaser_distance.v1"`.
- `schema_version`: integer schema version.
- `method`: `"offline_detection_to_chaser_distance"`.
- `method_version`: implementation/contract version.
- `row_axis`: `"camera_frames"`.
- `source_detection_path`: exact detection source, normally
  `refined_detect_runs/<run>/instances`.
- `source_detection_kind`: detection resolver mode, such as `"active"`.
- `source_stimulus_run`: exact imported stimulus run.
- `source_stimulus_path`: exact stimulus run path.
- `source_stimulus_epoch_run`: exact epoch-window run when epoch summaries are
  produced.
- `source_stimulus_epoch_path`: exact epoch-window run path.
- `pixels_per_mm_projector`: projector/canvas pixels per millimetre used for
  distance conversion.
- `coordinate_frame`: output frame for distance measurements, currently
  `"arena_relative_canvas_px"`.
- `coordinate_origin`: currently `"top_left_of_active_arena"`.
- `x_axis_direction`: currently `"right"`.
- `y_axis_direction`: currently `"down"`.
- `arena_origin_in_canvas_xy`: active-arena origin used to convert canvas
  coordinates to arena-local coordinates.
- `source_refs`, `parameters`, `summary`, `provenance`, and run-lineage
  fingerprint attrs following `docs/derived_analysis_run_contract.md`.

## Coordinate Contract

Current external-IPC GoodCopBadCop stimulus imports store
`tracking_data/chaser_states` with group-local coordinate metadata:

```text
coordinate_frame = "arena_relative_canvas_px"
coordinate_origin = "top_left_of_active_arena"
x_axis_direction = "right"
y_axis_direction = "down"
position_fields = "chaser_pos_x,chaser_pos_y,..."
```

For this coordinate frame, chaser positions are already arena-local canvas
pixels. Offline fish detection centroids are source-image pixels and are mapped
into the same arena-local canvas frame by applying the stored H5/import
homography to get canvas coordinates, then subtracting
`calibration/arena_geometry.attrs.arena_origin_in_canvas_{x,y}_px`.

Distances are computed in arena-local canvas pixels and converted to
millimetres by:

```text
distance_mm = distance_px / pixels_per_mm_projector
```

Readers must prefer the child-group `chaser_states` coordinate attrs over any
legacy run-level `coordinate_transform` attrs on `analysis/stimulus_runs/<run>`.
The run-level transform is a compatibility hint for older texture-space
imports, not the authority for external-IPC arena-relative chaser states.

## Array Semantics

`frames/*` arrays are dense over camera frames. Missing or unmapped stimulus
frames are represented by `-1` for integer frame IDs.

`positions/fish_valid` identifies frames with a usable offline detection
centroid. `positions/chaser_valid` identifies frames with a chaser position for
each `chaser_index`.

`distances/distance_mm` is shaped `(camera_frame, chaser)` and uses `NaN` where
either source is invalid. Consumers should use `fish_valid`, `chaser_valid`, and
finite-distance checks rather than treating `NaN` alone as the complete failure
state.

`distances/nearest_chaser_index` is `-1` when no finite chaser distance exists
for a frame.

`epoch_summary/*` arrays are shaped `(window, chaser)` where applicable and
reference `analysis/stimulus_epoch_runs/<run>/windows/window_id`. Window labels
are copied into `label_bytes` for convenience; the referenced stimulus epoch run
remains the authority for the window definition.

`epoch_distributions/*` stores fixed-bin distance histograms for each
`(window, chaser)` pair. `bin_edges_mm` and `bin_centers_mm` are shared across
all epochs and chasers within a run, so pre/training/post distribution shapes
can be compared directly. `hist_counts` stores raw sample counts, while
`hist_density` stores probability density normalized so that
`sum(hist_density * bin_width_mm) == 1` for non-empty window/chaser pairs.
Visualizers should prefer these arrays over recomputing histogram bins from
the dense framewise distances when they need fast distribution plots.

Array shape convention:

```text
hist_counts[window, chaser, distance_bin]
hist_density[window, chaser, distance_bin]
```

## Current GoodCopBadCop Writer

The generic writer is:

```bash
scripts/py -m fisheye.analysis.chaser_distance_runs <analysis.zarr> \
  --run-name goodcopbadcop_chaser_distance_v1_20260617 \
  --source active \
  --threshold-mm 20 \
  --distribution-bin-width-mm 2 \
  --apply
```

The registry wrapper for the current GoodCopBadCop batch is:

```bash
scripts/py -m fisheye.utils.run_goodcopbadcop_chaser_distance_analysis \
  --coverage-min 90 \
  --limit 12 \
  --apply
```

The wrapper selects recordings from the registry, requires adequate refined
detection coverage, consumes the latest complete GoodCopBadCop
`analysis/stimulus_epoch_runs` run, and writes
`analysis/chaser_distance_runs/goodcopbadcop_chaser_distance_v1_20260617` by
default.

## Visualization Artifact

The framewise distance time-series PNG is stored in zarr at:

```text
analysis/chaser_distance_runs/<run>/visualizations/chaser_distance_timeseries_png
```

It can be exported with:

```bash
scripts/py -m fisheye.utils.view_zarr_visualization <analysis.zarr> \
  --run-path analysis/chaser_distance_runs/<run> \
  --artifact chaser_distance_timeseries_png \
  --output /tmp/chaser_distance_timeseries.png
```

The epoch-median distance barplot is a separate artifact:

```bash
scripts/py -m fisheye.utils.view_zarr_visualization <analysis.zarr> \
  --run-path analysis/chaser_distance_runs/<run> \
  --artifact chaser_distance_epoch_median_png \
  --output /tmp/chaser_distance_epoch_median.png
```

The per-epoch distance distribution plot is also separate:

```bash
scripts/py -m fisheye.utils.view_zarr_visualization <analysis.zarr> \
  --run-path analysis/chaser_distance_runs/<run> \
  --artifact chaser_distance_epoch_distribution_png \
  --output /tmp/chaser_distance_epoch_distribution.png
```

To view chaser distance beside the corresponding detection occupancy summary
for the same recording, use:

```bash
scripts/py -m fisheye.utils.view_detection_chaser_overview <analysis.zarr> \
  --output /tmp/detection_chaser_overview.png
```

## Future Extensions

- Add a compact columnar export for cross-recording analysis only after the
  per-recording run is stable.
- Add support for legacy texture-space chaser states only through an explicit
  coordinate-path branch that records the selected transform in `parameters`.
- Add keypoint-, mask-, or track-derived fish positions as separate methods or
  source parameters, not by changing the meaning of the current detection
  centroid arrays.
