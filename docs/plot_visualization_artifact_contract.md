# Plot Visualization Artifact Contract

This contract defines how Palette analysis runs should persist static and
interactive plot products.

## Goals

- Keep numeric analysis arrays as the source of truth.
- Store stable static plot snapshots for QC, registry thumbnails, reports, and
  review.
- Allow future interactive viewers to render from source arrays without storing
  full HTML apps or decoded RGB image arrays as canonical data.
- Keep all plot artifacts attached to the run whose data they summarize.

## Storage Locations

Run-local visualizations live under:

```text
<run>/visualizations/
```

Examples:

```text
analysis/track_kinematics_runs/offline/<run>/visualizations/
analysis/swim_bout_runs/<run>/visualizations/
analysis/eye_angle_runs/<run>/visualizations/
refined_subject_masks_runs/<run>/visualizations/
```

The parent run should also maintain a lightweight `attrs["visualizations"]`
manifest keyed by artifact name.

## Static PNG Snapshots

Static review plots use the existing Palette finalized-artifact pattern:

```text
<run>/visualizations/<artifact_name>      # zarr array, uint8 PNG bytes
```

Required artifact attrs:

- `artifact_schema_id="palette.visualization.png_bytes.v1"`
- `artifact_type="visualization"`
- `artifact_role="snapshot"`
- `media_type="image/png"`
- `mime="image/png"` for compatibility with existing exporters
- `storage_encoding="png_bytes_uint8"`
- `description`
- `created_at_utc`
- `created_by`
- `content_sha256`
- `byte_length`

Recommended attrs:

- `artifact_signature`
- `source_paths`
- `source_runs`
- `parameters`
- stage-specific summary fields used for filtering or registry display

Use PNG snapshots for:

- stable QC review images
- registry/UI thumbnails
- final reports
- quick operator inspection

Do not store decoded RGB/RGBA plot images as raw arrays by default. They are
larger than PNG bytes, less portable, and can be mistaken for primary
scientific data.

## Interactive Plot Specs

Interactive plots should be persisted as small renderer specs, not as full HTML
documents:

```text
<run>/visualizations/<artifact_name>/      # zarr group
    spec_json                              # zarr array, uint8 UTF-8 JSON bytes
```

Required artifact attrs:

- `artifact_schema_id="palette.visualization.interactive_spec.v1"`
- `artifact_type="visualization"`
- `artifact_role="interactive_spec"`
- `media_type="application/vnd.palette.plot-spec+json"`
- `description`
- `created_at_utc`
- `created_by`
- `renderer`
- `spec_path="spec_json"`
- `content_sha256`
- `byte_length`

Recommended attrs:

- `artifact_signature`
- `snapshot_artifact` pointing at the static PNG snapshot for the same logical
  plot
- `source_paths`
- `source_runs`
- `parameters`

The `spec_json` payload should be renderer-neutral enough that a notebook,
Crimson, or a CLI viewer can interpret it. It should point at canonical arrays
inside the source run rather than duplicating full-resolution data.

When a visualization overlays data from another analysis run, such as swim-bout
intervals on a track-kinematics plot, the spec should store the source run and
selection parameters. Viewers should resolve overlays from the canonical
analysis run, for example `analysis/swim_bout_runs/<run>`, and should not treat
per-track mirrored copies as the source of truth. If the overlay run declares
`source_track_kinematics_run` or `track_id`, viewers should verify those fields
match the plotted run before drawing the overlay.

## Decimated Data

If an interactive plot needs fast rendering for large time series, prefer this
order:

1. Let the viewer decimate directly from canonical arrays.
2. Store explicit decimated arrays under the interactive artifact group only
   when needed for performance.
3. Record the decimation method, source paths, and signature in attrs.

Decimated arrays are cache artifacts, not canonical scientific outputs.

## External Artifacts

Large reports, many-frame galleries, or web bundles may live outside the Zarr
store. In that case, the Zarr run should store only a manifest entry with:

- external path or URI
- content hash
- byte count
- source run/path lineage
- generation parameters

Avoid making external artifacts the only discoverable record of a plot. The run
manifest should be sufficient to know what was generated.

## Implementation Helper

Use `fisheye.shared.plot_artifacts` for new code:

- `write_png_visualization_artifact(...)`
- `write_interactive_plot_spec_artifact(...)`

Existing finalized detect/keypoint/eye-mask artifacts may keep their current
writers, but new plot-producing workflows should use the shared helper so
snapshot and interactive artifacts have consistent attrs and manifest entries.

## Viewing Without Exporting

Use `fisheye.utils.view_zarr_visualization` to inspect a PNG snapshot directly
from a Zarr store without writing an external PNG:

```bash
scripts/py -m fisheye.utils.view_zarr_visualization <archive.zarr> --list
scripts/py -m fisheye.utils.view_zarr_visualization <archive.zarr> \
  --run-path analysis/track_kinematics_runs/offline/<run> \
  --artifact track_kinematics_summary_track_0_png
```

If the selected artifact is an interactive spec with a `snapshot_artifact`
attribute, the viewer resolves that sibling PNG snapshot automatically.

## Marimo Apps

Experimental operator-facing marimo apps live under `apps/marimo/`. They are
viewer layers over the persisted artifact specs and source arrays; they are not
the canonical artifact format.

The track-kinematics app expects optional UI packages (`marimo`, `plotly`) to be
installed in the `scripts/py` environment:

```bash
scripts/py -m marimo run apps/marimo/track_kinematics_explorer.py -- \
  --zarr-path <archive.zarr>
```

The app discovers persisted track-kinematics interactive artifacts and exposes
them as the top-level selector. Once a track run is selected, the app lists
compatible swim-bout overlays derived from that run by checking
`source_track_kinematics_run` and `track_id` on `analysis/swim_bout_runs/<run>`.
`--run-path`, `--swim-bout-run`, and `--speed-level` may still be passed as
initial selections.

By default, the app appends performance events to:

```text
/tmp/palette_track_kinematics_explorer_perf.jsonl
```

Override the path with `--performance-log <path>`, or disable logging with
`--performance-log none`. Each JSONL row records one app phase, such as track
discovery, derived swim-bout discovery, Zarr loading, dataframe construction,
time-window filtering, or Plotly figure construction. This is intended for
profiling interactive selection latency without changing the Zarr artifact
contract.

### Track-Kinematics Overlay Performance Plan

Observed on the 2026-01-28 arena 2 canary, selecting
`tk_hyst4_low2_s005` with 517 visible swim-bout intervals spent most server-side
time in Plotly figure construction:

- `load_interactive_data`: about `0.2 s`
- dataframe construction and filtering: under `0.02 s`
- `build_timeseries_figure`: about `49 s`

The current slow path is the one-`add_vrect(...)`-per-bout overlay. That creates
hundreds of Plotly layout shapes, which is expensive before the browser even
renders the result.

Current implementation:

1. Swim-bout overlays are rendered as one translucent `go.Bar` trace using bout
   midpoints and bout durations as bar widths on a hidden overlay y-axis.
2. The JSONL timing record reports `swim_bout_overlay_renderer`,
   `n_rendered_traces`, and `n_layout_shapes` so regressions are visible.
3. On the same 517-bout candidate, `build_timeseries_figure` dropped from about
   `49 s` with per-bout `vrect` layout shapes to about `0.01 s` with one batched
   bar trace.

If trace rendering becomes the bottleneck again, add optional decimated
interactive arrays under the run-local visualization artifact. Decimation is a
cache layer only; canonical analysis arrays remain the source of truth. Consider
a non-Plotly renderer only if the app needs millions of points, dense
high-frequency overlays, or custom GPU primitives.

WebGPU is not the first optimization for this Plotly/Marimo viewer. Plotly's
documented high-performance path is WebGL-enabled trace types, and its WebGL
scatter implementation does not support area fills as a complete drop-in
replacement for SVG scatter/fill overlays. WebGPU-capable Python stacks exist,
including `wgpu-py`, `pygfx`, and `fastplotlib`, but using them would mean a
separate viewer backend or custom widget rather than a small change to the
current Plotly app.

References:

- Plotly shapes docs, including scatter-filled shapes and layout shapes:
  <https://plotly.com/python/shapes/>
- Plotly high-performance docs, including WebGL tradeoffs:
  <https://plotly.com/python/performance/>
- `wgpu-py`, Python WebGPU API:
  <https://github.com/pygfx/wgpu-py>
- `pygfx`, WebGPU-based Python render engine:
  <https://pygfx.org/>
- `fastplotlib`, plotting library built on `pygfx`/WGPU:
  <https://pypi.org/project/fastplotlib/>

Use `marimo edit` instead of `marimo run` when changing the app:

```bash
scripts/py -m marimo edit apps/marimo/track_kinematics_explorer.py -- \
  --zarr-path <archive.zarr>
```
