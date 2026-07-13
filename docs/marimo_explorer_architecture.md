# Palette Marimo Explorer Architecture

Palette marimo apps are viewer layers over persisted Zarr artifacts. They
should not define canonical scientific outputs, recompute analysis results, or
invent a separate visualization run family.

The published group-analytics application is also intended to run as a
FileGlancer service app. FileGlancer owns authentication, filesystem selection,
and service lifecycle; Marimo owns the reactive export selector and viewer UI.
See [FileGlancer and Marimo Integration Design](fileglancer_marimo_integration_design.md).
The desired dataset-selection flow, provider navigation, scientific display
defaults, and packaging gate are specified in
[Group Analytics Marimo Application Design](group_analytics_marimo_application_design.md).

## Direction

`apps/marimo/palette_explorer.py` is the general entrypoint for interactive
visualization specs stored in analysis Zarr archives. It discovers persisted
interactive spec artifacts, groups them into capability providers, and offers
only the analyses supported within the selected provider. Available analyses
are arranged as anchored document sections with a vertical contents sidebar.
An analysis section entering the viewport is the data-loading boundary;
off-screen sibling panels are not evaluated merely because their provider is
present.

Existing focused notebooks remain available while this pattern settles:

```text
apps/marimo/track_kinematics_explorer.py
apps/marimo/goodcopbadcop_explorer.py
apps/marimo/group_analytics_explorer.py
```

Those notebooks are useful as protocol-specific debug harnesses, but new
protocol views should be designed as components that can be mounted by the
general explorer.

## Boundaries

Zarr artifact specs are the routing contract. A persisted interactive artifact
stores:

```text
<run>/visualizations/<artifact_name>/spec_json
```

with attrs such as `artifact_role="interactive_spec"`, `renderer`, `source_paths`,
`source_runs`, and `snapshot_artifact`.

Protocol-specific data loading and spec validation live under `src/`:

```text
src/fisheye/visualization/interactive_track_kinematics.py
src/fisheye/visualization/goodcopbadcop_interactive.py
```

Marimo rendering lives under `apps/marimo/components/`:

```text
apps/marimo/components/registry.py
apps/marimo/components/common.py
apps/marimo/components/static_artifacts.py
apps/marimo/components/provenance.py
apps/marimo/components/goodcopbadcop_chaser.py
apps/marimo/components/core_behavior.py
apps/marimo/components/analysis_catalog.py
```

The top-level notebook should only discover specs, choose a renderer, and route
to the registered component. It should not hard-code protocol-specific Zarr
paths except through the selected component.

## Current Renderer Registry and Providers

The recording explorer currently registers:

```text
palette-track-kinematics-summary-v1     -> Core behavior
palette-chaser-protocol-dashboard-v1    -> Chaser stimulus
legacy chaser and CRA renderer aliases  -> Chaser stimulus
```

Provider membership is based on the persisted renderer/capability, not the
recording's protocol name. A recording can therefore expose both Core behavior
and Chaser stimulus. The contents sidebar contains only analyses from the
selected provider, preventing navigation to a visualization family that is not
active. Recording, provider, and analysis-run selectors remain in that sidebar.
Each contents entry is a stable hash link to its analysis section.

Core behavior currently provides projected speed, heading/turning, position,
eye-angle/convergence, lineage-compatible swim-bout segmentation overlays,
and canonical pre-period views. Chaser stimulus
provides distance, epoch, egocentric, polar, spatial, CRA, near-field, escape,
artifact, and provenance views when their persisted inputs are present.

## Lazy and Deferred Data Semantics

Polars does not have a native Zarr scanner. The recording explorer therefore
uses a precise two-stage contract:

1. Zarr access is deferred until an analysis section enters the viewport. A
   contents link scrolls directly to its anchor, which makes its lazy sentinel
   visible. The component then reads the time coordinate plus only that
   analysis' source arrays and selected contiguous row interval.
2. The resulting projection is represented as a `polars.LazyFrame`; filtering,
   column projection, grouping, and descriptive display queries remain lazy
   until collection.

This is called a **deferred Zarr projection**, not a lazy Zarr scan. Source Zarr
arrays are still materialized for the selected projection.

For immutable analytics exports, Parquet has a stronger contract:
`scan_export_parquet(...)` delegates to `polars.scan_parquet(...)`, retaining
projection and predicate pushdown from storage through collection. Dense
frame/sample export panels should use that path rather than eagerly constructing
Python row dictionaries.

Both paths are read-only. Viewer projections and exploratory summaries are not
written into the recording or export.

Section shells and controls are inexpensive and may exist before their data are
read. Each section has an independent control instance; changing an epoch,
series, or binning choice in one section does not mutate another section. A
loaded section reuses the existing in-process provider caches for subsequent
reactive control changes.

The single-recording explorer uses targeted track/chaser provider discovery
rather than constructing the audit-oriented whole-recording artifact
inventory. Within one selected recording/run, the process retains the time
coordinate, resolved bout source, and bout-event rows in RAM so reactive time
window changes do not repeat network metadata and table reads. Speed samples
remain bounded Zarr slices from the authoritative source.

Dense core plots also enforce a display-only serialization budget. Speed and
heading read only explicitly selected series, with one physical speed trace as
the default, and Plotly traces are deterministically decimated to at most
60,000 total displayed points. Raising Marimo's output-size limit is not part
of the design; the source data remain unchanged and exact event boundaries are
retained.

Chaser-distance lines use a separate 24,000-point budget shared across visible
traces. The display projection selects real source samples and retains each
time bucket's endpoints, minimum, and maximum, preserving transient extrema
while reducing browser hover targets. The full filtered dataframe remains the
read-only debug/query surface.

## Running

General explorer:

```bash
scripts/py -m marimo run apps/marimo/palette_explorer.py -- \
  --zarr-path <analysis.zarr>
```

For registry-backed recording discovery without opening every sibling Zarr:

```bash
scripts/run_palette_explorer.sh \
  --zarr-path <analysis.zarr> \
  --registry <palette_registry.sqlite>
```

GoodCopBadCop focused notebook:

```bash
scripts/py -m marimo run apps/marimo/goodcopbadcop_explorer.py -- \
  --zarr-path <analysis.zarr>
```

Track-kinematics focused notebook:

```bash
scripts/py -m marimo run apps/marimo/track_kinematics_explorer.py -- \
  --zarr-path <analysis.zarr>
```

Group analytics export notebook:

```bash
scripts/py -m marimo run apps/marimo/group_analytics_explorer.py -- \
  --export-root /nvme1/exports/palette_analytics \
  --export-run-id latest \
  --stats-run-id auto
```

The group analytics app discovers immutable export manifests beneath the
authorized root and provides a reactive export selector. `--export-run-id`
chooses the initial export; it does not prevent switching datasets without a
process restart. Registry-free discovery and symlink-aware root confinement are
implemented. Composable capability and panel-provider routing is the next
application layer.
