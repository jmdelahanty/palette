# Palette Marimo Explorer Architecture

Palette marimo apps are viewer layers over persisted Zarr artifacts. They
should not define canonical scientific outputs, recompute analysis results, or
invent a separate visualization run family.

## Direction

`apps/marimo/palette_explorer.py` is the general entrypoint for interactive
visualization specs stored in analysis Zarr archives. It discovers persisted
interactive spec artifacts, selects a registered renderer, and delegates the
protocol-specific UI to a component.

Existing focused notebooks remain available while this pattern settles:

```text
apps/marimo/track_kinematics_explorer.py
apps/marimo/goodcopbadcop_explorer.py
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
```

The top-level notebook should only discover specs, choose a renderer, and route
to the registered component. It should not hard-code protocol-specific Zarr
paths except through the selected component.

## Current Renderer Registry

The first registered renderer is:

```text
palette-goodcopbadcop-chaser-dashboard-v1
```

It renders GoodCopBadCop chaser-distance specs written beside
`analysis/chaser_distance_runs/<run>/visualizations/*`.

Track kinematics remains in the existing focused notebook for now. Migrating it
later should mean extracting coherent UI pieces into a component without
changing the underlying `interactive_track_kinematics.py` adapter.

## Running

General explorer:

```bash
scripts/py -m marimo run apps/marimo/palette_explorer.py -- \
  --zarr-path <analysis.zarr>
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

