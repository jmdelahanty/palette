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

FileGlancer presentation is split into four independently discovered app
manifests: the root group-analytics manifest, the locked recording viewer under
`apps/fileglancer/recording_explorer/`, and the editable recording workspace
under `apps/fileglancer/recording_workspace/`, plus the generic source-Zarr
workspace under `apps/fileglancer/zarr_workspace/`. They share Palette code and
Pixi environments but do not share an app card because their authority and
loading contracts are different.

## Direction

`apps/marimo/palette_explorer.py` is the general entrypoint for persisted
analysis capabilities stored in recording Zarr archives. It discovers canonical
core-analysis runs directly and protocol-specific interactive specs, groups
them into capability providers, and offers only the analyses supported within
the selected provider. Selecting an analysis is the data-loading boundary;
sibling panels are not evaluated merely because their provider is present.

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

Canonical analysis contracts are sufficient for general Core Behavior views.
The explorer currently resolves `analysis/track_kinematics_runs` directly and
adds compatible eye-angle, tail-kinematics, and swim-bout capabilities from
their canonical run families. A plot manifest is therefore not required merely
to expose those persisted arrays. Tail discovery also admits a capability-only
source when no track run is present.

Protocol-specific dashboards continue to use Zarr artifact specs as their
routing contract. A persisted interactive artifact stores:

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
apps/marimo/components/tail_kinematics.py
apps/marimo/components/analysis_catalog.py
```

The top-level notebook should discover capabilities, choose a provider, and
route to the registered component. It should not hard-code protocol-specific
Zarr paths except through the selected component.

## Modular Recording-Explorer Composition

The supported recording explorer is one Marimo entrypoint with modular Python
components, not one notebook per analysis and not one growing notebook that
implements every analysis inline. `apps/marimo/palette_explorer.py` owns only
application-level reactive concerns:

- launch arguments and the selected recording workspace;
- provider, source, and analysis selection;
- generic loading/error state and stale-selection protection;
- invocation of the selected provider adapter; and
- final layout of controls, output, and provenance.

Scientific source validation, projection loading, display parameters, and plot
construction belong to ordinary Python modules below
`apps/marimo/components/`. Those modules remain directly unit-testable without
executing the whole Marimo notebook. Adding an analysis must not require a new
analysis-specific import, load cell, and render branch in the top-level app.

Each modular provider adapter should expose one stable routing boundary:

```text
provider identity and catalog declarations
        -> available analysis IDs for one exact source
        -> load selected read-only projection
        -> construct optional controls
        -> render selected analysis or typed failure
```

The adapter dispatches by exact analysis ID through closed maps. It must reject
unknown IDs, and a component failure must not fall through to a legacy or
candidate provider. Provider adapters may share pure validation and projection
primitives under `src/fisheye/`, but Marimo objects and reactive state do not
belong under `src/`.

### Exact chaser package direction

The exact-successor component is the first provider to adopt this boundary as
new visualization families are mounted. Its target organization is:

```text
apps/marimo/palette_explorer.py                 # thin recording-app shell
apps/marimo/components/analysis_catalog.py      # provider/analysis declarations
apps/marimo/components/chaser_exact_successors.py
                                                  # compatibility facade only
apps/marimo/components/chaser_exact_escape_freeze_contract.py
                                                  # shared closed admission grammar
apps/marimo/components/chaser_exact_escape_freeze_discovery.py
                                                  # consolidated metadata-only join
apps/marimo/components/chaser_exact/
    provider.py                                  # closed load/render dispatch
    projection.py                                # shared verified bundle projection
    radial_near_field.py                         # persisted radial/near-field view
    distance_traces.py                           # exact-time distance view
    trajectory_overlays.py                       # reviewed-arena trajectories
    spatial_occupancy.py                         # persisted occupancy grids
    controller_trials.py                         # logged active trial members
    bout_response.py                             # persisted segmented-bout response
    escape_freeze.py                             # persisted event classifications
    escape_freeze_projection.py                  # exact binding and deep loader
    full_profile.py                              # exact cross-module composition
    provenance.py                                # readable sealed identities
```

The existing `chaser_exact_successors.py` import path remains as a facade while
the package is extracted, so callers and tests do not require a flag-day
rewrite. A mechanical extraction should be a distinct commit from the first
new spatial-occupancy implementation even when both are reviewed in one pull
request. The facade is removed only after all supported callers use the
provider adapter and focused modules.

The legacy `goodcopbadcop_chaser.py` surface remains isolated compatibility
code. New exact-successor analyses must not be added to it, and its size or
branching structure is not a template for modern components. Legacy migration
can proceed provider by provider after the exact-chaser adapter proves the
interface; it is not a prerequisite for publishing the missing exact views.

### Modular acceptance rules

- One analysis module owns its required persisted arrays, validation,
  display-only parameters, plot construction, and focused tests.
- Shared bundle validation runs once at the provider projection boundary; a
  module may request additional arrays but cannot weaken or bypass that proof.
- Expensive arrays load only for the selected analysis. Importing or listing a
  provider performs no scientific array read.
- Render dispatch is closed and deterministic. The notebook does not infer a
  builder from a run name or renderer substring.
- Controls and caches are keyed by exact archive/run/manifest identity plus the
  renderer and display-parameter versions. Late results cannot render under a
  different selection.
- Display projections remain read-only and cannot become inputs to scientific
  metrics, exports, selectors, or authority decisions.
- Focused component tests and a top-level routing test are both required.
  `marimo check` and real-artifact smokes remain release gates.

The incremental implementation now includes the mechanical package split,
spatial occupancy, controller trials, generalized bout response, and
escape/freeze. Each plugged into the same closed adapter without adding a
top-level app branch. Full-profile composition remains the next planned module.

## Current Renderer Registry and Providers

The recording explorer currently registers:

```text
palette-track-kinematics-summary-v1     -> Core behavior
palette-chaser-protocol-dashboard-v1    -> Chaser stimulus
legacy chaser and CRA renderer aliases  -> Chaser stimulus
```

Provider membership is based on the persisted renderer/capability, not the
recording's protocol name. A recording can therefore expose both Core behavior
and Chaser stimulus. The second dropdown contains only analyses from the
selected provider, preventing a control from another visualization family from
remaining selectable while doing nothing.

Core behavior currently provides projected speed, heading/turning, position,
eye-angle/convergence, tail posture/curvature, lineage-compatible swim-bout
segmentation overlays, and canonical pre-period views. Chaser stimulus
provides distance, epoch, egocentric, polar, spatial, CRA, near-field, escape,
artifact, and provenance views when their persisted inputs are present.

## Lazy and Deferred Data Semantics

Polars does not have a native Zarr scanner. The recording explorer therefore
uses a precise two-stage contract:

1. Zarr access is deferred until an analysis is selected. The component reads
   the time coordinate plus only that analysis' source arrays and selected
   contiguous row interval. Compact eye-angle runs additionally project only
   the selected channel columns; the UI defaults to a 60-second window and
   refuses windows above the configured viewer row limit. Tail projections use
   binary search over the persisted sparse frame coordinate, default to ten
   seconds, and refuse more than 10,000 framewise rows.

   New eye-angle materializations also arrange related named channels into
   column-local semantic bundles. Numeric indexes remain non-semantic; see
   [Eye-angle physical column layout](eye_angle_physical_layout.md).
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

The editable single-recording workspace adds a stronger deployment boundary.
It starts a writable copy of this notebook with `marimo edit`, but exposes the
selected Zarr and Palette checkout through read-only Bubblewrap mounts. A final
visible cell provides `exploration`, while implementation cells above it are
collapsed. Pair agents may mutate the notebook copy and write beneath the
session `/workspace`; they cannot write the mounted recording or canonical
Palette source. This mode is intentionally limited to one selected recording.

The single-recording explorer uses targeted track/chaser provider discovery
rather than constructing the audit-oriented whole-recording artifact
inventory. Its recording roster is metadata-only: provider specs are opened
once, after a recording is selected. Unpromoted provider-chaser candidates are
excluded from ordinary discovery because admitting them performs a deep array
audit; they remain inspectable through an explicit candidate-renderer filter.

A validated recording-behavior bundle is admitted at launch by validating its
complete JSON envelope and record digest. Current scientific sources are then
validated at the selected provider boundary: exact chaser views use their
bound child receipts, while Core Behavior verifies the selected provider-motion
manifest, track partition, and every consumed array. This deferral does not
weaken producer, publication, or selector validation.

Within one explorer session, bounded LRU caches retain exact-chaser projections
and cohort view payloads. Keys include the immutable receipt or manifest digest,
the selected analysis/metric and weighting, and display-method identity; exact
chaser keys also include renderer and display-parameter versions. A changed
identity is always a cache miss. These caches are display accelerators only and
cannot become scientific evidence, publication authority, or selector state.
The exact-chaser cache retains at most two projections because those values can
own large arrays.

Within one selected recording/run, the process retains the time coordinate,
resolved bout source, and bout-event rows in RAM so reactive time window changes
do not repeat network metadata and table reads. Speed samples remain bounded
Zarr slices from the authoritative source.

Dense core plots also enforce a display-only serialization budget. Speed and
heading read only explicitly selected series, with one physical speed trace as
the default, and Plotly traces are deterministically decimated to at most
60,000 total displayed points. Raising Marimo's output-size limit is not part
of the design; the source data remain unchanged and exact event boundaries are
retained.

Tail posture is deliberately represented by two complementary persisted
surfaces: the canonical 10-position body-frame local-tangent angles and the
exact 32-position subject-shape spline curvature named in the tail run's source
lineage. The viewer validates bounded source-frame alignment before exposing
the dense curvature. It overlays selected tail scalar traces, bounded x/y
position traces, and lineage-compatible persisted bout intervals. Missing rows
remain gaps; the viewer does not interpolate or recompute tail geometry.

Sampling metadata is part of the interpretation contract. The panel reports
FPS and Nyquist frequency and warns when the acquisition cannot resolve the
typical 20–40 Hz larval tail-beat band. It does not calculate oscillation
frequency, phase, or wave speed from under-sampled recordings. For example, a
30 Hz recording has a 15 Hz Nyquist frequency: its acquired posture samples are
still displayable, but tail-beat spectral claims would be aliased.

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

A direct `--zarr-path` launch shows only the selected recording. Broader
recording discovery is opt-in through `--recordings-root` or `--registry`.
The deployable Pixi entry point is:

```bash
pixi run -e recording recording-app -- --zarr-path <analysis.zarr>
```

The opt-in editable, Pair-compatible entry point is:

```bash
pixi run -e recording recording-workspace -- --zarr-path <analysis.zarr>
```

Its security and connection contract is documented in
[Marimo Pair Recording Workspace](marimo_pair_recording_workspace.md).

For generic exploration of a source, training, analysis, or other Zarr without
a visualization-contract requirement:

```bash
pixi run -e recording zarr-workspace -- --zarr-path <source.zarr>
```

This notebook exposes bounded metadata traversal, explicit NumPy slices, and
direct Polars construction through a stable `exploration` helper. See
[Marimo Pair Source-Zarr Workspace](marimo_pair_zarr_workspace.md).

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
