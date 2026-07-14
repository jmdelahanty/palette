# Palette explorer application deployment

Palette owns the deployable group analytics application because the notebook,
export contracts, catalog checks, and visualizations must resolve to one Palette
commit. FileGlancer may catalog the repository, but `fg-interactive-apps` is not
the source of truth for this application.

## Local Pixi launch

The application has a deliberately small, Linux-only Pixi environment. It does
not install Palette's full acquisition and processing dependency set.

```bash
pixi run app -- --export-root /groups/johnson/johnsonlab/palette_analytics
```

Then open `http://127.0.0.1:2718`. Configure a different local endpoint with
`PALETTE_ANALYTICS_APP_HOST` and `PALETTE_ANALYTICS_APP_PORT`. Set
`PALETTE_ANALYTICS_APP_TOKEN` when local token authentication is needed.

The task launches `apps/marimo/group_analytics_explorer.py` with `marimo run`.
It runs from the repository snapshot and adds only the repository root and
`src/` to `PYTHONPATH`; it does not install the broad `palette` Python project.
Runtime table reads use Polars lazy Parquet scans. The full Palette processing
environment may continue to use PyArrow for export writing and other workflows.

### Pandas and PyArrow runtime boundary

The lightweight Pixi application environment deliberately does not require
PyArrow. Polars reads the immutable Parquet exports directly, and the recording
explorer projects Zarr arrays through NumPy and Polars. Core-behavior plotting
must pass bounded NumPy column mappings to Plotly; it must not call
`Polars.to_pandas()` or `Polars.from_pandas()`, because both operations activate
Polars' optional PyArrow interchange path even when the persisted source is a
Zarr array.

Pandas is still an intentional application dependency for now. An audit of the
three published entry points found three remaining migration boundaries:

- the group analytics component uses pandas DataFrames for numeric coercion,
  pivots, and Plotly preparation;
- the shared legacy track-kinematics and chaser visualization adapters construct
  and return pandas DataFrames; and
- the recording explorer's legacy chaser visualization contracts and component
  APIs return and manipulate pandas DataFrames throughout.

Removing pandas from `pixi.toml` is therefore a separate API migration, not a
dependency cleanup. Migrate the group figures to Polars expressions and NumPy
Plotly inputs first, then change the legacy track-kinematics and chaser
visualization adapters to return Polars frames. Once the three app entry points
and their imported components contain no pandas imports, remove pandas from the
Pixi manifest, regenerate `pixi.lock`, and run all three application smokes.
Until then, keeping pandas declared is required; adding PyArrow solely to bridge
between Polars and pandas is not.

Commit `pixi.lock` with changes to `pixi.toml`. FileGlancer and local launches
therefore resolve the same runtime.

## FileGlancer launch

The root `runnables.yaml` declares only the group analytics explorer. The
launch form requires an authorized analytics export root and then runs:

```bash
pixi run app -- --export-root <selected-directory>
```

The launcher detects `FG_SERVICE_PORT` and binds to `0.0.0.0`. It passes
`FG_SERVICE_TOKEN` to Marimo as the required token password, while FileGlancer
publishes the matching one-click authenticated URL. The service runs from the
repository checkout so the manifest, lockfile, notebook, and scientific
contracts all have the same Git identity.

The selected directory is the application authorization boundary. The notebook
selects an immutable `export_run_id` within that directory, and the catalog
rejects manifests or Parquet parts that resolve outside it. The first published
application has no server-side writer or report-save action.

## Individual recording launch

The recording explorer uses a separate Pixi feature/environment so Zarr,
Matplotlib, and SciPy do not enlarge the group-viewer runtime:

```bash
pixi run -e recording recording-app -- \
  --zarr-path /path/to/recording_analysis.zarr
```

The task launches `apps/marimo/palette_explorer.py` with the same read-only
Marimo service and token protocol as the group app. A direct `--zarr-path`
launch shows exactly the selected recording. Collection browsing is available
only when `--recordings-root` or `--registry` is supplied explicitly, and an
optional `--recording-name-contains` value can then narrow the collection.

FileGlancer discovers the recording viewer independently from
`apps/fileglancer/recording_explorer/runnables.yaml` and exposes it as the
**Palette Recording Explorer** app. Its required
**Recording Analysis Zarr** directory parameter becomes `--zarr-path`; the
directory picker and FileGlancer file-share policy provide the outer path
authorization boundary. Palette opens the Zarr with mode `r`, keeps projections
bounded, and exposes no save action.

## Editable recording workspace

The opt-in **Palette Recording Exploration Workspace** is a third, independently
discovered FileGlancer app declared in
`apps/fileglancer/recording_workspace/runnables.yaml`:

```bash
pixi run -e recording recording-workspace -- \
  --zarr-path /path/to/recording_analysis.zarr
```

It copies `palette_explorer.py` into a per-session writable directory and
launches that copy with `marimo edit`. Existing application cells have their
code collapsed, so their normal controls and plots render above one visible
starter cell. The starter exposes a compact `exploration` object that follows
the selected provider and analysis.

Because an editor or Marimo Pair agent can execute arbitrary Python, application
level `mode="r"` is not the write boundary. The launcher creates a Bubblewrap
mount namespace containing only minimal OS paths, the active Palette/Pixi
checkout, the selected Zarr, and the session workspace. Palette code and the
Zarr are read-only mounts; only `/workspace` and its private temporary/cache
paths are writable. The selected host Zarr appears inside the namespace as
`/data/recording.zarr`. Collection and registry arguments are rejected so the
first canary has exactly one dataset authority.

The FileGlancer execution node must provide `bwrap` and permit unprivileged user
namespaces. The launcher fails closed if Bubblewrap is absent or cannot create
the namespace. See
[Marimo Pair Recording Workspace](marimo_pair_recording_workspace.md) for the
complete trust boundary and Pair connection workflow.

Keeping these as three manifests gives FileGlancer three app cards rather than
one app with several launch modes. The group export viewer, locked recording
viewer, and editable Pair workspace therefore communicate their different data
and write authorities before a user starts a job. All three still resolve to
the same Palette commit, Pixi lock, and reusable notebook/component code.

## Packaging boundary

Pixi is the first deployment target. An Apptainer image remains an optional
hardening and cluster-portability layer after this launch path is stable. A
separate application repository is only warranted after the viewer has an
independently versioned package/API and an independent release cadence.

Individual-recording exploration remains a separate application in this same
repository because it has a different authority and loading contract from the
Parquet viewer. `palette_explorer.py` is the supported recording entry point;
`track_kinematics_explorer.py` remains a focused development and debugging
surface rather than another published FileGlancer service.
