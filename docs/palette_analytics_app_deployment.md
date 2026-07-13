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

Commit `pixi.lock` with changes to `pixi.toml`. FileGlancer and local launches
therefore resolve the same runtime.

## FileGlancer launch

The root `runnables.yaml` declares the Pixi task as a FileGlancer service. The
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

FileGlancer exposes this as the **Recording Explorer** service. Its required
**Recording Analysis Zarr** directory parameter becomes `--zarr-path`; the
directory picker and FileGlancer file-share policy provide the outer path
authorization boundary. Palette opens the Zarr with mode `r`, keeps projections
bounded, and exposes no save action.

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
