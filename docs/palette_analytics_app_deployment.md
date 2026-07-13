# Palette analytics application deployment

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

## Packaging boundary

Pixi is the first deployment target. An Apptainer image remains an optional
hardening and cluster-portability layer after this launch path is stable. A
separate application repository is only warranted after the viewer has an
independently versioned package/API and an independent release cadence.

Individual-recording exploration remains a second application in this same
repository. It has a different authority and loading contract: it opens one
analysis Zarr with array-aware readers instead of scanning cohort Parquet
tables. A future `recording-app` task should use its own Pixi feature/environment
so Zarr and rendering dependencies do not enlarge this group-viewer runtime.
FileGlancer can expose both tasks as separate service entry points while they
continue to share Palette visualization components. The existing
`palette_explorer.py` and `track_kinematics_explorer.py` should be consolidated
into that supported recording entry point rather than introducing another
notebook.
