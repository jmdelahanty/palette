# Registry browser (Datasette)

A zero-maintenance, read-only **web browser for the Palette registry**
(`palette_registry.sqlite`). It exposes every table and view — datasets,
provenance, per-stage quality/performance, training lineage, and the recording
step-status ledger — with faceting, full-text-ish search, arbitrary read-only
SQL, JSON/CSV export, and color-coded status cells.

It complements the two existing surfaces:

| Tool | Where | Scope |
|------|-------|-------|
| **Registry TUI** | `python -m fisheye.utils.registry_tui` | Terminal; all tables/views + relationships pane |
| **Status page** | `fisheye.utils.serve_recording_status_page` | Web; recording step-status dashboard only |
| **Datasette browser** (this) | `datasette ...` (below) | Web; all 86 tables/views, ad-hoc SQL, exports |

Datasette reads the live SQLite file directly, so it can never drift from the
schema — there is no code to keep in sync as migrations land.

## Prerequisites

Datasette is a pure-Python package; install it into the project env once:

```bash
pip install datasette
# or: conda install -c conda-forge datasette
```

(`datasette --version` should report ≥ 0.65.)

## Launch

From the repo root:

```bash
datasette --immutable /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --metadata docs/registry_browser/datasette-metadata.yaml \
  --plugins-dir docs/registry_browser/plugins \
  --static registry-static:docs/registry_browser/static \
  --template-dir docs/registry_browser/templates \
  --setting sql_time_limit_ms 8000 --setting default_page_size 100 \
  --host 127.0.0.1 --port 8011
```

Then open <http://127.0.0.1:8011/>. Over SSH, forward the port from your laptop:

```bash
ssh -L 8011:127.0.0.1:8011 <workstation>
```

> **Pass the DB file once.** Use *either* a positional path *or* `--immutable`,
> not both — passing both mounts the database twice (`palette_registry_2`).
> `--immutable` is correct here: the pipeline owns all writes, so opening with
> no write locks is safe and lets Datasette cache row counts.

The registry path follows `RegistryPaths.from_env` conventions; point the
command at whatever copy you want to browse (e.g. `$PALETTE_REGISTRY_PATH`).

## Handy entry points

| What | Path |
|------|------|
| Home (all tables/views) | `/` |
| **Group any view by a column** | `/group` |
| **Models by type** (grouped scrollable tables) | `/models` |
| Datasets, faceted | `/palette_registry/datasets` |
| Pipeline status (wide, color-coded) | `/palette_registry/recording_step_status_wide` |
| Lineage for a dataset | `/palette_registry/dataset_lineage` |
| Quality (detect + keypoint) for a dataset | `/palette_registry/dataset_quality` |
| Training sets containing a dataset | `/palette_registry/training_sets_for_dataset` |
| Runs + exported models for a set | `/palette_registry/training_lineage_for_set` |
| Arbitrary read-only SQL | `/palette_registry?sql=...` |

Append `.json` or `.csv` to any table/query URL for machine-readable output.

## Files

```
docs/registry_browser/
├── README.md                     # this file
├── datasette-metadata.yaml       # titles, sort orders, facets, canned queries, extra_css_urls
├── plugins/
│   ├── render_status_cells.py    # render_cell hook (status colors) + legend
│   └── download_bundle.py        # /download/model/<run_id> -> model zip bundle
├── tests/
│   └── test_download_bundle.py   # security + integrity unit tests for the bundle
├── static/
│   └── registry.css              # cell tints, legend, /models + download styling
└── templates/
    ├── index.html                # home-page override: adds a nav to /models + /group
    └── pages/
        ├── group.html            # custom page served at /group (generic grouper)
        └── models.html           # custom page served at /models (model families)
```

`templates/index.html` mirrors Datasette 0.65's default home page and only adds
the `.home-views` nav under the title. If you bump Datasette and its `index.html`
changes, re-sync the database-listing loop (the nav block itself is independent).

### Custom pages

`templates/pages/group.html` is served at **`/group`** — a generic grouper. Pick
any table/view and a column (via the form or query params) and each distinct
value gets its own scrollable table:

```
/group?table=model_input_shapes&by=task_type
/group?table=recordings&by=recording_type&cols=recording_name,camera_id
```

Params: `table` and `by` (required), `cols` (optional comma-separated display
columns; default all-except-`by`), `limit` (default 2000), `db` (default
`palette_registry`). The table is validated against `sqlite_master` and the
columns against `pragma_table_info` before any identifier is interpolated into
SQL, so params can't inject. Grouping on a high-cardinality column is capped at
200 groups with an on-page notice rather than rendering thousands of tables.

`templates/pages/models.html` is a Datasette **custom page** served at `/models`
(enabled by `--template-dir`). It fetches `model_input_shapes` via the JSON SQL
API and renders one independently-scrollable, sticky-header table per model
family (pose / detect / subject_masks / eye_masks / ...). Family is `task_type`
when present, else inferred from the `set_id`/`run_id` text, so models with a
NULL `task_type` (e.g. eye masks) still group correctly instead of falling into
an "other" pile. `set_id`/`run_id` cells link to their `training_sets` /
`training_runs` rows.

The page's JavaScript lives inside a Jinja `{% raw %}` block — required, since
Datasette renders custom pages through Jinja and would otherwise choke on the
JS template literals and braces. Adding a new grouped page is a copy-paste of
this file with a different SQL/grouping column. Editing the template needs only
a browser refresh (templates are not cached like plugins).

### Model download bundles

`plugins/download_bundle.py` adds **`GET /download/model/<run_id>`**, which streams
a zip of one model's portable artifacts plus a generated `manifest.json`:

```
<set_id>__<run_id>.zip
├── manifest.json     # inference contract, sha256s, NMS/plugin reqs, skeleton, training, engine descriptor
├── weights/best.pt   # always
└── onnx/<name>.onnx  # when the run has an onnx export
```

The `/models` page shows a **Bundle** button per model (and a **+engine** link for
runs that have one). The hardware-specific TensorRT `.engine` is **excluded by
default** — add `?include_engine=1` to include it. Either way the engine is
described in the manifest with its build env and a note that it may not load (or
may run sub-optimally) on different hardware; rebuild from the `.onnx` for best
results.

**Safety model (read this before exposing it anywhere):**
- The `run_id` from the URL is only ever a parameterized DB lookup key — it never
  becomes a filesystem path.
- Every artifact path comes from the registry and must resolve (via `realpath`,
  so symlink escapes are caught) under an allowlisted root. Override the roots
  with `PALETTE_MODEL_ROOTS` (os.pathsep-separated). Default roots are the two
  observed model trees.
- Files are sha256-checked against the registry before shipping; a mismatch omits
  the file and records a warning in the manifest rather than shipping bad bytes.
- **Tunnel-only.** This serves files from disk, so keep the server bound to
  `127.0.0.1` and reach it over SSH. Binding to the network would require auth and
  is explicitly out of scope for now.

Tests: `scripts/py -m pytest docs/registry_browser/tests/test_download_bundle.py -q`
(allowlist/symlink-escape rejection, sha-mismatch omission, engine flag, 404).

### Canned queries

The four parameterized queries in `datasette-metadata.yaml` reproduce the TUI's
relationships pane as web forms (lineage parents/children, per-dataset quality,
training-set membership, set → runs → ONNX/TensorRT). They're plain SQL with
`:dataset_id` / `:set_id` placeholders — add more by copying the pattern.

### Status-cell coloring

`render_status_cells.py` wraps status tokens in `<span class="s s-*">` for the
status views (`recording_step_status_wide`, `_latest`, `recording_step_overview`,
`recording_overview`) and `registry.css` tints the cell. Classification is
worst-first and substring-based, mirroring the TUI palette and the status page's
blocking logic (`MISS`/`STALE`/`UNVER`/`ERR`/`FAIL`):

| State | Match (case-insensitive) | Color |
|-------|--------------------------|-------|
| error | `FAIL`, `ERROR`, `ERR…` | crimson |
| missing | `MISS` | red |
| stale | `STALE` | amber |
| warn | `UNVER`, `PENDING`, `NEEDS`, `WARN` | amber |
| muted | `N/A`, `ABSENT`, `—` | grey |
| ok | `OK`, `APPROV…`, `COMPLET…`, `…%` | green |

Decorated cells resolve correctly (`0 (MISS)` → red, `OK (99%)` → green).
Identity columns (recording names, paths) never match a token, so they're left
uncolored without an explicit allow-list.

To change colors, edit `registry.css` and refresh the browser. To change
*which* values map to which state, edit `classify()` in the plugin and **restart
Datasette** (`--plugins-dir` is not hot-reloaded).

## Caveats

- **Full-cell tint needs CSS `:has()`** (Chrome/Edge/Firefox/Safari, 2023+).
  Older browsers still get colored *text*; only the background fill is skipped.
- **Read-only.** `--immutable` + Datasette's `query_only` mean this surface can
  never modify the registry. Use the pipeline tools for writes.
- **Plugin edits require a restart;** CSS and metadata edits do not (refresh /
  Datasette re-reads metadata per request).
