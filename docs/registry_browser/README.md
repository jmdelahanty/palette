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
datasette --immutable /nvme1/palette_registry.sqlite \
  --metadata docs/registry_browser/datasette-metadata.yaml \
  --plugins-dir docs/registry_browser/plugins \
  --static registry-static:docs/registry_browser/static \
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
│   └── render_status_cells.py    # render_cell hook: colors status tokens
└── static/
    └── registry.css              # cell tints + text colors (light theme)
```

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
