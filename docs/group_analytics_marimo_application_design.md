# Group Analytics Marimo Application Design
<!-- contract-meta
status: active-design
last_updated: 2026-07-12
-->

## Decision

Palette will settle the group analytics Marimo application before packaging or
deploying it. The existing `apps/marimo/group_analytics_explorer.py` is a useful
GoodCopBadCop epoch-summary prototype, not yet the general published
application.

The published application will use a two-stage dataset-selection model:

1. FileGlancer selects and authorizes a shared analytics **dataset root** when
   the service launches.
2. Marimo discovers the immutable exports available beneath that root and lets
   the user select an individual dataset by `export_run_id`.

The first application is read-only and displays one export at a time.

## Current prototype assessment

The current notebook:

- fixes `export_root`, `export_run_id`, and optional `stats_run_id` when the
  Marimo process starts;
- presents a generic “Palette Group Analytics” title but primarily consumes
  GoodCopBadCop epoch-behavior tables;
- exposes bout histograms, inter-bout intervals, one selected epoch summary,
  epoch statistics, recording rows, and table counts;
- hard-codes epoch and metric controls in notebook cells;
- reports health as one summary value without a diagnostic panel;
- does not expose provenance, report inventory, spatial, chaser, CRA,
  near-field, or egocentric panels;
- does not route panels from available table capabilities;
- cannot switch exports reactively.

The library query layer is substantially richer than the notebook. It already
contains queries for export summary and health, options, spatial occupancy,
chaser summaries and histograms, epoch speed and bout distributions,
speed-distance bins, CRA object-phase and specificity endpoints, quadrant and
near-field analyses, egocentric summaries and histograms, statistics,
recordings, and provenance.

The redesign should reuse those tested queries through normalized panel
payloads rather than reimplementing scientific aggregation in Marimo cells.

## What “dataset” means in this application

The user-selectable dataset is one immutable Palette analytics export:

```text
export_run_id=<id>
  export manifest
  one or more Parquet table partitions
  collection identity and source lineage
  optional linked statistics export
  optional indexed reports
```

It is not an arbitrary directory, an individual Parquet part file, or a source
recording Zarr. A FileGlancer directory picker supplies the authorized search
boundary; the Marimo selector supplies the scientific dataset identity.

## How FileGlancer exposes datasets to the app

FileGlancer app manifests support typed `file` and `directory` parameters.
For local paths, FileGlancer:

1. presents a server-side file browser to the user;
2. verifies that the chosen path lies inside an allowed file share;
3. checks existence, readability, and file-versus-directory type using the
   requesting user's permissions;
4. passes the absolute path as a safely quoted command-line argument;
5. bind-mounts a selected directory, or the parent of a selected file, into an
   Apptainer container at the same absolute path.

FileGlancer does not inspect Palette manifests or SQLite rows. A path mentioned
inside a registry is not automatically authorized or mounted. Palette must
validate all resolved export paths against the explicit FileGlancer-selected
root.

### Recommended root-selection mode

The FileGlancer launch form provides:

```yaml
- flag: --export-root
  name: Analytics Export Root
  type: directory
  required: true
  exists: true
```

The selected directory is normally a root such as:

```text
<analytics-root>/
  v1/
    manifests/
    <table-name>/export_run_id=<id>/
    reports/export_run_id=<id>/
```

Marimo lists every valid export manifest beneath this root, optionally enriched
by a registry snapshot. Switching `export_run_id` changes the application
context without restarting the FileGlancer service.

This is the preferred interpretation of “use Marimo to select the dataset.”
FileGlancer chooses the authorized collection of possible datasets; Marimo
chooses one dataset from that collection.

### Optional registry/catalog input

A second FileGlancer parameter may select a read-only SQLite registry or
analytics catalog snapshot:

```yaml
- flag: --registry
  name: Analytics Registry Snapshot
  type: file
  required: false
  exists: true
```

The registry improves labels and queryability but does not expand the mounted
dataset boundary. Rows whose output roots or table paths resolve outside
`--export-root` are excluded or rejected.

The app must also work without a registry by reading immutable export manifests
directly from the selected root. This provides a portable “copy this export
root to a server and browse it” mode.

### Exact-export launch mode

A future runnable may accept one exact export directory or `export_run_id` and
open it immediately. This is useful for deep links and “open this export”
actions, but it should reuse the same application and validation layer.

It is not the first workflow because the current FileGlancer app launch URLs do
not encode arbitrary parameter values, and the desired primary experience is
selection inside Marimo.

### Multiple roots and comparisons

FileGlancer's inspected parameter model accepts individual scalar directory
parameters rather than an arbitrary list of directories. Cross-root access
would therefore require one of:

- a common authorized parent root;
- multiple explicitly declared optional directory parameters;
- a future FileGlancer multi-path parameter;
- a server-managed catalog whose referenced roots are also explicitly mounted.

The first application will not compare exports across roots. It will select one
export root at launch and one export within that root at a time. Comparison
within one export, such as conditions or stimulus epochs, remains supported.

## Target user journey

### 1. Launch

The user starts “Palette Analytics Explorer” from FileGlancer, selects a shared
analytics export root, and optionally selects a read-only registry snapshot.
FileGlancer starts an authenticated Marimo service.

### 2. Select a dataset

Marimo shows a searchable export selector. Each choice should display:

- collection or dataset name;
- protocol/cohort label when available;
- `export_run_id`;
- recording count;
- creation time;
- health state;
- major available table or analysis families.

The selector should prefer a recently created healthy export but should not use
the lexicographically latest manifest silently without showing the choice.

### 3. Understand the selected export

The overview shows identity, collection hash, manifest path relative to the
authorized root, source recording count, available tables, diagnostics,
statistics linkage, and indexed reports.

### 4. Explore available panels

Navigation is generated from table capabilities. Unsupported providers are not
shown as broken controls. Missing expected tables and unhealthy exports receive
explicit explanations.

### 5. Inspect provenance

Every panel exposes source tables, aggregation unit, filters, sample counts,
and applicable statistics provenance. Recording and table inventories remain
available as secondary details.

## Application information architecture

```text
Dataset selector
  └── Export overview and health
       ├── Core behavior
       │    ├── Speed and path length
       │    ├── Swim-bout summaries and distributions
       │    ├── Heading and turning
       │    └── Spatial occupancy
       ├── Stimulus providers, when supported
       │    ├── Chaser distance
       │    ├── CRA and near-field
       │    ├── Egocentric bearing
       │    └── Future OMR, looming, flash, and other providers
       ├── Statistics
       ├── Recordings and tables
       ├── Provenance
       └── Existing reports
```

The first provider set is capability-based:

- core behavior appears when the relevant generic or epoch summary tables are
  present;
- GoodCopBadCop/chaser panels appear only when their table families are
  present;
- statistics appear only when a correctly linked statistics export exists;
- reports appear only when indexed report manifests exist.

Protocol name may help label or prioritize providers, but it is not sufficient
evidence that a panel's data exists.

## Scientific presentation defaults

- Recording-weighted summaries are the default when recordings are the
  experimental unit.
- Pooled frames, bouts, or histogram counts are labeled explicitly as pooled
  descriptive views.
- Counts are pooled before probabilities or densities are recomputed where the
  table contract requires count-first aggregation.
- Statistical results remain visually separate from exploratory descriptive
  plots.
- Each panel states its aggregation unit, contributing recording count, and
  missing/excluded data count.
- A backend or table change must not silently change the estimand.

## Application states

The UI must distinguish:

- no export selected;
- export manifest missing or invalid;
- export outside the authorized root;
- healthy supported export;
- healthy export with an unsupported table profile;
- expected table absent;
- table present but empty;
- linked statistics absent or stale;
- report inventory absent;
- query or rendering error.

These states should not collapse into a blank plot or generic traceback.

## Code boundary

The deployable Marimo file is a reactive application shell. It owns:

- dataset and panel controls;
- navigation and layout;
- display composition;
- user-facing status and explanations.

Library code under `src/fisheye/` owns:

- export discovery and root confinement;
- registry/catalog queries;
- export and table capability detection;
- scientific filtering and aggregation;
- normalized plot payloads;
- health and provenance payloads;
- report-manifest queries.

Reusable Marimo component modules own panel rendering. Notebook cells should
not hard-code Parquet paths or reimplement statistical calculations.

## V1 scope

V1 includes:

- one authorized export root;
- optional read-only registry/catalog;
- reactive single-export selection;
- overview, health, recordings, tables, and provenance;
- capability-driven core and existing GoodCopBadCop panels;
- correctly linked persisted statistics;
- existing report inventory;
- explicit unsupported and failure states;
- no writes.

V1 defers:

- cross-root or cross-export comparison;
- source-Zarr mutation or analysis execution;
- arbitrary filesystem navigation inside Marimo;
- report generation;
- registry writes;
- user-authored SQL;
- notebook editing or source-code display.

## Implementation sequence before deployment

1. Add read-only export discovery for a selected root, with and without a
   registry snapshot.
2. Add symlink-aware root confinement and manifest/table validation.
3. Replace startup-fixed `export_run_id` with a reactive dataset selector.
4. Add export overview, full health diagnostics, recordings, tables,
   provenance, and report inventory.
5. Define a provider/capability catalog for panels.
6. Extract existing plot construction from the notebook into reusable
   components or normalized renderer inputs.
7. Mount the already-implemented GoodCopBadCop query surface through the
   provider catalog.
8. Add core behavior panels that work without a stimulus-specific table set.
9. Validate aggregation labels and empty/error states with fixtures.
10. Run `marimo check`, unit tests, and a local read-only app smoke.
11. Only then package the installed command and analytics container.

## Acceptance criteria for packaging

The notebook is ready to containerize when:

- two or more exports can be selected without restarting the process;
- a generic/core export and a GoodCopBadCop export route to different valid
  panel sets;
- all resolved files remain beneath the selected root;
- health and provenance failures are understandable in the UI;
- aggregation units are visible;
- no UI action writes to source data or registries;
- the Marimo application passes static checking and focused tests;
- the same arguments work from `scripts/py`, an installed command, and a
  container.
