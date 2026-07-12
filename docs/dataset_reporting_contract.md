# Dataset Reporting Contract
<!-- contract-meta
status: implemented-foundation
last_updated: 2026-07-11
-->

## Purpose

`fisheye.reporting` plans composable per-recording and cohort reports from the
Palette registry. Every selected recording is considered for the
stimulus-independent `core_behavior.v1` provider. Canonical stimulus-step modes
then activate zero or more stimulus providers.

The reporting layer does not own scientific calculations. Analysis modules own
numeric outputs, and run-local renderers own visualization artifacts. Reporting
owns selection, applicability, concrete run resolution, artifact-contract
validation, status reporting, and eventual report/montage composition.

The scientific hierarchy, stimulus-alignment, trajectory-coordinate,
experimental-unit, and dense-trace export decisions are defined in
[Behavioral Event And Trajectory Analysis Decision](behavior_event_analysis_design_decision.md).
Publication and interactive selection of indexed analytics exports through a
FileGlancer-managed Marimo service are defined in
[FileGlancer and Marimo Integration Design](fileglancer_marimo_integration_design.md).

## Read-only planning

The initial command is strictly read-only with respect to the registry and
recording Zarrs:

```bash
scripts/py -m fisheye.reporting plan \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name GoodCopBadCop
```

Registry connections use SQLite `mode=ro` plus `PRAGMA query_only=ON`. Zarrs
are opened with mode `r`. The plan is emitted to stdout as
`palette.dataset_report_plan.v1`; the command does not persist a plan, create
analysis runs, or render artifacts.

Catalog inspection is also read-only:

```bash
scripts/py -m fisheye.reporting list --kind providers
scripts/py -m fisheye.reporting list --kind visualizations
```

An explicit cohort selector is required unless `--all-recordings` is passed.
Exact recording IDs, path filters, protocol names, and limits can be combined.

Execution is deliberately separate from planning. Only allowlisted actions run,
and each mode must be opted into explicitly:

```bash
scripts/py -m fisheye.reporting apply \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name GoodCopBadCop \
  --render-missing

scripts/py -m fisheye.reporting apply \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name GoodCopBadCop \
  --apply-analysis
```

Contract-mismatch refreshes additionally require
`--refresh-contract-mismatches`. Unsupported planned actions are reported as
unsupported; the executor does not dynamically import arbitrary action names.

## Declarative objects

The static in-library catalog uses three main declarations:

- `AnalysisFamilySpec`: logical stage identity, candidate run parents,
  prerequisite analysis families, upstream source requirements, and optional
  entity identity attrs.
- `VisualizationSpec`: semantic visualization ID, source analysis family,
  entity scope, artifact selector, and expected visualization contract,
  renderer, and renderer version.
- `ProviderSpec`: a versioned collection of visualizations plus an
  applicability rule.

The initial provider IDs are:

- `core_behavior.v1` (always applicable)
- `stimulus.chaser.v1` (canonical `CHASER` steps)
- `stimulus.moving_grating.v1` (canonical `MOVING_GRATING` steps)
- `stimulus.concentric_grating.v1` (canonical `CONCENTRIC_GRATING` steps)
- `stimulus.looming.v1` (canonical `LOOMING_DOT` steps)
- `stimulus.flash.v1` (canonical `DARK_FLASH` or `BRIGHT_FLASH` steps)

Protocol name selects a cohort. It does not determine scientific
applicability. A mixed protocol may activate multiple providers from its
canonical stimulus steps.

## Dynamic cardinality

Visualization declarations state their entity scope. The planner currently
supports recording, track, stimulus-step, and chaser scopes. Track IDs are
read from the resolved track-kinematics run. Configured chasers are read from
the resolved stimulus run's `protocol_json` and retain their canonical
aggressive, random-non-chasing, or inert behavior labels.

Variable numbers of subjects, steps, and chasers expand into multiple plan
items; they are not stored as fixed database columns.

## Plan statuses

Each requested visualization receives exactly one status:

- `ready`: the concrete source run and matching contracted artifact exist.
- `needs_render`: the analysis exists but the semantic artifact is absent.
- `needs_analysis`: prerequisites exist but the analysis run is absent.
- `contract_mismatch`: an artifact exists but uses a missing or different
  visualization contract, renderer, or renderer version.
- `blocked_missing_source`: an upstream source or prerequisite analysis is
  absent.
- `not_applicable`: an explicitly requested provider does not match the
  recording's canonical stimulus modes.
- `error`: the recording could not be inspected read-only.

Items also contain proposed action identifiers such as
`analyze:core.swim_bouts`, `render:core.position.xy_trace`, or
`resolve_source:refined_subject_masks`. These are plans only; the read-only
command never executes them.

## Run and artifact identity

Run resolution is per recording. The plan freezes the concrete run ID, Zarr
path, schema/method metadata, lineage/fingerprint fields, selection policy, and
entity identity where present.

Artifact references record the concrete path, content hash, artifact
signature, visualization contract, renderer, and renderer version. Nested
run-local artifacts are discoverable through path patterns, allowing chaser
component visualizations to retain their specialized physical layout.

Numeric arrays remain authoritative. PNG artifacts are contracted review and
report snapshots.

## Semantic montages

Montages select artifacts by semantic visualization ID, not by hard-coded Zarr
paths. Only `ready` artifacts with the declared visualization contract are
loaded by default:

```bash
scripts/py -m fisheye.reporting montage \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name GoodCopBadCop \
  --visualization-id stimulus.chaser.egocentric_bearing \
  --output-dir /path/to/montages
```

`--allow-nonready` renders labeled placeholders and records every omission in
`semantic_montage_manifest.json`.

## Normalized stimulus registry

Registry schema 61 adds reusable `stimulus_protocols` and
`stimulus_protocol_steps` tables. Recording-specific cardinality remains in
the child tables `recording_stimulus_runs`, `recording_stimulus_steps`, and
`recording_stimulus_modes`. The `recording_stimulus_mode_counts` view provides
one row per dataset, stimulus run, and canonical mode, including step count and
total known duration. This supports protocol and mode cohort queries without
fixed columns for particular stimulus families.

## Immutable report exports

The export command creates a new directory atomically and refuses to overwrite
an existing one. A `reference` export retains exact Zarr paths; a `copy` export
verifies each declared PNG content hash and writes portable copies:

```bash
scripts/py -m fisheye.reporting export \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name RedScare \
  --visualization-id stimulus.chaser.egocentric_bearing \
  --materialization copy \
  --output-dir /path/to/report-redscare-v1
```

`report_manifest.json` embeds the complete frozen report plan, concrete source
runs and artifacts, contracts, renderers, materialization policy, non-ready
items, optional source-collection-manifest hash, and its own verifiable
`manifest_sha256`.

To bind a report to an indexed analytics export, provide an explicit report
identity and let the command derive its canonical immutable location:

```bash
scripts/py -m fisheye.reporting export \
  --registry /path/to/palette_registry.sqlite \
  --protocol-name RedScare \
  --analytics-export-run-id export_20260711_001 \
  --report-id redscare-chaser-overview-v1 \
  --visualization-id stimulus.chaser.egocentric_bearing \
  --materialization copy \
  --index-registry
```

The binding records the analytics export manifest path and its byte-level
SHA-256, collection identity, and the export's available table names. Existing
Zarr visualization snapshots remain labeled `source_backend: zarr`; being
co-located with Parquet does not relabel them as Parquet-derived.

## Zarr-backed and Parquet-backed visualization

Palette supports two complementary visualization source backends:

- recording-local Zarr runs for authoritative dense arrays, frame traces,
  trajectories, geometry, diagnostic state, and existing contracted snapshots;
- indexed analytics Parquet exports for cohort distributions, protocol or
  genotype comparisons, group statistics, and other cross-recording queries.

This does not imply two independent implementations of every plot. A semantic
visualization should use a source adapter to construct a normalized render-data
object, then pass that object to a shared renderer whenever the scientific
meaning is identical:

```text
Zarr recording adapter ----\
                            +--> normalized render data --> shared renderer
Parquet export adapter ----/
```

Visualization IDs describe the scientific question and aggregation scope, not
the storage backend. For example, a recording-level swim-bout summary and a
recording-weighted protocol comparison are different contracts even if both
contain histograms. Suitable IDs distinguish scopes such as
`core.swim_bouts.recording_summary` and
`cohort.swim_bouts.protocol_comparison`.

Each visualization declares a backend capability of `zarr`, `parquet`, or
`both`. Supporting both is optional and should only be done when both adapters
can satisfy the same normalized render-data contract. Backend-equivalence
fixtures should verify that equivalent Zarr and Parquet inputs produce the same
normalized payload before they share a renderer.

Every rendered artifact records:

- `source_backend` (`zarr` or `parquet`);
- concrete Zarr run and array paths, or analytics `export_run_id` and table
  names;
- source collection and export manifest hashes;
- cohort filters and query parameters;
- aggregation unit, such as frame, bout, fish, recording, or protocol;
- visualization contract, renderer, and renderer version.

A backend must not silently change scientific semantics. In particular,
pooling every bout across a cohort, summarizing fish equally, and summarizing
recordings equally are different estimands and require explicit aggregation
metadata and, where their interpretation differs, separate visualization
contracts.

## Co-locating reports with analytics exports

Cross-recording analytics exports use a table-first directory layout, so one
export run is spread across multiple Parquet table partitions. Reports should
therefore be co-located under the same analytics root as an immutable sibling,
not written inside individual Parquet table directories:

```text
palette_analytics/v1/
  manifests/export_run_id=<export_run_id>.json
  <table>/export_run_id=<export_run_id>/part-*.parquet
  reports/
    export_run_id=<export_run_id>/
      report_id=<report_id>/
        report_manifest.json
        montages/
        artifacts/
```

The report manifest binds `report_id` to the analytics export manifest and
collection hashes. It also distinguishes copied original-Zarr visualizations
from plots actually computed from exported Parquet. Physical co-location alone
must never imply that an image was derived from the Parquet tables.

Registry schema 62 implements the report index as a one-to-many child of
`analytics_exports`. `analytics_reports` stores report identity, location,
status, manifest hash, materialization policy, and source backends/tables.
`analytics_report_visualizations` stores one compact summary per semantic
visualization. Detailed tile lineage remains in `report_manifest.json`; the
registry does not create one row for every montage tile.

Existing manifests can be verified and indexed, and the resulting catalog can
be queried without opening recording Zarrs:

```bash
scripts/py -m fisheye.reporting index-report \
  --registry /path/to/palette_registry.sqlite \
  --manifest /path/to/report_manifest.json

scripts/py -m fisheye.reporting query-reports \
  --registry /path/to/palette_registry.sqlite \
  --export-run-id export_20260711_001 \
  --visualization-id stimulus.chaser.egocentric_bearing

scripts/py -m fisheye.reporting check-report \
  --manifest /path/to/report_manifest.json \
  --check-files
```

## Analytics export grain and authority

The current default Parquet exports contain multiple row grains:

- one row per recording (`recording_summary`);
- one row per stimulus step (`stimulus_steps`);
- one row per fish and stimulus step (`stimulus_step_summary` and
  `stimulus_response_per_fish_step`);
- one row per detected swim bout (`swim_bout_metrics`);
- one row per bout and measurement level (`bout_kinematics_metrics`).

The GoodCopBadCop export family additionally contains epoch, object-phase,
histogram-bin, radial-density, distance-CDF, and egocentric
distance-by-bearing-bin rows. These preserve substantially more than final
figure summaries, but they remain derived analysis tables.

Full frame-by-frame speed, heading, X/Y/Z position, eye-angle, convergence,
object-position, and other dense traces are not part of the default analytics
export. Their authoritative representation remains in the recording Zarr.
Future trace exports should use explicit long-form row axes or documented
fixed-size array columns, include time/frame identity and source lineage, and
be added selectively because frame-level cohort exports can be much larger
than event-level tables.

## Implemented artifact contracts

Core contracts cover track overview, X/Y traces, run-local swim-bout summaries,
bout movement and heading, eye angle/convergence, and neutral full-session
occupancy. Swim-bout summary runs persist their histogram edges, counts, and
fractions under `report_tables/swim_bout_summary` so the rendered distribution
can be reproduced without inferring bins from PNG pixels. Chaser distance,
distance distributions, and egocentric-bearing artifacts and moving-grating
OMR artifacts also have stable contracts. Concentric-grating, looming, and
flash providers are registered against canonical stimulus modes; their report
items remain explicit `needs_analysis`, `needs_render`, or `unsupported` until
a matching contracted artifact/executor exists.
