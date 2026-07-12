# Shared Analytics Exports and Cluster Jobs

## Decision

The authoritative Palette cross-recording analytics root is:

```text
/groups/johnson/johnsonlab/palette_analytics
```

New exports are built directly against the recording Zarrs on `/groups` and
written directly to this shared root. `/nvme1` is not part of the publication
contract and is only optional operator scratch.

The shared layout remains the analytics export layout consumed by Palette:

```text
palette_analytics/
├── collections/
├── logs/lsf/
└── v1/
    ├── manifests/
    ├── baseline_behavior_summary/
    ├── baseline_behavior_time_bins/
    ├── baseline_kinematic_samples/     # optional
    ├── chaser_epoch_behavior_summary/
    ├── chaser_epoch_bout_events/
    ├── ...
    ├── group_statistical_summary/
    └── group_descriptive_summary/
```

`v1/` is the physical layout version. Selectable exports inside it use the
strict `palette.analytics_export` schema version 2.

## Completion and failure behavior

An export run ID is immutable. Shared jobs must not use `--overwrite`.

The exporter writes each Parquet part through a temporary file and atomically
renames it. It writes the export manifest only after every requested table has
finished. Marimo discovers datasets from manifests, so a failed job may leave
unreferenced run directories but cannot expose them as a completed dataset.

The cluster job then validates every referenced part:

- manifest and filename identity;
- V2 schema and exact table-contract snapshots;
- Parquet footer schema and table-contract metadata;
- required columns and absence of `benign` legacy columns;
- consistent schemas across all parts;
- Parquet row totals against the manifest; and
- capabilities resolved from the physical Parquet schemas.

Statistics are computed only after the base export passes validation. The
statistics export is then validated by the same command.

## Cluster submission

Render and inspect a job without submitting it:

```bash
scripts/submit_analytics_export_bsub.sh \
  --collection-manifest /groups/johnson/johnsonlab/palette_analytics/collections/example.manifest.json \
  --export-run-id chaser_v2_example_20260712T180000Z
```

Submit after inspecting the generated script and command:

```bash
scripts/submit_analytics_export_bsub.sh \
  --collection-manifest /groups/johnson/johnsonlab/palette_analytics/collections/example.manifest.json \
  --export-run-id chaser_v2_example_20260712T180000Z \
  --submit
```

When `bsub` is unavailable locally, the wrapper follows the Citrus poller
boundary and SSHes only the shell-quoted `bsub` command to
`login1-citrus-poller`. Override this with `--submit-host` or
`PALETTE_LSF_SUBMIT_HOST`. The shared job script already exists before SSH;
the login host does not run the exporter, validator, or statistics process.

Equivalent explicit login-host submission is:

```bash
ssh login1-citrus-poller \
  'bsub ... bash /groups/.../run_analytics_export.sh'
```

Do not SSH to the login host and execute `run_analytics_export.sh` directly.
All data reads, Parquet writes, validation, and statistics must occur inside
the LSF allocation on an execution host.

One CPU LSF job owns the entire collection export. `--ncores` controls both the
LSF CPU request and the exporter's per-recording process pool. This avoids
multiple jobs attempting to publish the same table partitions or manifest.

The generated job captures the exact commit of the shared Palette checkout and
fails if that checkout changes before execution. Submission logs, the rendered
job script, parsed LSF job ID, validation JSON, and completion status are retained below
`palette_analytics/logs/lsf/analytics_export_<run-id>/`.

Registry indexing is optional and occurs only after the base exporter has
successfully written its manifest. Use `--index-registry` when the shared
registry should advertise the completed export.

## Baseline behavior products

Chaser collection exports include two stimulus-independent pre-period tables
by default:

- `baseline_behavior_summary`: one row per recording, track, and canonical
  baseline window. It includes activity, bouts, tracking coverage, arena and
  wall affinity, and normalized spatial/quadrant entropy on a declared grid.
- `baseline_behavior_time_bins`: fixed-duration rows from the start of the
  baseline. The default is 5 seconds and records speed, travelled distance,
  representative arena position, center distance, wall occupancy, bouts, and
  validity.

Both use arena-centered millimetres with image-style axes (`x` right, `y`
down). The exporter resolves the exact epoch-behavior, track-kinematics,
swim-bout, chaser-distance position, and circular arena-geometry sources and
records those paths in every row. It does not independently choose unrelated
latest runs.

`baseline_kinematic_samples` is opt-in because it is much larger. Enable the
portable default 10 Hz representation with:

```bash
scripts/submit_analytics_export_bsub.sh \
  --collection-manifest /groups/johnson/johnsonlab/palette_analytics/collections/example.manifest.json \
  --export-run-id chaser_v2_example_with_baseline_samples \
  --include-baseline-samples
```

Use `--baseline-sample-rate-hz 5` or another positive rate to change the
deterministic integer-frame sampling stride. Use
`--baseline-full-resolution-samples` when every source kinematic sample is
required. The requested rate, effective rate, stride, and sampling policy are
stored in both rows and the export manifest.

Other controls are `--baseline-time-bin-s` and
`--baseline-spatial-grid-size`. Changing any of these settings creates a new
immutable export run; existing exports are not patched.

The sample table is a derived kinematic surface, not raw video. Source Zarr
arrays remain authoritative. Cohort-dependent strategy/cluster assignments do
not belong in these base tables and should be written as a separately versioned
derived analysis that references the immutable export manifest.

## Viewer and deployment

The Marimo application defaults to the shared root. It can be overridden with
`--export-root` or `PALETTE_ANALYTICS_EXPORT_ROOT` for tests and mounted
fixtures. FileGlancer and Apptainer should bind the authoritative shared root
read-only; neither requires access to workstation `/nvme1` storage.
