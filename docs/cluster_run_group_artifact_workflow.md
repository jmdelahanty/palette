# Cluster Run-Group Artifact Workflow
<!-- contract-meta
status: design
last_verified: 2026-05-14
purpose: Define how Palette cluster jobs should produce Zarr-compatible run-group artifacts on node-local scratch and safely import them into canonical analysis archives.
-->

## Purpose

Palette analysis archives are Zarr stores with many chunk and metadata files.
That layout is useful for local partial reads, but direct high-concurrency
mutation over shared storage is a poor fit for cluster jobs that decode large
videos or write many derived arrays.

The cluster-safe pattern is:

1. run compute on node-local scratch;
2. write complete Zarr-compatible output run groups on scratch;
3. validate the scratch result;
4. package the result as an immutable transfer artifact;
5. import it into the canonical analysis Zarr with a serialized importer.

This document defines that workflow. It is a storage and provenance contract,
not a new analysis schema.

For the broader detect, pose, segmentation, and refinement migration checklist,
see `docs/cluster_pipeline_migration_checklist.md`.

## Design Decision

Cluster workers should not write directly into canonical analysis Zarrs on
shared storage during active compute. A worker may read source metadata and
arrays, but the durable mutation of the canonical archive should happen in a
separate import step.

The unit of exchange is a complete run-group package. Examples:

```text
analysis/detect_runs/detect_...
analysis/swim_bout_runs/bouts_...
analysis/bout_kinematics_runs/bk_...
analysis/eye_angle_runs/eye_angle_...
```

The package is immutable after creation. If the run needs to be regenerated,
create a new run name and a new package.

## Janelia Cluster Assumptions

This design is written for the Janelia compute cluster guidance at
`https://hpc.int.janelia.org/docs/ai-agent-hints` (checked 2026-05-14; page
content reports last update 2026-04-22). If that live guidance changes, it
takes precedence over this cached interpretation.

Important environment assumptions:

- Scheduler is IBM Spectrum LSF, not Slurm. Use `bsub`, `bjobs`, and `bkill`
  terminology in job examples.
- Shared storage is NFS-backed PRFS/NRS, not Lustre or GPFS. This reinforces
  the rule that many small concurrent Zarr metadata/chunk writes should not be
  sprayed directly at shared storage.
- Node-local scratch is `/scratch/$USER/`. Cluster jobs should create a
  job-specific scratch directory such as `/scratch/$USER/$LSB_JOBID`.
- `/tmp` on compute nodes is not the correct scratch target.
- Docker is not available for cluster jobs. If containerization is needed, use
  Apptainer and include `--nv` for GPU jobs.
- LSF sets `CUDA_VISIBLE_DEVICES`; Palette job scripts should read it for
  provenance but should not override it by default.
- Jobs should set a wall-time limit with `-W` and write stdout/stderr with `-o`
  and `-e`.
- Python/ML threading variables should be set explicitly from the allocated
  slot count. Use `$LSB_DJOB_NUMPROC` as the starting point, then reduce values
  when the job has layered parallelism such as process pools, Dask workers, or
  PyTorch dataloader workers.

For Palette detection and analysis jobs, the immediate policy is
recording-level parallelism with conservative per-job threading. A job array is
appropriate only when each array task owns distinct recordings and writes
distinct scratch packages.

## Non-Goals

- Do not make tarballs the canonical data format.
- Do not replace per-recording Zarr archives.
- Do not make every analysis stage internally Dask-aware.
- Do not allow many cluster jobs to concurrently update `latest`,
  consolidated metadata, or the SQLite registry.
- Do not use this workflow to hide incomplete or failed runs in canonical
  archives.

## Roles

### Cluster Job

The cluster job owns compute and scratch output. It should:

- stage required inputs to node-local scratch when practical;
- run one recording, or a small independent batch of recordings;
- write output run groups into a scratch Zarr or scratch directory;
- capture full provenance;
- validate the output;
- package the run group and manifest;
- write a job report.

The cluster job must not:

- update canonical parent attrs such as `analysis/<family>.attrs["latest"]`;
- update consolidated metadata on the canonical archive;
- write directly into an existing canonical run group;
- update the registry as the source of truth for the completed run.

### Importer

The importer owns canonical archive mutation. It should run serially, or under a
lock that guarantees one importer per target archive. It should:

- unpack the package into an incoming/staging path on the storage node;
- validate the package again against the current canonical archive;
- verify upstream fingerprints still match;
- atomically promote the run group into its final path when possible;
- update parent attrs such as `latest` only after the run group is complete;
- refresh consolidated metadata when policy requires it;
- update registry projections from the final archive state.

## Package Layout

A run-group artifact should have a single package root:

```text
palette_run_group_artifact/
  artifact_manifest.json
  run_group/
    zarr.json
    ...
  validation/
    strict_json_report.json
    array_presence_report.json
  logs/
    command.log
```

The `run_group/` directory contains the exact Zarr group that will be imported
under the target parent group. For example, if the manifest target is:

```text
analysis/swim_bout_runs/bouts_tk_hyst4_low2_latch_s005_peak_event...
```

then `run_group/` should contain the contents of that `bouts_...` group, not
the entire analysis archive.

Transport format can be `tar.zst`, `tar.gz`, or an unpacked directory. Prefer a
compressed tarball for network transfer because it converts many small Zarr
chunk files into one sequential transfer. The tarball is only a transport
artifact; the canonical store remains Zarr.

## Manifest

Each artifact must include `artifact_manifest.json`. Required fields:

```json
{
  "artifact_schema": "palette_run_group_artifact_v1",
  "created_at": "2026-05-14T00:00:00Z",
  "archive_id": "optional-stable-recording-or-archive-id",
  "target_archive_path": "/path/to/recording_analysis.zarr",
  "target_group_path": "analysis/swim_bout_runs/run_name",
  "run_family": "swim_bout_runs",
  "run_name": "run_name",
  "layout": "compact_tabular_v2",
  "schema_version": "optional-stage-schema-version",
  "source_inputs": [
    {
      "path": "analysis/track_kinematics_runs/offline/tk_...",
      "role": "source_track_kinematics_run",
      "expected_fingerprint": "..."
    }
  ],
  "provenance": {
    "palette_git_commit": "...",
    "command": "scripts/py -m ...",
    "hostname": "...",
    "cluster_job_id": "...",
    "cluster_task_id": "...",
    "cuda_device": "...",
    "decoder_backend": "decord_gpu"
  },
  "timing": {
    "wall_seconds": 0.0,
    "decode_read_seconds": null,
    "inference_seconds": null,
    "write_seconds": null
  },
  "checksums": {
    "run_group_tree_hash": "..."
  },
  "validation": {
    "strict_json": "pass",
    "required_arrays": "pass",
    "upstream_fingerprints": "pass"
  }
}
```

Numeric fields that are not applicable should be `null`, not `NaN` or
`Infinity`.

## Import Algorithm

The importer should use a two-phase import:

1. Unpack to an incoming path:

   ```text
   analysis/<family>/.incoming/<run_name>/
   ```

2. Validate:

   - strict JSON parse of all `zarr.json` files;
   - required arrays and attrs for the declared layout;
   - no unexpected non-finite JSON attrs;
   - source input paths exist in the target archive;
   - expected upstream fingerprints match current upstream fingerprints;
   - final target path does not already exist unless explicit overwrite is
     requested.

3. Promote:

   ```text
   analysis/<family>/.incoming/<run_name>/
   analysis/<family>/<run_name>/
   ```

4. Finalize parent metadata:

   - update parent `latest` only if requested;
   - update run-family indexes if present;
   - refresh consolidated metadata according to current Zarr policy;
   - rescan or update registry rows from the final archive.

If validation fails, leave the incoming directory untouched for diagnosis or
move it to:

```text
analysis/<family>/.failed/<run_name>_<timestamp>/
```

Do not leave failed packages under the normal run-family namespace.

## Latest Policy

The package should declare whether it requests promotion to latest:

```json
{
  "latest_policy": "do_not_set_latest"
}
```

Allowed values:

- `do_not_set_latest`: import the run but leave current defaults unchanged.
- `set_latest_if_newer`: set latest only if validation passes and the importer
  determines this package supersedes the current latest.
- `set_latest_explicit`: set latest after successful import. This should be
  used sparingly and logged.

The cluster job itself never updates latest.

## Registry Policy

The registry is a query index, not the authoritative writer during cluster
compute. Cluster jobs should emit job reports and package manifests. A serial
reconciliation step should update registry projections after import.

This avoids concurrent SQLite writes and keeps the registry rebuildable from
canonical Zarr state plus artifact manifests.

## Concurrency Rules

- Parallelize by recording first.
- Use one writer per target run group.
- Use one importer per target archive.
- Do not run two imports that write the same parent attrs concurrently.
- Do not unpack directly into the final target path.
- Do not share one scratch output directory across jobs unless each job owns a
  unique subdirectory.

Internal Dask remains valid only when the stage obeys
`docs/dask_zarr_write_safety.md`: workers must write disjoint physical chunks
or write temporary outputs that are merged by one owner.

## Failure Handling

Cluster jobs should be safely retryable:

- The output run name should include a deterministic parameter signature and a
  run timestamp or job suffix.
- A failed package must not be imported.
- Retrying should create a new package or overwrite only scratch-local outputs.
- Canonical overwrite requires explicit importer approval and should preserve
  an audit trail.

Importer failures should be recoverable:

- If unpack fails, delete or quarantine the incomplete incoming path.
- If validation fails, quarantine and report.
- If promotion succeeds but metadata finalization fails, rerun finalization from
  the canonical run group rather than recomputing the analysis.

## Relationship To Existing Docs

- `docs/dask_zarr_write_safety.md` defines the low-level chunk-write safety
  rule.
- `docs/cluster_batching_guide.md` defines current cluster batching guidance.
- `docs/zarr_storage_lifecycle_policy.md` defines storage-tier and archival
  policy.
- `docs/analytics_query_layer_design.md` defines how exported analytics should
  query registry-selected Zarr archives and Parquet products.

This document fills the gap between cluster compute and canonical Zarr mutation.

## First Implementation Slice

1. Add a package writer for one run family, preferably detection or swim-bout
   runs.
2. Add a dry-run importer that validates a package and prints the planned
   canonical mutations.
3. Add an apply mode that imports one package into one analysis archive.
4. Add strict JSON and source-fingerprint checks.
5. Add registry reconciliation after import, not during cluster compute.

Detection is the most valuable first target for cluster execution because large
video decode and inference are recording-local and expensive. Swim-bout and
bout-kinematics packages are useful second targets because they exercise compact
tabular run layouts with smaller transfer artifacts.
