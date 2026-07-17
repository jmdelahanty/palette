# Analytics Storage Reconciliation — 2026-07-17

<!-- contract-meta
status: diagnostic
last_verified: 2026-07-17
-->

This report records the read-only inventory that preceded the analytics-storage
reconciliation and the disposition of the surfaces it found. The current
contract is the
[analytics storage schema matrix](../analytics_storage_schema_matrix.md); this
file is dated audit evidence, not a second authority.

## Inventory result

Palette has seven current recording-local derived analytics families:

1. track kinematics
2. swim-bout segmentation
3. bout kinematics
4. eye angles and gaze
5. subject shape
6. tail kinematics
7. stimulus response

Visualization artifacts are projections owned by those scientific runs, not an
eighth scientific authority. Cross-recording Parquet datasets are immutable
query/export products and remain outside the recording Zarr authority.

The implemented families already converged on immutable named runs, explicit
source lineage, logical readers, compact or semantic layouts, and node-local
materialization. The remaining contract gaps found by the audit were:

- track-kinematics runs had no family-level `schema_id`, `schema_version`,
  `method_version`, or declared row axis;
- production swim-bout and stimulus-response commands could write directly into
  visible authoritative run paths;
- several active documents still described historical layouts, direct PRFS
  writing, Zarr-local Parquet sidecars, or compact-mask authority;
- dated implementation plans and completed migration TODOs were mixed with
  active contracts.

## Disposition

### Retained as current authority

- [analytics storage schema matrix](../analytics_storage_schema_matrix.md)
- [derived analysis run contract](../derived_analysis_run_contract.md)
- [analysis workflow DAG](../analysis_workflow_dag.md)
- [Zarr run completion contract](../zarr_run_completion_contract.md)
- [storage lifecycle policy](../zarr_storage_lifecycle_policy.md)
- [cross-recording analytics export design](../cross_recording_analytics_export_design.md)
- [`zarr_structure.md`](../../src/fisheye/docs/zarr_structure.md)
- each family's scientific contract and logical reader

### Archived, not deleted

Eighteen superseded documents were moved into `docs/archive/` with an archive
banner and repaired backlinks. They comprise:

- the dated 2026-07-05 handoff;
- completed dense-array, tabular-identity, object-count, NFS, transfer, and
  provenance migration plans;
- obsolete analysis-Zarr creation and post-detection status documents;
- the rejected Zarr-local Parquet-sidecar design;
- the superseded analytics-query and exported-artifact snapshots;
- the historical movement online/offline plan and track/bout status snapshot.

Nothing was permanently culled. These files still contain useful chronology,
benchmarks, or rejected alternatives, but they are no longer presented as
current instructions.

### Contract gaps closed

- Track kinematics now writes and validates `analysis.track_kinematics_runs`
  schema version 1, `track_kinematics.v1`, `track_samples`, parameters, and exact
  source references.
- Swim-bout schema-8 compact runs now expose the common derived-run fields and
  are computed in disposable local Zarr storage before atomic publication.
- Stimulus-response schema-2 compact runs now expose the common fields and use
  the same publication transaction.
- Production DAG and moving-grating/batch entry points route to the materialized
  writers. Direct writers remain low-level disposable/test surfaces.

## Deliberately retained compatibility

The audit does not recommend deleting historical Zarr data. Logical readers
continue to recognize documented legacy layouts, including schema-7 embedded
swim-bout frame axes, hierarchical swim-bout/stimulus-response tables, and
metadata-declared eye-angle variants. New production runs use only the current
matrix defaults; older recordings can be regenerated when strict current
contracts are required.

## Validation evidence

- modified-document relative-link audit: zero missing links;
- production invocation audit: no direct swim-bout or stimulus-response writer
  calls outside the low-level modules themselves;
- focused materializer, workflow, frame-axis, logical-reader, track-kinematics,
  stimulus-response, and batch-pipeline tests: 179 passed;
- broader publication/completion/lineage suite: 106 passed and one repository-
  wide raw-parent guard failed on five pre-existing utilities; the identical
  guard failure was reproduced on the base checkout;
- Python compilation, shell syntax, and `git diff --check`: passed.
