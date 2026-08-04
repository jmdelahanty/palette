# Eye Trace Samples Export Contract V1

Date: 2026-08-04

Status: implemented non-default query-product contract. It does not activate a
workflow profile, registry entry, selector, Zarr physical profile, or canonical
archive mutation.

## Decision

The recording-local scientific authority remains exactly one maintained
compact-v7 run at:

```text
analysis/eye_angle_runs/<run>
```

`eye_trace_samples` is an immutable Parquet query projection of that authority,
not another eye-angle authority. The exporter requires an explicit completed,
selector-eligible run and never discovers a legacy layout. It records the
selection, completion state, schema identity, source method, frame count, and
digests of every required compact-v7 semantic manifest before reading any
payload.

## Logical Table

The grain is one row per recording, selected eye-angle run, and source camera
frame. The exact primary key is:

```text
(recording_id, source_eye_angle_run, source_acquisition_frame_index)
```

The table contains the fixed export/source envelope, the implicit camera-frame
index, `support/frame_time_seconds`, these twelve float32 channels from
`frame_angles`:

- left, right, and vergence eye angles, raw and smoothed;
- left and right signed gaze, raw and smoothed; and
- mean eye-vergence gaze, raw and smoothed.

It also contains `valid_frame` and `major_axis_marginal` as Arrow booleans and
`reason_codes` as uint16. Frame identity is int64. Scientific angle and time
values remain float32, matching compact-v7. Invalid scientific floats remain
IEEE NaN; they are not converted to Arrow nulls.

The ordered Arrow field set, physical types, and nullability are digest-bound
by `palette.analytics_export.arrow_contracts`. Additional, missing, reordered,
or differently typed fields fail validation. Every primary-key value must be
present and unique.

## Source And Payload Identity

The source binding requires:

- `analysis.eye_angle_runs` schema version 7;
- the `compact_dense_v2` layout;
- exact frame-axis channel catalogs containing every projected channel;
- canonical completion and selector eligibility;
- the parent selection/completion snapshot; and
- canonical JSON digests of the array schema, source contracts, algorithm
  contract, output schema, variant schema, and optional storage plan.

The source binding is computed both before and after extraction. Any change to
selection, completion, schema manifests, method identity, or storage-plan
identity aborts publication.

The projected payload receipt hashes the canonical little-endian decoded bytes
of every scientific column independently. Its aggregate digest does not depend
on extraction batch or Parquet row-group boundaries. Publication validation
reopens the manifest-selected Parquet part, verifies the exact Arrow schema and
constant source fields, proves a contiguous zero-based frame axis, and
recomputes the complete decoded receipt.

All nested manifest objects have exact field sets. Recomputed-digest tampering,
including adding a field inside the source binding, projected payload, Parquet
policy, or semantic projection, fails closed.

## Physical And Publication Policy

The v1 physical policy is:

- one Parquet part per recording;
- default 65,536-row bounded extraction and row groups;
- Zstandard compression level 3;
- dictionary encoding only for declared string columns; and
- construction on explicitly supplied node-local scratch.

The exporter copies the completed part from scratch into a hidden destination
generation, compares the scratch and staged file digests, validates the exact
part inventory and decoded payload, then uses the shared analytics publication
boundary. The immutable generation rename precedes one short locked
compare-and-swap manifest commit. A failed validation or lost manifest race
removes the unpublished generation. Consumers open only the files enumerated by
the manifest; directory globbing is not authoritative.

The generic compact-table exporter rejects `eye_trace_samples`. This prevents
an accidental full-duration in-memory materialization and directs callers to
the bounded streaming implementation.

## Implementation Surface

- `src/fisheye/analytics_exports/eye_trace_samples.py`
- `src/fisheye/utils/export_eye_trace_samples.py`
- `src/fisheye/analysis/eye_angle_io.py`
- `src/fisheye/analytics_exports/contracts.py`
- `src/fisheye/analytics_exports/arrow_contracts.py`
- `src/fisheye/analytics_exports/publication.py`
- `src/fisheye/analytics_exports/validation.py`
- `src/fisheye/analysis_workflows/execution.py`
- `src/fisheye/utils/execute_analysis_workflow.py`
- `scripts/submit_analysis_workflow_bsub.sh`

## Remaining Gates

- Run representative short- and full-duration writer/read/publication
  benchmarks, including node-local scratch, copy, validation, bytes, row-group
  count, CPU, RSS, and manifest-commit timing.
- Exercise the manifest-selected query product through its intended
  cross-recording consumer.
- Keep the exporter opt-in until those gates pass. No production selector or
registry activation is implied by this contract.

The declared workflow node now has an opt-in execution adapter. It binds the
exact selected eye-angle run, fixed framewise policy, immutable export-run ID,
65,536-row policy, publication root, and node-local scratch root. The LSF
wrapper defaults scratch to an execution-specific child of `${TMPDIR}` and
records the effective path. A successful subprocess is accepted only after
full manifest-selected export validation. Parquet exports remain outside Zarr
stage discovery and derived-stage registry projection.

## Validation Evidence

The final focused matrix passed 247 tests covering the eye-trace exporter,
exact Arrow contracts, atomic publication and recovery, maintained eye-angle
I/O, and the established compact cross-recording exporter. Ruff, Python
compilation, and `git diff --check` also passed. Black reformatted the two new
Python implementation/test files successfully; its worker pool did not exit
before the bounded 30-second command timeout, matching the known formatter
behavior in this environment.

The follow-up execution matrix passed 67 tests covering planner/CLI rendering,
LSF `${TMPDIR}` scratch resolution, shell syntax, manifest-selected export
verification, Zarr/export output separation, registry exclusion, malformed
receipt rejection, DAG behavior, and the exporter itself. Ruff, Python
compilation, shell syntax, and `git diff --check` passed for that checkpoint.
