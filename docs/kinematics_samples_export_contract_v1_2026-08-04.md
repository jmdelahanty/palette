# Generic Kinematics Samples Export Contract V1

Date: 2026-08-04

Status: implemented non-default query-product contract. It does not activate a
workflow profile, registry entry, selector, Zarr physical profile, or canonical
archive mutation.

## Decision

The recording-local scientific authority remains one completed canonical run
at:

```text
analysis/track_kinematics_runs/<online|offline>/<run>
```

`kinematics_samples` is a portable immutable Parquet projection, not another
track-kinematics authority. It is deliberately independent of
`baseline_kinematic_samples`, whose 71-field grain and semantics are
chaser/baseline specific.

The generic export requires an explicit completed and selector-eligible track
run with the exact full-motion publication manifest/commit and physical-mm
authority. Historical inspection layouts, pixel-only runs, implicit `latest`
fallback, and chaser-specific fields are forbidden.

## Logical Table

The grain is one row per recording, track-kinematics run, track, and selected
acquisition frame. Its exact primary key is:

```text
(recording_id,
 source_track_kinematics_scope,
 source_track_kinematics_run,
 track_id,
 source_acquisition_frame_index)
```

The ordered 45-field Arrow schema contains exact source/run identity, sampling
policy, physical-coordinate authority, track/sample identity, nullable source
instance lineage, physical position, filtered speed/path distance, motion
heading, smoothed motion heading/angular velocity, and exact validity/reason
fields.

Scientific representations preserve the maintained source contract:

- physical positions are float32 millimetres, exactly projecting the
  maintained source authority without export-only widening;
- time, speed, distance, heading, and angular velocity are float32;
- frame, source-row, track, and source-sample identities are int64;
- nullable `instance_key` is uint64 plus a separate boolean validity field;
- detection source is int8 and reason codes are int16; and
- invalid scientific floats remain IEEE NaN rather than Arrow nulls.

The heading fields are explicitly motion headings. Body heading remains owned
by the separate body-frame authority and is not silently duplicated here.
Scientific consumers may upcast positions to float64 for calculations;
aggregate activity/spatial statistics remain float64 outputs.

## Sampling

Sampling is aligned globally to acquisition-frame identity:

```text
stride = max(1, floor(source_fps / requested_rate_hz + 0.5))
select frame when source_acquisition_frame_index % stride == 0
nominal_rate_hz = source_fps / stride
```

This produces stable rows regardless of source read-window or Parquet
row-group boundaries. Sampling is not restarted for each track, clip, or
worker. At 700 FPS and a requested 10 Hz it selects a 70-frame stride.

## Source Binding

The exporter verifies without performing another full 69-array payload walk:

- the exact track-run schema, explicit scope/run path, completion, and
  selector eligibility;
- the closed v1/v2 full-motion manifest and its canonical digest;
- the exact publication commit reconstructed from that manifest;
- the complete ordered track inventory and each track-record digest;
- exact manifest records and live dtype, shape, and attribute declarations for
  the 20 selected surfaces;
- one common `physical_mm` coordinate descriptor and physical-authority digest;
  and
- parent/scope selector and completion snapshots.

The binding embeds the selected surface record/content digests already proven
by track publication. Payload extraction then reads only bounded first-axis
windows. The complete source binding is recomputed from a fresh direct-metadata
open after extraction; any source, selector, completion, manifest, commit,
track, or declaration change aborts before visibility.

Within each bounded window the exporter also proves exact track keys,
strictly increasing acquisition frames, frame-derived time values, structured
source-instance null semantics, and position-finite flags.

## Physical And Publication Policy

The v1 query-product policy is:

- one Parquet part per recording containing all tracks in ascending ID order;
- default 131,072-row source windows and 65,536-row Parquet row groups;
- Zstandard compression level 3;
- dictionary encoding only for declared strings; and
- construction on an explicit non-overlapping node-local scratch root.

Decoded little-endian bytes for all 23 variable scientific columns are hashed
independently. The exporter copies the complete scratch part into a hidden
destination generation, verifies file and decoded receipts, and commits only
through the shared manifest-exclusive compare-and-swap publication boundary.
Consumers may open only the manifest-enumerated part. The generic compact
exporter rejects this table so callers cannot accidentally use its unbounded
in-memory path.

## Workflow Boundary

The existing `kinematics_samples` workflow node now renders the dedicated
publisher with:

- its exact `track_kinematics` dependency run;
- offline scope;
- the versioned temporal-policy sample rate;
- an explicit immutable export-run ID and publication root; and
- an explicit node-local scratch root.

The LSF wrapper requires `--export-root` for this target and otherwise selects
an execution-specific child of `${TMPDIR}` for scratch. A successful subprocess
is accepted only after the complete manifest-selected export validates.
Parquet query products remain outside Zarr stage discovery and derived-stage
registry projection.

## Implementation Surface

- `src/fisheye/analytics_exports/kinematics_samples.py`
- `src/fisheye/utils/export_kinematics_samples.py`
- `src/fisheye/analytics_exports/contracts.py`
- `src/fisheye/analytics_exports/arrow_contracts.py`
- `src/fisheye/analytics_exports/validation.py`
- `src/fisheye/analysis_workflows/execution.py`
- `scripts/submit_analysis_workflow_bsub.sh`

## Remaining Gates

- Benchmark short and full-duration multi-track writer, bounded read, scratch,
  copy, validation, publication, file bytes, CPU, RSS, and manifest timing.
- Exercise the exact manifest-selected table through its intended portable
  cross-recording consumer.
- Keep the exporter opt-in and retain recording-local track Zarr as rollback
  authority until those gates pass.

## Validation Evidence

The final focused export, atomic-publication, workflow, DAG, registry, and LSF
matrix passed 307 tests. Dedicated coverage proves deterministic stride
selection, exact Arrow types, batch-boundary-independent decoded receipts,
multiple tracks sharing acquisition-frame values without key collisions,
selected and unsampled source-byte verification, source changes before
visibility, rehashed nested/semantic/constant-column tampering, failed
replacement recovery, physical-mm authority requirements, and non-overlapping
scratch/publication roots.

Ruff, Python compilation, shell syntax, and `git diff --check` passed. A bounded
Black check did not finish within 30 seconds and reported that it would
reformat seven pre-existing touched modules/tests. That unrelated broad
formatting rewrite is deliberately excluded from this semantic checkpoint.
