# Exact tabular candidate read benchmark contract — 2026-08-03

Status: implemented diagnostics contract for selector-ineligible
`published_http_v1` candidates. It does not promote a storage profile, update a
selector, register a run, or authorize writes to an analysis archive.

## Scope

The runner compares one explicit authoritative source with one explicit
selector-ineligible byte-planned candidate for either:

- `analysis/swim_bout_runs/<run>`; or
- `analysis/bout_kinematics_runs/<run>`.

The two families use the same runner but remain separate benchmark matrices so
their different logical tables, shapes, and access classes are not blended into
one timing summary.

Implementation:
`fisheye.diagnostics.benchmark_exact_tabular_candidates`.

## Fail-closed inputs

Every matrix invocation requires:

- the exact archive path;
- one family ID;
- an explicit source run name;
- an explicit candidate run name;
- a caller-supplied benchmark-only output directory; and
- a truthful cache-state description.

`latest`, `latest_complete`, path separators, missing completion markers, an
ineligible source, an eligible candidate, and mismatched candidate-source
bindings are rejected. The candidate must carry:

- the byte-planner-adopted exact array manifest;
- `analysis_storage_profile_role=explicit_unpromoted_candidate`;
- a complete executable `analysis_storage_plan_receipt`; and
- the exact source-run binding minted by the candidate materializer.

The source and candidate inventories must have identical logical paths,
dtypes, shapes, and array contracts before any measured subprocess starts.

## Benchmark-suite binding

The runner reconstructs the candidate storage receipt with
`analysis_storage_plan_receipt_from_manifest()`. It then builds and deeply
validates one `palette.analysis_storage_benchmark_suite` using:

- the receipt's observed dimensions;
- the receipt's exact array declarations and physical plans;
- deterministic seed 17 by default; and
- five repetitions by default.

Every trial records both the benchmark-suite payload digest and candidate
storage-receipt payload digest. Candidate trials independently require their
persisted receipt to equal the suite-bound receipt.

Primary reads execute the suite's frozen access-class selections:

- `EAGER`: whole array;
- `WINDOWED`: bounded contiguous ranges;
- `PER_ROW`: deterministic complete rows; and
- `INDEXED`: deterministic indexed rows.

The current compact schemas do not expose one common CSR `ptr/len` index for
all tables. Accordingly, the v1 `INDEXED` adapter reads the suite-selected
complete table rows by deterministic row position and records
`indexed_resolution=deterministic_complete_table_rows`. It does not claim that
those row reads exercise a nonexistent common range-index object.

The runner also performs a deterministic blocked full scan of every declared
array. Full-scan dtype, shape, and logical digests must match between source and
candidate in every repetition.

### Known suite-v1 selection limitation

`analysis_benchmark_suite` v1 derives its deterministic selection extent from
`logical_shape[0]`. The swim-bout detector-signal matrix grows along frame axis
1. This runner executes the frozen range values on the receipt-declared growth
axis and records both `execution_axis` and
`suite_v1_selection_extent_source=logical_shape_axis_0`. Consequently the
detector-signal primary window is deterministic but smaller than the intended
frame-window workload. Full scans are unaffected. Fixing selection generation
belongs to a future shared-suite version and must not be hidden in this
diagnostics adapter.

## Fresh-process matrix

The controller launches two new Python processes per repetition: one source
trial and one candidate trial. Their order alternates deterministically by
repetition. The default five-repetition matrix therefore produces ten trial
documents. An overridden repetition count is useful for integration tests, but
the result records `balanced_read_matrix_complete=false` unless it used all
five. Even a complete balanced read matrix is not profile-promotion evidence:
this adapter does not measure writer/publication phases, physical transfer,
representative short/full scales, or real consumers.

Fresh processes isolate Python/Zarr decoded state and process RSS. They do not
clear the operating-system or mounted-filesystem cache. Trial order rotation
balances that uncontrolled effect; the caller's cache-state declaration makes
it explicit rather than claiming a cold-cache guarantee.

## Measured evidence

Each trial records:

- direct metadata open wall and CPU time;
- consolidated metadata open wall and CPU time;
- direct/consolidated run-attribute and array-declaration comparison time;
- exact-manifest and candidate-receipt validation time;
- every primary access-class read, selection, decoded bytes, digest, wall time,
  and CPU time;
- every full scan, decoded bytes, digest, wall time, and CPU time;
- run-local file, metadata-file, and payload-object counts;
- apparent and allocated filesystem bytes; and
- initial and final process peak-RSS high-water marks.

The controller reports per-role medians and retains every raw trial envelope.

Physical request counts and transferred bytes are explicitly `null` with the
reason `unavailable_without_os_or_filesystem_tracing`. Logical decoded bytes,
file counts, apparent bytes, and allocated bytes are not relabeled as physical
transfer telemetry. A later PRFS/SMB/HTTP experiment must add OS, filesystem,
or driver tracing if those metrics are required.

## Metadata and read-only proof

Published immutable inputs must open through both direct and consolidated
metadata, and their run attributes plus all exact array declarations must
normalize identically. Missing or stale consolidated metadata is a benchmark
failure, not a reason to fall back permanently to direct traversal.

Before and after all subprocesses, the controller hashes the exact root,
family-parent, source-run, and candidate-run `zarr.json` files. Any difference
fails the matrix. All Zarr handles use read-only mode.

## Output safety and evidence layout

The output directory:

- must not already exist;
- must be outside and not contain the source archive;
- must contain a path component identifying it as benchmark-only; and
- is created exclusively by the matrix controller.

It contains:

```text
<output>/
  analysis_benchmark_suite.json
  matrix_result.json
  trials/
    rep_00_pos_0_<role>.json
    rep_00_pos_1_<role>.json
    ...
```

Every document is strict JSON with closed schema fields and a recomputed
payload SHA-256. Trial output uses an atomic temporary-file rename and refuses
replacement.

## Example

```bash
scripts/py -m fisheye.diagnostics.benchmark_exact_tabular_candidates matrix \
  /path/to/recording_analysis.zarr \
  --family swim_bouts \
  --source-run explicit_source_v8 \
  --candidate-run explicit_http_candidate_v1 \
  --cache-state mounted_prfs_uncontrolled_os_cache \
  --output-dir /tmp/.palette_benchmarks/swim_bouts_read_20260803
```

Run bout kinematics with a second output directory and
`--family bout_kinematics`. Do not point the output at `/groups` until a
separate benchmark publication action is authorized.

## Checklist

- [x] Explicit immutable source and candidate names; aliases rejected.
- [x] Source exact-manifest validation.
- [x] Candidate exact-manifest and executable receipt validation.
- [x] Suite-bound deterministic primary selections and full scans.
- [x] Direct/consolidated metadata comparison.
- [x] Source/candidate logical equality in every repetition.
- [x] Rotated fresh-process order with five repetitions by default.
- [x] Wall, CPU, peak RSS, object, apparent-byte, and allocated-byte evidence.
- [x] Strict JSON, immutable benchmark output, and read-only metadata guard.
- [x] Explicitly unavailable physical request/transfer telemetry.
- [ ] Run a full-duration matrix on an authorized benchmark-only copy.
- [ ] Add OS/filesystem tracing for physical request and transfer attribution.
- [ ] Version the shared suite's nonzero-growth-axis selection generation.
