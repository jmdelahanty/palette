# Tail-posture-view v3 source/candidate read benchmark contract

Date: 2026-08-03  
Status: implementation candidate; selector/profile promotion is forbidden  
Family: `analysis/tail_posture_view_runs`  
Benchmark: `tail_posture_view_v3_source_candidate_read_matrix_v1`

## Decision

Palette needs one executable read matrix for the maintained tail-posture-view
v3 surface before its byte-planned physical profile can be considered for
promotion. The matrix compares one already-selected source publication with
one explicitly named, complete, selector-ineligible candidate. It is
read-only. It cannot mutate the archive, move family selectors, update a
registry, or authorize a profile.

This family does **not** have a standalone public payload reader. Both payload
adapters in the benchmark are diagnostic-only. Evidence is separated into the
following honest boundaries:

- `megabouts_classifier_inputs._resolve_tail_posture_view_run` is a maintained
  but private/internal source selection boundary. It is not described as a
  public API.
- `load_tail_posture_coordinate_publication` is the public validator for the
  selected source's coordinate publication. It validates coordinate and
  lineage authority; it is not a table reader.
- The selector-ineligible candidate is opened only by the benchmark's exact
  diagnostic payload validator. Its coordinate publication is checked with
  the low-level ineligible-publication validator and is not presented as a
  consumer surface.
- `build_megabouts_classifier_input_pack` is the real broader consumer. It
  additionally requires matching track-kinematics and swim-bout authorities.
  That end-to-end gate remains open rather than being simulated by this
  family-local fixture.

## Exact logical surface

The source and candidate must both implement
`analysis.tail_posture_view_runs` schema v3 with exactly ten direct arrays.
Every declaration is required, immutable, and classified `windowed`.

| Array | Exact dtype and shape | Meaning |
| --- | --- | --- |
| `instance_key` | `uint64[n_rows]` | Unique upstream observation identity. |
| `source_crop_row_ids` | `int64[n_rows]` | Canonical crop-row identity. |
| `source_acquisition_frame_index` | `int64[n_rows]` | Acquisition-camera frame identity. |
| `valid` | `bool[n_rows]` | Row-level payload validity. |
| `failure_reason_bytes` | `uint8[n_rows,64]` | NUL-terminated UTF-8 followed by zero padding. |
| `head_xy` | `float32[n_rows,2]` | Source-camera pixel point. |
| `head_yaw_rad` | `float32[n_rows]` | Source-camera image-XY heading in radians. |
| `tail_keypoints_xy` | `float32[n_rows,n_keypoints,2]` | Tail-base-to-tail-tip source-camera pixels. |
| `tail_angle_rad` | `float32[n_rows,n_angles]` | Megabouts cumulative segment angle in radians. |
| `tail_angle_deg` | `float32[n_rows,n_angles]` | The same cumulative segment angle in degrees. |

`n_keypoints >= 2` and `n_angles == n_keypoints - 1`. An invalid row has
`valid=false`, a nonempty reason, and NaN in every floating posture payload.
A valid row has reason `ok`. The fixed-width reason array is a logical schema
choice, not a physical chunk-size convention.

## Source and candidate lifecycle

The source is created through the guarded-direct writer without a storage
profile. It is sealed with the tail-coordinate publication, marked complete,
then atomically made selector-eligible. For the benchmark fixture, all three
family pointers must exist and name that source:

- `latest`
- `latest_complete`
- `latest_megabouts_compatible`

The candidate uses the same guarded-direct writer with
`published_http_v1`. Its ten physical plans are derived from uncompressed
bytes and persisted in the executable analysis-storage receipt. The candidate
is sealed and complete, but remains literally selector-ineligible. None of the
three pointers may name it. The benchmark freezes the selector snapshot before
the first read and proves that it is unchanged afterward and after the full
matrix.

The logical source and candidate payloads must be identical in path, exact
dtype, exact shape, and contiguous bytes. That byte digest comparison is
NaN-safe because matching IEEE payload bits are required rather than treating
NaN as unequal.

The candidate must identify `published_http_v1` both in its run attributes and
in the parsed storage receipt. The complete profile manifest must round-trip
to the executable profile, and all ten parsed declarations must exactly match
the maintained byte-planned v3 declarations. Matching an outer receipt digest
or merely carrying ten entries is insufficient.

The pair also freezes a storage-independent scientific identity. It includes
method/version, view family, head source, subject-shape/refined-mask/optional
tail-run references, source manifest digest, geometry kind, keypoint and angle
counts/order/convention/units, copied lineage policy, source references,
algorithm provenance, and reason encoding. It deliberately excludes storage
plans, publication-owner UUIDs, lifecycle selectors, timestamps, runtime
provenance instances, and other fields that should differ across physical
publications.

## Required validation

Before any timing is accepted, the benchmark fails closed unless it proves:

- canonical, nonsymlink archive, run, and output paths;
- explicit immutable source and candidate names (no `latest` aliases);
- exact schema id/version, ten-array inventory, dimensions, semantic fills,
  and array-schema manifest/digest;
- absence of a candidate receipt on the source;
- executable reconstruction of the candidate byte plan and exact physical
  metadata/fill agreement;
- strict completion plus source eligibility and candidate ineligibility;
- source selection through the maintained private resolver;
- public coordinate-publication validation of the source;
- exact completed-ineligible coordinate-publication validation of the
  candidate;
- identical source-subject-shape authority and manifest digest;
- direct/consolidated metadata equivalence for the complete family subtree,
  including parent selectors, both runs, all coordinate-record groups, every
  codec, chunk, shard, dtype, shape, fill, and attribute declaration;
- bit-exact decoded equality for all ten arrays; and
- hard nonpromotion plus null physical-I/O fields unless an external trace is
  actually supplied by a future contract revision.

Executable row semantics additionally require unique `instance_key`,
nonnegative crop/frame lineage, strict UTF-8 with a NUL terminator and only
zero padding in all 64-byte reasons, `ok` plus finite floating payloads on
valid rows, a non-`ok` reason plus all-NaN floating payloads on invalid rows,
and float32 `rad→deg` agreement using `rtol=1e-6` and `atol=1e-5`. The fixture
contains both a valid and invalid row so these rules are exercised rather than
remaining declarations.

## Workload and process matrix

The deterministic workload covers every array with:

1. one complete eager read; and
2. a configurable number of first-axis row-window reads.

Every operation freezes path, mode, first-axis span, exact expected dtype, and
exact expected result shape. Every receipt records operation index, dtype,
shape, element count, decoded bytes, bit digest, wall time, and CPU time. The
validator rebuilds the workload from the live pair and replays each logical
receipt. Recomputing only the outer JSON digest cannot legitimize changed
spans, roles, arrays, storage plans, selectors, coordinate bindings, or
payload digests.

The controller alternates source-first and candidate-first order. Each role in
each repetition runs in its own fresh child process, so five repetitions mean
ten distinct child PIDs rather than five processes that share decoded state
between the roles. The default uses 4,096-row windows and four windows per
array. Each child receives the shared single-thread benchmark environment and
emits strict JSON. PIDs must be unique. The controller retains raw trials and
derives summaries only from those receipts. Repetition count is an exact
positive integer, the completion timestamp is valid ISO-8601 with an explicit
UTC offset, and the controller replays the complete matrix validator before it
writes `matrix.json`.

Decoded bytes are logical payload volume. Filesystem object sizes and logical
operation counts are not physical reads, range requests, cache hits, or bytes
transferred. Those fields remain `null` with an explicit external-tracing
reason.

## Adversarial requirements

The focused test must reject, including after outer evidence digests are
recomputed:

- changed candidate storage-plan claims;
- changed coordinate-publication claims;
- coordinated changes to both source and candidate scientific-identity or
  semantic claims;
- changed selector receipts;
- changed deterministic workload parameters or spans;
- changed per-operation payload digests;
- physical-I/O claims without a trace;
- claims that either diagnostic adapter is a public payload reader;
- promotion claims;
- live selectors that point at the candidate;
- a self-consistent but executable-plan-incompatible storage receipt;
- wrong profile attributes or a re-signed receipt naming another profile;
- duplicate instance keys, negative lineage, malformed/incorrect reasons,
  invalid finite payloads, valid nonfinite payloads, and inconsistent degree
  angles;
- alias run names, symlink archive aliases, and archive-overlapping output
  paths; and
- boolean/zero repetition claims and malformed or non-UTC completion
  timestamps.

## Invocation

```bash
scripts/py -m fisheye.diagnostics.benchmark_tail_posture_view_v3_candidate \
  --archive /path/to/recording_analysis.zarr \
  --source-run <selected-v3-run> \
  --candidate-run <completed-ineligible-byte-planned-run> \
  --output-root /path/to/.palette_benchmarks/tail_posture_view_v3/<matrix>
```

The archive must already contain a current consolidated root that is exactly
equivalent to its direct family subtree. This diagnostic does not consolidate
or repair metadata.

## Implementation checklist

- [x] Freeze the exact ten-array schema and semantic fill rules.
- [x] Distinguish diagnostic payload adapters from public/internal coordinate
  and selection evidence.
- [x] Require eligible source, ineligible candidate, and unchanged three-way
  family selection.
- [x] Replay the byte-planned receipt and direct/consolidated declarations.
- [x] Require bit-exact, NaN-safe equality for all arrays.
- [x] Define deterministic eager and row-window receipts for every array.
- [x] Run balanced fresh child processes with pinned thread environment.
- [x] Fail closed on re-signed evidence, live selector/storage attacks, aliases,
  symlinks, unsafe outputs, and false consumer/I/O/promotion claims.
- [ ] Obtain independent implementation review before committing or catalog
  registration.
- [ ] Run a representative recording-scale source/candidate matrix.
- [ ] Run the real `build_megabouts_classifier_input_pack` consumer with
  manifest-matched track and swim-bout dependencies.
- [ ] Add external file/range tracing if physical transfer claims are needed.
- [ ] Consider profile promotion only under the shared documented gates; this
  document and its benchmark never authorize it.

## Non-goals

This work does not edit writers, schemas, planners, selectors, profiles,
registries, production archives, canonical data, or the Megabouts consumer.
It does not claim HTTP, SMB, PRFS, TensorStore, or OS-cache performance without
the corresponding real trace and consumer evidence.
