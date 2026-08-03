# Eye-angle compact-v7 read benchmark contract — 2026-08-03

Status: implemented diagnostic; candidate remains selector-ineligible and
unpromoted.

## Goal

Measure one explicit immutable maintained compact-v7 eye-angle source against
one explicit immutable byte-planned candidate without modifying archive data,
metadata, selectors, profiles, registries, or publication state. The benchmark
establishes logical equivalence and records wall time, CPU time, peak RSS, and
storage-tree facts. Physical read counts and transferred bytes remain null
unless a separate OS/filesystem trace is supplied.

## Consumer boundary

The selector-eligible source is read through
`load_eye_angle_run_tables()`, Palette's maintained public reader. That reader
correctly rejects the selector-ineligible candidate. The benchmark therefore
uses the existing strict v7 payload validator and compact-table adapter only
after an explicit candidate name, exact candidate envelope, storage plan,
publication receipt, and ineligible lifecycle state have all passed.

This private adapter is diagnostic-only. It is not a maintained Palette
consumer, not selector acceptance, and not evidence that normal readers can
select a candidate. Every workload, trial, and matrix freezes:

- `palette_consumer_implemented: false`
- `candidate_adapter_scope: diagnostic_only_private_strict_payload_adapter`
- `candidate_selector_eligible: false`
- `promotion_authorized: false`

The analytics catalog must therefore keep `palette_consumer_implemented=false`
until Palette has either a public ineligible-candidate benchmark API or a real
selected consumer.

## Frozen correctness surface

- Exactly 41 maintained arrays with executable v7 paths, shapes, and dtypes.
- Exact decoded full-array equality, including raw IEEE payload bytes.
- Equal logical ROI, frame, QA, and support tables after excluding only run
  name/path metadata that must differ between immutable children.
- Executable source and candidate array-schema manifests.
- Candidate storage receipt exactly replanned from uncompressed data shape,
  dtype, and access class.
- Candidate direct physical declarations exactly reconstructed from that plan,
  including chunks, shards, codecs, and fills.
- Complete output, variant, algorithm, and source-lineage manifests with their
  executable validation and dependency metadata binding.
- Exact atomic-publication owner, target, copy verification, validation gates,
  visibility policy, unchanged parent-selector snapshot, and nonpromotion
  policy.
- Direct and root-inline consolidated metadata equivalence for both runs.

## Safety and evidence rules

- Archive, selected runs, selected dependencies, and output ancestry reject
  symlinks; run names reject aliases and path syntax.
- The benchmark opens Zarr only with `mode="r"`.
- The output must be a new disjoint path whose name is explicitly benchmark
  scoped.
- A before/after metadata guard covers root, analysis, the eye-angle parent,
  both runs, and bound source dependency declarations.
- Each role/order position runs in a distinct fresh process; the controller PID
  cannot appear as a trial PID.
- Candidate/source order rotates deterministically by repetition.
- The JSON envelope is self-digested for corruption detection, but portable
  authority still requires an external SHA-256 or signature over the final
  evidence artifact.

## Implementation checklist

- [x] Require distinct explicit immutable source and candidate names.
- [x] Validate exact compact-v7 logical/schema contracts.
- [x] Recompute candidate byte planning and physical declarations.
- [x] Compare all 41 decoded arrays and maintained logical tables.
- [x] Bind source dependency metadata and atomic publication evidence.
- [x] Require direct/consolidated equivalence.
- [x] Record wall/CPU/RSS/storage facts and null untraced physical I/O.
- [x] Run rotated fresh processes with PID/order enforcement.
- [x] Guard archive metadata read-only before/after the matrix.
- [x] Freeze diagnostic-only adapter and hard nonpromotion fields.
- [x] Cover coordinated receipt and lineage rehash attempts.
- [x] Cover aliases, unsafe outputs, symlink archives, order errors, and false
  physical-I/O claims.
- [ ] Run a five-repetition full-duration matrix on a published matched pair.
- [ ] Add OS/filesystem tracing if physical transfer attribution is required.
- [ ] Independently review evidence before any profile-promotion decision.
- [ ] Implement or explicitly decline a public candidate benchmark reader API.

## Invocation

```bash
scripts/py -m fisheye.diagnostics.benchmark_eye_angle_v7_reads matrix \
  --archive /path/to/recording_analysis.zarr \
  --source-run EXPLICIT_ESTABLISHED_RUN \
  --candidate-run EXPLICIT_CANDIDATE_RUN \
  --output /path/to/.palette_benchmarks/eye_angle_v7_read_matrix_20260803 \
  --repetitions 5
```

The output contains `workload.json`, one trial JSON per fresh process, and
`matrix.json`. It does not update a selector, registry, or profile.

