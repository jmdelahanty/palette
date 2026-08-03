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
- Executable source and candidate array-schema manifests. The source's exact
  41-array paths, resolved shapes, and dtypes are replayed from the installed
  v7 schema; its embedded physical declarations are also required to match the
  immutable archive while no independent legacy physical receipt exists.
- Candidate storage receipt exactly replanned from uncompressed data shape,
  dtype, and access class.
- Candidate direct physical declarations exactly reconstructed from that plan,
  including chunks, shards, codecs, and fills.
- Complete output, variant, algorithm, and source-lineage manifests with their
  executable validation and dependency metadata binding.
- Exact atomic-publication root and nested field sets, owner, target, policy,
  physical-copy evidence, all four complete validation phases, materialization
  and source-revision bindings, visibility policy, unchanged parent-selector
  snapshot, and nonpromotion policy. The materialization check reuses the
  installed canonical staged-input receipt validator and exact-checks the
  compute, source-staging, inventory, capacity, algorithm, and output
  subcontracts with cross-envelope digest/identity binding. Duplicated
  subject-shape, canonical-keypoint, and diagnostic-keypoint run names must
  equal the scientific source contract and staged authority, while the staged
  selected-array inventory is reconstructed exactly from those contracts and
  the canonical staged logical-input references.
- Direct and root-inline consolidated metadata equivalence for both runs,
  including the exact installed receipt schema ID and version.

Historical timestamps, elapsed times, throughput, host/job identity, measured
free space, and the free-form invocation string are exact-shaped immutable
runtime observations. The benchmark rejects additions, type changes, and
cross-envelope divergence in those fields, but does not claim it can reproduce
their historical truth. They are not scientific or coordinate authority;
scientific inputs, compute parameters, logical schemas, and contract identities
remain executable and digest-bound.

The atomic publisher's physical-copy file count, byte count, inventory digest,
and content digest are also exact-shaped immutable publisher observations. They
describe the copied tree before owner and publication-receipt metadata were
added, so they are not presented as a replayable hash of the final run tree.
Executable final evidence comes from strict decoded equality, reconstructed
array declarations, contract validation, and direct/consolidated metadata
equivalence.

## Safety and evidence rules

- Archive, selected runs, selected dependencies, and output ancestry reject
  symlinks; every archive-relative selected/dependency path component is
  checked with `lstat` and resolved containment before that hierarchy is read;
  run names reject aliases and path syntax.
- The benchmark opens Zarr only with `mode="r"`.
- The output must be a new disjoint path whose name is explicitly benchmark
  scoped.
- A before/after metadata guard covers root, analysis, the eye-angle parent,
  both runs, and bound source dependency declarations.
- Each role/order position runs in a distinct fresh process. The controller PID
  is passed into, persisted by, and validated within every child; every child
  requires it to equal its live parent PID, every child PID differs from it,
  and matrix/trial driver identities must match exactly.
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
- [x] Bind the exact controller PID into every child trial and reject replayed
  or independently rehashed driver identity.
- [x] Guard archive metadata read-only before/after the matrix.
- [x] Freeze diagnostic-only adapter and hard nonpromotion fields.
- [x] Cover coordinated storage, publication, metadata-identity, source
  declaration, lineage, nested materialization, staged-receipt, and
  cross-envelope rehash attempts, including renamed source runs and altered
  staged-array inventories.
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
