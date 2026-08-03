# Track-kinematics v2 read benchmark contract — 2026-08-03

Status: implemented diagnostic awaiting independent review. The flat-lineage
candidate remains selector-ineligible and unpromoted.

## Purpose and boundary

This benchmark compares one explicit immutable maintained track-kinematics v1
source with one explicit immutable flat-lineage v2 storage candidate. It does
not modify archive data, metadata, selectors, registries, or storage profiles.
Evidence is written only to a new disjoint benchmark directory.

The source is loaded through the maintained public
`load_track_kinematics_track()` authority boundary. The v2 candidate has no
maintained public consumer: it is opened only by the explicit private validator
in `benchmark_track_kinematics_v2_candidate`. Every pair, trial, and matrix
therefore freezes:

- public source consumer implemented: true;
- public candidate consumer implemented: false;
- diagnostic candidate consumer implemented: true;
- candidate selector eligible: false;
- profile promoted: false.

## Exact inventory

The maintained v1 source stores two structured lineage arrays. The v2
candidate flattens their five fields into independently typed arrays. The
executable inventory therefore distinguishes the layouts instead of using one
misleading constant:

```text
source arrays    = (1 + arena_inventory_present) + T * (69 + physical * 35)
candidate arrays = (1 + arena_inventory_present) + T * (72 + physical * 35)
```

`T` is the number of tracks. The real-Zarr integration fixture has one track,
an arena inventory, no physical bundle, 71 source arrays, and 74 candidate
arrays.

This benchmark version intentionally covers only the no-physical surface. It
hard-codes and validates both:

- `physical_surfaces_present: false`
- `physical_bundle_benchmarked: false`

The optional 35-array physical bundle is counted but not read, tested, or
claimed. A source with any `positions_mm` bundle is rejected rather than
silently treated as covered.

## Correctness and storage evidence

Before timing, the benchmark requires:

- exact maintained v1 schema, completion, selector eligibility, and public
  authority for every selected track;
- exact closed v1 paths, groups, dtypes, shapes, row counts, and second counts;
- exact v2 candidate schema, completion, selector ineligibility, and
  nonpromotion state;
- complete decoded equality for all 74 candidate logical projections;
- reconstruction of structured v1 lineage into the five exact v2 primitive
  arrays;
- executable replay of the persisted `published_http_v1` byte-planned receipt;
- exact atomic publication envelope, owner, target, policy, validation phases,
  and parent attribute snapshots that are unchanged and exactly equal to the
  live frozen source-selection parents;
- exact publisher backend/verification pairs: Python copy requires
  `sha256_all_physical_files`, while rsync requires
  `rsync_checksum_dry_run`; invented or crossed pairs fail closed;
- exact direct/root-inline consolidated metadata equivalence for source and
  candidate;
- component-wise nonsymlink containment for selected runs and every bound
  dependency before those hierarchies are consumed;
- unchanged selected source authority before and after the candidate.

The atomic publisher's nested `physical_copy` hashes describe a deleted
node-local publication source. They cannot be replayed from the surviving
candidate and are therefore explicitly classified as
`opaque_publisher_provenance_not_replayed_not_benchmark_authority`. Their
counts, sizes, and hashes are not used in any benchmark conclusion. The matrix
instead records current run-tree storage facts directly. Those facts use a
closed receipt bound to the selected role, Zarr run path, and canonical
filesystem path. Matrix validation recomputes them from the live immutable
archive; coordinated edits followed by re-signing therefore fail.

## Read workload and process model

The deterministic primary workload visits every candidate logical array:

- eager arrays are read in full;
- windowed arrays use deterministic bounded row windows, including first,
  middle, and final boundaries when available;
- source structured fields and candidate flat arrays are hashed into the same
  logical stream;
- every trial also performs a complete logical hash scan.

Each array receipt records its declared access class, exact source/candidate
read path and projected structured field, dtype, shape, deterministic half-open
row spans, operation count, decoded bytes, and payload digest. The validator
reconstructs every non-payload field from the executable byte-plan
declarations and the matrix seed/window parameters, recomputes the aggregate,
and requires matched source/candidate logical projections. Matrix validation
then reruns both logical workloads against the live immutable archive. Thus a
coordinated payload-digest rewrite, even with every trial, summary, and outer
envelope re-signed, is not accepted as an observation.

Source/candidate order rotates by repetition. Each role/order position runs in
a distinct fresh child process. The controller PID is passed explicitly; the
child requires it to equal its live parent PID, and matrix validation binds
every child PID, role, order position, pair receipt, and controller PID.

Wall time, CPU time, peak RSS, decoded bytes, operation counts, environment,
and current filesystem object/byte facts are recorded in closed, exact-shaped
receipts. The primary timing is explicitly
`fresh_child_post_pair_validation_os_cache_uncontrolled`: it is a post-
validation historical observation, not a cold-open result. Timings, RSS, and
environment facts are process observations carried by digest-protected JSON;
they are not cryptographically attested measurements. Current storage facts
and metadata guards are additionally replayed against the live immutable
archive during matrix validation. Physical file reads, range reads, and
transferred bytes remain null because Python/Zarr timing does not provide that
telemetry. Physical transfer claims require a separate OS/filesystem trace.

The final matrix validator reopens the immutable archive and reconstructs the
pair validation, archive guard, and selected run-tree storage facts. It also
requires exact nested environment, runtime, storage, and metadata-guard field
sets, identical child environments, and the declared deterministic thread
environment. Coordinated changes to metadata-equivalence receipts,
publication bindings, logical hashes, byte plans, storage inventories,
physical-coverage flags, workload parameters/payload observations, or
selectors therefore fail even if all outer JSON digests and aggregate
summaries are recomputed.

## Implementation checklist

- [x] Build a genuine public real-Zarr v1 fixture through production writers and
  the unpatched maintained reader.
- [x] Publish its v2 candidate through the production byte planner and atomic
  candidate publisher.
- [x] Freeze separate 69-array structured-source and 72-array flat-candidate
  per-track formulas.
- [x] Validate all exact paths, shapes, dtypes, topology, lineage projection,
  decoded values, and storage plans.
- [x] Require direct/consolidated metadata equivalence.
- [x] Preserve source selectors and hard-code candidate nonpromotion.
- [x] Run deterministic all-array primary reads and complete scans.
- [x] Reconstruct exact per-array spans, operation counts, decoded bytes, and
  aggregate digests from declarations plus matrix parameters; require matched
  source/candidate projections and live workload replay.
- [x] Rotate fresh child processes and bind the live controller PID.
- [x] Record wall/CPU/RSS/storage facts and refuse fabricated physical I/O.
- [x] Close and cross-bind environment, storage, and archive-guard receipts;
  replay current storage facts against the live archive.
- [x] Reject selected-run and dependency descendant symlinks before reads.
- [x] Cover coordinated storage-plan rehash, metadata-receipt divergence,
  coordinated current-storage rehash, nested telemetry field injection, false
  physical coverage, false publication authority, and atomic receipt field-set
  attacks.
- [x] Cover crossed/unknown publisher backend-verification pairs, a coordinated
  forged parent-pointer snapshot, and a fully re-signed matched primary-payload
  attack.
- [x] Exercise a genuine rsync publication when rsync is installed; the
  integration test skips deterministically when that executable is absent,
  while exact backend-pair unit coverage remains unconditional.
- [ ] Receive an independent ACCEPT review before commit.
- [ ] Run a five-repetition full-duration matrix on an immutable matched pair.
- [ ] Add OS/filesystem tracing if physical transfer attribution is required.
- [ ] Implement or explicitly decline a maintained public v2 candidate reader.
- [ ] Benchmark the optional physical bundle in a separate explicitly scoped
  fixture before making any statement about those 35 arrays per track.

## Invocation

```bash
scripts/py -m fisheye.diagnostics.benchmark_track_kinematics_v2_candidate matrix \
  --archive /path/to/recording_analysis.zarr \
  --source-run EXPLICIT_SELECTED_V1_RUN \
  --candidate-run EXPLICIT_INELIGIBLE_V2_RUN \
  --output /path/to/.palette_benchmarks/track_kinematics_v2_reads_20260803 \
  --repetitions 5
```

The output contains `pair_validation.json`, one JSON document per child trial,
and `matrix.json`. It does not publish, register, select, or promote anything.
