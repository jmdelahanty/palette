# Chaser-distance sealed-base read benchmark contract — 2026-08-03

Status: implemented as a read-only, benchmark-only diagnostic for the exact
30-array sealed base. It does not publish data, register runs, change selectors,
change a writer/default/profile, or authorize promotion.

Implementation:
`fisheye.diagnostics.benchmark_chaser_distance_base_candidate`.

## Exact comparison boundary

Every invocation names all four identities explicitly:

```text
source parent:    analysis/chaser_distance_runs
source run:       <explicit canonical v1 run>
candidate parent: analysis/chaser_distance_storage_candidates
candidate run:    <explicit selector-ineligible v2 candidate>
```

The exact parent literals are required. Run aliases, separators, whitespace,
missing names, and identical source/candidate names fail before evidence is
written. The output must be a new benchmark-labelled directory disjoint from
the analysis archive.

The preflight and every fresh trial call the canonical
`load_bound_chaser_distance_run()` path and rebuild the source authority
binding. This rechecks the current publication seal, surface manifest, row
identity, input/measurement authorities, chaser collection, and epoch-window
authority. The candidate then passes `validate_base_candidate()` against that
external source group and exact binding. The benchmark therefore cannot bless
an arbitrary live v1 tree or a candidate detached from its source.

The candidate's executable `analysis_storage_plan_receipt` must reconstruct
exactly. Its declared dimensions, 30 array declarations, access classes,
chunks, shards, codecs, and payload digest define the workload suite. Source
and candidate declarations and complete decoded logical hashes must agree
before trials begin. The preflight source-authority SHA-256 and exact persisted
candidate-manifest payload digest are matrix identities; every fresh trial must
reproduce those same identities.

## Workloads

The persisted receipt generates one deterministic primary read per array:

- `EAGER` arrays are loaded completely;
- `WINDOWED` arrays use the shared suite's bounded 4,096-row selections; and
- every array is additionally read by a blocked full scan.

The sealed base currently contains only eager and windowed access classes. A
receipt that unexpectedly generates per-row or indexed workloads is rejected
rather than interpreted through an invented adapter.

Both primary-selection digests and dtype/shape/full-scan digests must match
across every source and candidate trial. Full scans use bounded blocks targeted
at 8 MiB decoded, so the correctness check does not require one whole-array
allocation for long frame-major arrays.

## Fresh-process matrix

The controller launches a fresh Python process for each role. Default seed 17
and five repetitions produce this balanced order:

```text
candidate, source
source, candidate
candidate, source
source, candidate
candidate, source
```

This isolates Python/Zarr decoded state and process RSS. It does not clear the
operating-system or mounted-filesystem cache, so callers must record an exact
cache-state description. Rotation balances but does not relabel that cache
state as cold.

Each trial records:

- source authority/candidate contract validation CPU and wall time;
- direct and consolidated open CPU and wall time;
- complete persisted direct/consolidated subtree comparison time and digest;
- per-array primary and full-scan CPU, wall, operations, decoded bytes, and
  logical digests;
- exact selected-array payload object count, metadata file count, apparent
  bytes, and allocated bytes, plus the whole-run context inventory;
- initial/final process peak-RSS high-water marks and total CPU/wall time;
- materialization, physical-copy, and publication timestamp only when the
  candidate already carries the exact completed atomic-publisher receipt. The
  benchmark reconstructs its complete field set, publisher/policy/rollback
  contract, source authority, all 30 logical hashes, four successful validation
  phases, physical-copy proof, target path, and exact run-stamped publication
  owner rather than trusting a matching top-level schema tag; and
- Palette/environment/thread/cache identity.

The selected storage totals sum the 30 array directories and explicitly exclude
group metadata, avoiding a misleading comparison between the base-only
candidate and extra unsealed arrays that may coexist in the v1 source run.

Physical request count and transferred bytes remain JSON `null` with an
unavailable declaration. Decoded bytes, filesystem object counts, and apparent
bytes are never presented as physical transfer telemetry. A mounted PRFS/SMB or
HTTP experiment needs OS/filesystem/driver tracing to populate those fields in
a future schema.

## Metadata and mutation guard

Every role must have one valid persisted consolidated generation exactly equal
to its direct subtree. Missing or stale consolidated metadata is terminal.

Before preflight and after all subprocesses, the controller hashes every
`zarr.json` under the root, both explicit parents, and both selected runs,
including path, size, and modification time. Any difference fails the matrix.
All Zarr handles use mode `r`; benchmark JSON is written only under the separate
output directory. Complete decoded scans additionally prove source/candidate
payload equality without hashing every physical storage object merely to check
for writes.

## Evidence format and promotion boundary

The output is immutable and contains:

```text
<benchmark-output>/
  analysis_benchmark_suite.json
  matrix_result.json
  trials/
    rep_00_pos_0_candidate.json
    rep_00_pos_1_source.json
    ...
```

Trial and matrix documents use closed strict-JSON envelopes and canonical
payload SHA-256 digests. Validators reconstruct role/path/parent/suite/receipt,
rotation, per-array aggregate totals, metadata-guard inventory digests,
physical-I/O availability, correctness, and promotion fields so recomputing a
digest after tampering does not make the document valid.

Every matrix states:

```json
{
  "authorized": false,
  "reason": "benchmark_only; profile promotion requires a separate reviewed decision"
}
```

Five repetitions provide a representative family result, not an automatic
profile-promotion decision.

## Example

```bash
scripts/py -m fisheye.diagnostics.benchmark_chaser_distance_base_candidate matrix \
  /path/to/recording_analysis.zarr \
  --source-parent analysis/chaser_distance_runs \
  --candidate-parent analysis/chaser_distance_storage_candidates \
  --source-run explicit_canonical_distance_v1 \
  --candidate-run explicit_sealed_base_http_v1 \
  --cache-state mounted_prfs_uncontrolled_os_cache \
  --output-dir /tmp/.palette_benchmarks/chaser_distance_read_20260803
```

## Checklist

- [x] Require explicit exact parents and immutable child names.
- [x] Rebuild the canonical source authority binding in every trial.
- [x] Bind the preflight source authority and candidate manifest into the
      matrix identity and every fresh trial.
- [x] Deeply validate the selector-ineligible, unpromoted candidate.
- [x] Bind workloads to the executable candidate storage receipt.
- [x] Compare all 30 decoded arrays and primary selections exactly.
- [x] Require complete direct/consolidated persisted metadata equivalence.
- [x] Run deterministic rotated fresh processes.
- [x] Record CPU, wall, RSS, storage objects/bytes, and existing publication
      timing truthfully.
- [x] Keep unmeasured physical request/transfer values null.
- [x] Guard source metadata against writes and write evidence outside the
      archive.
- [x] Reject recomputed-digest identity, nested aggregate/metadata,
      publication-receipt, order, physical-I/O, and promotion tampering.
- [ ] Run the default five-repetition matrix on an authorized representative
      full-duration archive.
- [ ] Add mounted-filesystem tracing before claiming physical transfer counts.
- [ ] Make any promotion decision in a separate reviewed change.
