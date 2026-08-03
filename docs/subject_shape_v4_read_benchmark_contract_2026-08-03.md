# Subject-shape v4 read benchmark contract — 2026-08-03

Status: implementation checkpoint. The family-local diagnostic and focused
real-Zarr fixtures are implemented. A representative full-duration matrix,
physical-I/O tracing, mounted Crimson evaluation, and profile promotion remain
separate future gates.

Implementation:
`fisheye.diagnostics.benchmark_subject_shape_v4_candidate`.

## Exact comparison boundary

Both runs must be explicit, immutable children of the one maintained parent:

```text
parent:    analysis/subject_shape_runs
source:    <complete selector-eligible full-anatomy v4 publication>
candidate: <complete selector-ineligible access-aware v4 candidate>
```

Aliases, whitespace, path separators, the wrong parent, and identical names
fail closed. Evidence is written only to a new benchmark-labelled directory
outside the archive. The benchmark opens Zarr only with mode `r`.

The source must pass the canonical persisted-publication loader with its exact
publication owner. The candidate must pass the dedicated completed-ineligible
loader and `validate_subject_shape_candidate_storage(..., phase="bound")`.
The candidate storage-plan receipt is parsed back into executable declarations
and must equal a receipt recomputed from the live closed v4 inventory. Source
and candidate must have the same declared paths, dtypes, shapes, dimensions,
refined-mask source run, and complete decoded array digests.

Final source and candidate manifests are intentionally different records:
their run identities and physical metadata differ. The evidence binds them as
separate identities and also binds the candidate's retained producer-stage
seal. Manifest equality is never used as a substitute for decoded equality.

## Workloads

The candidate's executable receipt generates the deterministic access suite:

- eager semantic axes are read completely;
- per-row geometry uses 128 deterministic complete-row reads; and
- every declared array receives an additional bounded full scan targeted at
  8 MiB decoded blocks.

The complete trailing anatomy record is indivisible for a row read. The runner
rejects an unexpected indexed workload rather than inventing index semantics.
Primary selection digests and canonical dtype/shape/C-order payload digests
must match for source and candidate in every trial.

## Fresh-process matrix

The controller launches a new Python process for each role. Seed 17 and five
repetitions alternate candidate/source order, balancing process-first effects
without claiming to clear the OS or mounted-filesystem cache. Callers must
provide an explicit cache-state description.

The controller records its own PID and every child PID. Validation requires
all child PIDs to be distinct from one another and from the controller, and
requires each `(repetition, order position, role)` tuple to match the
deterministic schedule exactly. Replaying one trial under a second position is
therefore terminal even after recomputing all JSON envelope digests.

Each trial records and deeply validates:

- canonical source/candidate and retained producer-stage identities;
- exact direct and consolidated open/comparison CPU and wall time;
- per-array primary/full-scan operations, decoded bytes, CPU, wall, and digest;
- exact selected-array payload objects, apparent/allocated bytes, and whole-run
  storage context;
- initial and final process peak-RSS high-water marks and total CPU/wall time;
- existing materialization/copy/publication timing from a complete atomic
  receipt; and
- Palette, Python, Zarr, NumPy, host, thread, and cache declarations.

Physical request count and transferred bytes remain JSON `null` unless an OS,
filesystem, or driver trace measures them. Apparent bytes, decoded bytes, and
file counts are not relabelled as physical transfer telemetry.

## Self-contained contract evidence

Trial and matrix envelopes include the complete canonical final source
manifest, complete canonical final candidate manifest, candidate retained
producer manifest/link, executable storage-plan receipt, and available source
and candidate atomic publication receipts. They also embed the complete
normalized direct/consolidated metadata declaration document for every group
and array in both selected run subtrees; a bare subtree digest is not treated
as sufficient offline evidence. Offline validation reconstructs and
cross-binds:

- every coordinate-record and receipt digest;
- source/candidate run refs and refined-mask authority;
- the final closed array inventories against measured dtype, shape, and
  decoded payload digests;
- the producer seal against the candidate consumed-stage pointer and measured
  producer arrays;
- storage receipt paths and dimensions; and
- every candidate array's physical Zarr declaration from the executable
  storage receipt, including chunk grid, codecs, transformers, fill, and
  planned attributes;
- the exact source/candidate metadata inventory, attrs, and consumed-stage
  records; and
- atomic publisher schema, policy, target, owner, physical-copy proof, and all
  validation phases.

The set of arrays allowed to differ across coordinate publication is replayed
from the installed subject-shape v4 schema helpers over those embedded
metadata declarations. It is not taken from a mutable manifest map. A hostile
manifest therefore cannot exempt an otherwise invariant payload by simply
reclassifying its path as a coordinate or scalar transform.

Adversarial coverage changes a producer array digest and then coherently
recomputes the producer manifest, link, candidate consumed-stage pointer,
candidate final-manifest digest, validation identities, and outer evidence
digest. It is still rejected by the measured decoded-array cross-binding.

These embedded records make evidence independently inspectable; they do not
make it cryptographically authentic by themselves. A handoff must still pin
the immutable result-file digest outside the result document.

## Metadata and publication safety

Before and after all subprocesses, the controller records path, size, mtime,
and SHA-256 for root metadata, the subject-shape parent, both complete selected
run subtrees, and the present refined-mask source subtree. Any change is
terminal. Each run must also have exact direct/consolidated subtree metadata.
Every component of both selected paths and the refined-mask dependency path is
checked for symlinks before any Zarr hierarchy is opened. Exactly one canonical
root-level or analysis-prefixed refined dependency may exist.

The candidate must carry a complete atomic publication receipt. A historical
source may explicitly report no retained receipt, but any present source
receipt is validated completely. Candidate publication remains immutable,
selector-ineligible, and unregistered.

Every result contains the hard boundary:

```json
{
  "authorized": false,
  "reason": "benchmark_only; profile promotion requires a separate reviewed decision"
}
```

## Example

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_shape_v4_candidate matrix \
  /path/to/recording_analysis.zarr \
  --parent analysis/subject_shape_runs \
  --source-run explicit_full_anatomy_v4 \
  --candidate-run explicit_access_aware_candidate_v1 \
  --cache-state mounted_prfs_uncontrolled_os_cache \
  --output-dir /tmp/.palette_benchmarks/subject_shape_v4_read_20260803
```

## Implementation checklist

- [x] Require the exact parent and two explicit immutable child names.
- [x] Validate producer-sealed source and selector-ineligible candidate roles.
- [x] Reconstruct the executable candidate storage receipt.
- [x] Compare every declared decoded array exactly.
- [x] Bind deterministic eager/per-row/full-scan workloads to the receipt.
- [x] Run rotated fresh processes with explicit cache-state declaration.
- [x] Prove distinct controller/child PIDs and exact role-position binding.
- [x] Record and reconstruct CPU, wall, RSS, object, and apparent byte totals.
- [x] Keep unmeasured physical I/O null.
- [x] Require exact direct/consolidated metadata equivalence.
- [x] Embed full normalized metadata documents and replay candidate physical
  declarations offline.
- [x] Guard source/dependency metadata against mutation.
- [x] Reject selected-run and refined-dependency symlinks before Zarr open.
- [x] Embed and cross-bind complete manifests, receipts, and producer seal.
- [x] Derive transform exemptions from the installed schema, not manifest maps.
- [x] Reject coordinated rehash, aggregate, identity, order, metadata,
  publication, physical-I/O, and promotion tampering.
- [x] Keep implementation family-local; do not edit shared catalogs, writers,
  selectors, registries, defaults, or production data.
- [ ] Run the default five-repetition representative full-duration matrix.
- [ ] Add physical transfer tracing before making request/byte claims.
- [ ] Run the mounted Crimson consumer gate.
- [ ] Make any profile-promotion decision separately with explicit approval.

This checkpoint does not change the subject-shape writer, materializer,
planner, schema, profile default, selector, registry, or any archive.
