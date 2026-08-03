# Stimulus-Epoch Exact Storage Contract and Candidate — 2026-08-03

Status: implemented selector-ineligible candidate contract, strict maintained
v2 consumer, and read-only source/candidate benchmark harness; not a production
writer, selector, registry, or profile promotion.

## Census and authority boundary

The maintained producer is
`fisheye.analysis.stimulus_epoch_runs.write_stimulus_epoch_run`. It resolves a
versioned protocol profile against one exact stimulus run and writes
`analysis/stimulus_epoch_runs/<run>/windows`. The scientific inputs and output
dataclass are defined in `stimulus_epoch_runs.py`; the current direct writer
creates twelve arrays with hand-selected row chunk limits and identifies them
as `palette.stimulus_epoch_windows.v1`.

The v1 producer remains unchanged. It is now an explicit compatibility input,
not an exact v2 authority. It has no exact array manifest or executable storage
receipt. The candidate boundary in
`analysis_workflows/materializers/stimulus_epochs.py` accepts only one explicit,
complete, non-ineligible v1 name and validates its entire logical table before
copying it.

Most pre-existing consumers remain compatibility readers:

- detection occupancy reads seven display/interval columns and casts them to
  expected NumPy dtypes;
- chaser distance and several chaser summaries reuse that interval reader or
  equivalent local readers;
- those readers do not yet require the v2 schema, exact manifest, or physical
  receipt.

The maintained adapter now lives in
`fisheye.analysis.stimulus_epoch_consumer`. It accepts one exact, explicitly
named run and never resolves `latest`, guesses a child, or probes alternate
paths/dtypes. Exact v2 is the default. Legacy v1 is available only through the
typed `ALLOW_EXPLICIT_V1` compatibility policy; a rejected explicit v2 run is
terminal and never falls back to v1.

For a v2 candidate, the adapter first compares every persisted direct Zarr
declaration with the archive root's inline consolidated declaration. It then
opens the consolidated generation, requires exact completion and run-name
binding, requires both candidate-only booleans to remain false, and executes
the logical manifest, candidate lineage, run manifest, and physical storage
receipt validators. Only after all gates pass does it eagerly decode the small
table into backend-independent `EpochSegment` values, including the source
event boundary frames.

## Exact v2 logical inventory

Schema: `palette.stimulus_epoch_windows.v2`, version `2`.

Layout: `exact_columnar_v1`.

Every array is required. There are no optional bundles or aliases.

| Path | Dtype | Shape | Units / meaning | Authority | Access |
| --- | --- | --- | --- | --- | --- |
| `windows/window_id` | `int32` | `[W]` | stable window ID | lineage index | eager |
| `windows/label_bytes` | `uint8` | `[W,96]` | NUL-padded UTF-8 label | semantic metadata | eager |
| `windows/start_frame` | `int64` | `[W]` | inclusive camera frame | scientific | eager |
| `windows/end_frame` | `int64` | `[W]` | inclusive camera frame | scientific | eager |
| `windows/start_time_s` | `float64` | `[W]` | `start_frame / fps` | scientific | eager |
| `windows/end_time_s` | `float64` | `[W]` | exclusive `(end_frame+1) / fps` | scientific | eager |
| `windows/duration_s` | `float64` | `[W]` | inclusive frame count / FPS | scientific | eager |
| `windows/source_start_event_name_bytes` | `uint8` | `[W,96]` | resolved source event | lineage | eager |
| `windows/source_end_event_name_bytes` | `uint8` | `[W,96]` | resolved source event | lineage | eager |
| `windows/source_start_event_frame` | `int64` | `[W]` | inclusive source boundary | lineage | eager |
| `windows/source_end_event_frame` | `int64` | `[W]` | exclusive source boundary | lineage | eager |
| `windows/source_policy_bytes` | `uint8` | `[W,160]` | boundary/fallback policy | semantic metadata | eager |

The payload is 508 uncompressed bytes per window across all arrays. A normal
three-window GoodCopBadCop run therefore contains only 1,524 logical payload
bytes. Eager whole-array reads are the correct access model. The published HTTP
profile produces one chunk and one payload object per array for the tested
three-window fixture; sharding provides no object-count benefit at this scale.
The physical receipt remains executable and byte-derived so an unusually large
epoch table is still planned from its actual shapes and dtypes.

## Semantic invariants

The executable validator in `analysis/stimulus_epoch_schema.py` requires:

- exactly the twelve arrays above, with exact rank, shape widths, and dtypes;
- at least one row;
- nonnegative, unique, strictly increasing `window_id` values;
- unique nonempty UTF-8 labels;
- NUL padding with no nonzero bytes after the first terminator;
- positive finite FPS and positive exact-integer `total_frames`;
- inclusive, nonempty frame intervals inside the recording;
- chronological row order and no overlapping windows;
- ordered source-event boundaries inside `[0, total_frames]`;
- finite time columns exactly derived from frame bounds and FPS within a small
  float64 arithmetic tolerance.

Each v2 candidate also owns a canonical
`stimulus_epoch_run_manifest`. Its digest-bound payload is reconstructed from
the current run and binds:

- recording identity, window count, total frames, and FPS;
- the exact source stimulus run/path and a decoded logical-tree fingerprint of
  that stimulus group;
- the exact v1 source epoch run/path, lineage hash, lineage-payload digest, and
  full logical-content digest;
- the protocol profile, adapter, role resolver, method, and window-policy
  identities;
- the newly computed v2 candidate lineage hash and payload digest;
- every decoded candidate array, the exact array-manifest and storage-plan
  digests, and the authoritative `windows` group declaration;
- exact `stage_selector_eligible=false` and
  `storage_candidate_profile_promoted=false` publication state.

The candidate never inherits the v1 `lineage_payload_json`. It constructs and
persists a new complete v2 lineage payload that binds the candidate schema,
both source identities, source fingerprints, protocol parameters, dimensions,
and materializer revision. Validators recompute both the lineage and run
manifest. Rehashing only a modified outer manifest therefore cannot make
recording, dimensions, source, policy, profile, or lineage tampering valid.

Empty labels, invalid UTF-8, aliases such as `frame_counts`, missing arrays,
wrong dtypes, overlapping intervals, duplicate IDs, and inconsistent time
columns fail closed.

## Physical and publication contract

The candidate materializer:

1. resolves a safe explicit v1 source name and a distinct immutable v2 name;
2. rejects source-run symlinks, target replacement, and source/scratch tree
   containment in either direction;
3. validates v1 logical semantics and hashes every decoded array;
4. plans actual fixed-width bytes using `published_http_v1`;
5. creates a fresh node-local Zarr v3 through the shared array factory, which
   pins the profile's bytes/Zstd codec chain;
6. writes complete non-overlapping physical units;
7. writes the exact logical manifest and executable storage receipt;
8. proves source/candidate logical hashes are identical;
9. writes and validates candidate-owned v2 lineage plus the canonical run-level
   scientific/lineage manifest;
10. consolidates local metadata and proves the complete direct/consolidated
   declaration tree is equal, including the run group, `windows` group attrs,
   and every array declaration;
11. uses the common atomic run-group publisher to copy into a hidden sibling,
    validate it, rename it atomically, and retain selector-ineligible state;
12. leaves `latest` and `latest_complete` unchanged;
13. consolidates the authoritative archive as the final visibility step and
    proves direct/consolidated equivalence;
14. on a post-consolidation failure, writes an owner-bound failed tombstone,
    reconsolidates, and requires exact failed metadata in both views.

The candidate records `storage_candidate_profile_promoted=false` and
`stage_selector_eligible=false`. It is not registry authority, production
selection evidence, or permission to change the v1 writer.

## Read-only source/candidate benchmark contract

`fisheye.diagnostics.benchmark_stimulus_epoch_reads` benchmarks exactly one
explicit v1 source and its explicitly named selector-ineligible v2 candidate.
It never resolves `latest`, performs compatibility probing, opens an archive in
write mode, or treats a completed benchmark as profile-promotion evidence.

Preflight executes both family validators before creating the output
directory. The source must be an exact complete v1 run with canonical lineage
and cannot be explicitly selector-ineligible. The candidate is opened only
through the strict integrated v2 consumer. Its complete logical manifest,
candidate lineage, storage-plan receipt, run manifest, lifecycle booleans, and
persisted direct/consolidated declaration tree must validate. The candidate's
persisted source name, source path, source lineage, and complete source-content
digest must bind the exact selected v1 source.

The versioned workload records:

- the absolute archive path and both explicit immutable child names;
- the deterministic eager access order for all twelve arrays;
- exact dtype, shape, decoded-byte digest, logical-table digest, and decoded
  segment digest expectations;
- the complete canonical source and candidate lineage payloads, their hashes,
  and their canonical payload digests;
- the complete executable candidate storage receipt and complete scientific
  run manifest, plus their recomputed payload digests;
- the persisted candidate materializer commit/dirty identity, cross-bound to
  the candidate lineage's code document;
- exact direct/consolidated metadata-equivalence receipts for both runs; and
- explicit null physical request/transfer telemetry when no external file or
  HTTP tracer is active.

Each repetition uses two new Python processes. Source/candidate order rotates
deterministically, and the matrix rejects reused child PIDs or a child PID equal
to the controller. Every child performs its role-specific strict validation,
then eagerly reads and hashes all twelve complete arrays. The controller binds
each result back to the workload, requires complete source/candidate decoded
equality, and recomputes medians for validation, full-scan, total wall/CPU time,
peak process RSS, object counts, and apparent/allocated bytes.

A read-only guard hashes every direct `zarr.json` in the root, `analysis`, run
parent, and both complete selected run subtrees before and after the matrix. A
changed size, modification time, content hash, path inventory, or tree digest
fails the run. Evidence is strict finite JSON with closed field sets and
canonical payload digests. Offline validation reconstructs the full byte plan
from the embedded profile, declarations, dimensions, and observed array facts.
It cross-binds the run manifest's source identity, source lineage, logical
content, storage receipt, child-group declaration, and exact false publication
state. The embedded candidate lineage is additionally bound to those same
source identities and fingerprints, protocol parameters, dimensions, profile,
method, and persisted materializer code identity. Rehashing only edited outer
envelopes and matching bare digest fields therefore does not permit fabricated
transfer counts, receipt rebinding, workload/access drift, PID reuse, guard
drift, or profile-promotion authorization.

The JSON evidence is digest-bound but not cryptographically signed. A party
that replaces every embedded scientific document and every matching digest can
create a different internally consistent document. Long-lived or exchanged
evidence must therefore publish the top-level matrix digest in an external
immutable manifest (or add a signature) if authenticity against total evidence
replacement is required.

Benchmark evidence must live in a new path containing `benchmark` and outside
the source archive. The hard-coded evidence boundary states that it cannot
authorize writer, selector, registry, or physical-profile changes. A complete
five-repetition matrix is therefore read evidence only; it still requires a
representative archive execution, genuine physical-I/O tracing, real consumer
review, and a separate versioned promotion decision.

## Implementation checklist

- [x] Census the current v1 producer and permissive consumers.
- [x] Freeze exact paths, dtypes, widths, fills, null rules, and authority roles.
- [x] Freeze row-order, interval, and time-consistency semantics.
- [x] Declare v1 input-only compatibility and v2 strict-authority boundaries.
- [x] Add a digest-bound exact array manifest.
- [x] Add a deeply validated canonical run-level scientific/lineage manifest.
- [x] Replace inherited v1 lineage with candidate-owned complete v2 lineage.
- [x] Bind the actual source stimulus logical-tree fingerprint and exact source
  epoch lineage/content identities.
- [x] Replan actual bytes with the shared storage planner and factory.
- [x] Materialize on node-local scratch as Zarr v3.
- [x] Prove exact decoded equality and direct/consolidated equivalence.
- [x] Publish atomically without selector or registry changes.
- [x] Repair consolidated failed-tombstone visibility after injected failure.
- [x] Compare the complete direct/consolidated declaration tree, including
  authoritative child-group declarations.
- [x] Enforce `storage_candidate_profile_promoted` as exact JSON `false`.
- [x] Add adversarial inventory, dtype, ordering, containment, symlink, and
  failure-visibility tests, plus recomputed-digest manifest/lineage probes.
- [x] Add a strict v2 consumer adapter; keep current v1 readers explicitly
  legacy-compatible.
- [x] Add a deterministic explicit-source/candidate workload covering all
  twelve complete arrays and decoded segments.
- [x] Add rotated fresh-process trials with strict workload, receipt, manifest,
  lineage, and metadata bindings.
- [x] Embed and executably reconstruct the complete source/candidate lineage
  payloads, storage-plan receipt, and scientific run manifest; reject
  coordinated bare digest rebinding after every outer envelope is rehashed.
- [x] Record wall/CPU time, peak RSS, object counts, apparent/allocated bytes,
  and explicit null request/transfer telemetry when tracing is absent.
- [x] Add a complete direct-metadata read-only guard and hard nonpromotion
  boundary.
- [x] Add adversarial workload/trial/matrix, receipt, metadata-staleness,
  decoded-mismatch, PID-reuse, unsafe-path, and rehashed-evidence tests.
- [ ] Run a real archive canary and record read/open/object-count telemetry.
- [ ] Obtain consumer review before considering writer adoption.
- [ ] If adoption is approved, make the scientific writer emit v2 directly on
  node-local scratch and retain v1 read compatibility.
- [ ] Treat profile promotion, selector activation, and registry eligibility as
  separate reviewed changes.

## Focused validation

The implementation is covered by:

- `tests/unit/fisheye/test_stimulus_epoch_schema.py`;
- `tests/unit/fisheye/test_stimulus_epoch_candidate_materializer.py`;
- `tests/unit/fisheye/test_stimulus_epoch_consumer.py`;
- `tests/unit/fisheye/test_benchmark_stimulus_epoch_reads.py`.

The suite includes a real post-consolidation injected failure and verifies that
the resulting public child is failed and selector-ineligible in both direct and
consolidated metadata while the prior parent pointers remain unchanged. The
consumer suite additionally covers explicit legacy opt-in, stale consolidated
metadata, missing/unexpected arrays, wrong dtype/rank, lifecycle violations,
rehashed manifest tampering, receipt-digest tampering, and terminal v2 failure
without legacy fallback.
The benchmark suite additionally executes child processes and covers exact
source/candidate equality, rotated order, complete persisted metadata guards,
explicit unavailable physical-I/O telemetry, strict evidence rebinding, and
the nonpromotion boundary.
