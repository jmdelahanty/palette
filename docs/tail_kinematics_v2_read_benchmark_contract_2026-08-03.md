# Tail-Kinematics v2 Source/Candidate Read Matrix — 2026-08-03

Status: implemented diagnostic contract; no representative archive has been
run and no storage profile is promoted by this checkpoint.

## Decision boundary

This matrix compares one selected, complete
`analysis/tail_kinematics_runs/<source>` v2 run with one explicit
`published_http_v1` byte-planned candidate containing identical decoded
values. It is intentionally narrower than a writer or promotion benchmark:

- the selected source is opened through the maintained public
  `load_tail_kinematics_window` consumer;
- the selector-ineligible candidate is opened only through
  `fisheye.diagnostics.benchmark_tail_kinematics_candidate_reads`;
- the candidate diagnostic cannot mint reader authority or change parent
  selectors;
- physical file/range I/O is recorded as JSON `null`, with
  `physical_io_measured=false`, until an external tracer supplies real
  evidence; and
- every pair, trial, and matrix hard-codes `profile_promoted=false` and
  `promotion_authorized=false`.

The older Sleepyfish shard-row experiment is historical evidence about a
legacy physical policy. Its 262,144-row result is not promotion authority for
the byte-planned candidate.

## Exact logical coverage

Both sides must expose the same closed logical surface:

- 21 required arrays;
- exact `uint64`, `int64`, `bool`, `uint8`, and `float32` dtypes;
- exact observation, tail-sample, XY, and 64-byte failure-reason axes; and
- either both arrays in the optional
  `source_refined_subject_masks_revision` bundle or neither:
  - `source_refined_subject_masks/row_revision`
  - `source_refined_subject_masks/row_revision_available`

The source's exact executable declaration is reconstructed from the maintained
schema builder because the current legacy/default writer does not persist the
candidate array-schema receipt. The candidate must persist and executably
replay both its exact logical schema and its byte-planned storage receipt.

Complete equality is not inferred from sample rows. Every declared array is
hashed in bounded first-axis blocks using dtype + shape + C-order decoded
bytes. Core-only pairs therefore prove 21 hashes; complete revision-bundle
pairs prove 23.

## Publication and coordinate binding

Before reads, the diagnostic validates:

1. the source is complete, selector-eligible, and named by `latest` and
   `latest_complete`;
2. the candidate is complete, explicitly selector-ineligible, carries
   `storage_candidate_status=unpromoted_selector_ineligible`, and uses the
   exact `published_http_v1` planner profile;
3. the candidate storage receipt round-trips and replans against live array
   metadata, and its complete embedded profile equals the registered
   `PUBLISHED_HTTP_V1` manifest—not merely the same `profile_id`;
4. the atomic publisher receipt binds the exact archive, target path,
   publication owner, validation results, and copy verification policy;
5. direct and consolidated declarations are equivalent for both subtrees;
6. the selected source's public coordinate publication reloads exactly;
7. the candidate's ineligible coordinate publication reloads through a
   diagnostic-only validation path; and
8. both tail runs bind the same exact subject-shape publication digest.

The pair also derives one closed stable scientific-identity projection from
each live run and requires exact equality. It includes method/version, row
axis, row count, compute kernel, subject-shape and refined-mask source
identities, subject-shape publication and detached-authority digests,
body-frame and tail-angle conventions, tail sample counts, curvature source,
acquisition-frame and row-lineage policies, exact source references, and FPS.
The source-authority digest is executably checked against the stored authority
record, whose canonical publication must equal the independently loaded
subject-shape publication. Runtime timestamps, worker topology, physical
layout, and the canonical-versus-staged authority access mode are deliberately
excluded: those may differ without changing decoded scientific identity.

The root `zarr.json` digest/stat tuple and all tail-parent selector fields are
not sufficient on their own. The diagnostic rejects symlink aliases and
captures every direct `zarr.json` declaration plus complete archive
file/object/byte accounting before and after every read. Any mutation fails the
trial.

Pair, trial, workload, public-consumer, and matrix evidence use exact canonical
payload envelopes with SHA-256 digests and closed nested field sets. Matrix
validation recomputes the live pair receipt rather than merely requiring all
trials to repeat the same receipt. Re-signing a changed receipt is therefore
not authority.

The atomic publisher receipt is retained and validated as opaque publication
provenance. Its exact policy, source/target/owner, parent-selector snapshots,
four validation receipts, source-staging receipt, and physical-copy
hash/count/backend fields are checked. Historical copy hashes cannot be
replayed after node-local scratch is removed and are never reported as
physical-I/O benchmark evidence; the pair envelope instead binds them beside
the freshly computed complete logical hashes.

## Workloads

Each fresh process performs two distinct reads:

1. **Exact all-array workload.** Eager arrays are loaded once. Every
   observation-aligned array is read at deterministic beginning, middle, end,
   and seeded positions using the configured row window. The persisted
   workload is digest-bound and is replayed from live arrays by the parent
   process before evidence is accepted.
2. **Maintained public-consumer workload (source only).** A bounded source
   window loads native tail angles and all maintained scalar series through
   `load_tail_kinematics_window`. Its decoded result is replayed. A candidate
   trial must record this field as `null`.

Source/candidate order rotates by repetition. Every role is executed in a
separate subprocess. The child verifies its direct driver PID; the driver
records the actual `Popen.pid`; and the matrix requires exact equality among
spawn receipt, child PID, parent PID, role, repetition, seed, window, archive,
and run identities. Default evidence is five repetitions, ten fresh processes.

Example diagnostic invocation:

```bash
scripts/py -m fisheye.diagnostics.benchmark_tail_kinematics_candidate_reads \
  matrix \
  --archive /path/to/recording_analysis.zarr \
  --source-run selected_tail_v2 \
  --candidate-run tail_v2_published_http_candidate \
  --repetitions 5 \
  --window-rows 4096 \
  --output /path/outside/archive/tail_matrix.json
```

The output path must be outside the source archive and must not already exist.

## What this checkpoint proves

- [x] Public source and diagnostic-only candidate boundaries are distinct.
- [x] Exact 21-array core-only fixtures are covered.
- [x] Exact 23-array complete revision-bundle fixtures are covered.
- [x] Partial optional revision bundles fail closed.
- [x] Complete decoded equality is executable, bounded, and path-specific.
- [x] Candidate logical/storage receipts replay from live arrays.
- [x] Candidate profile equals the complete registered `PUBLISHED_HTTP_V1`;
      a re-signed same-ID budget or codec change fails closed.
- [x] Stable source/candidate scientific identities are closed, digest-bound,
      authority-bound, and exactly equal.
- [x] Atomic publication and coordinate lineage are bound.
- [x] Direct/consolidated metadata equivalence is required.
- [x] Workloads are persisted, digest-bound, and replayed live.
- [x] Live declarations reconstruct exact paths, modes, starts, and row spans;
      coordinated workload-subset re-signing fails.
- [x] Live pair validation defeats coordinated pair-receipt re-signing.
- [x] Coordinated source/candidate identity mutation plus receipt re-signing
      fails against the live subject-shape authority.
- [x] Source/candidate order rotates across fresh processes.
- [x] Driver, child, and spawned PIDs are explicitly bound.
- [x] Selector and root-metadata mutation fails closed.
- [x] Canonical path containment and complete archive metadata/storage guards.
- [x] Exact atomic publication, validation, staging, and copy receipt fields.
- [x] Physical I/O is honestly absent rather than inferred.
- [x] No result can authorize promotion.

## Still required before promotion

- [ ] Run the matrix against representative short and full-duration archives.
- [ ] Capture actual range/file reads, transferred bytes, and cache state with
      an external physical-I/O tracer.
- [ ] Exercise the maintained downstream consumers that ingest tail kinematics,
      not only the bounded tail-window reader.
- [ ] Benchmark candidate writer and atomic publication costs.
- [ ] Define and approve an explicit promotion gate in a later decision.
- [ ] Change a writer default or selector only in a separately reviewed
      activation checkpoint with rollback evidence.
