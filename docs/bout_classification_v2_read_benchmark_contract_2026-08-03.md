# Bout-classification compact-v2 read benchmark contract — 2026-08-03

Status: implemented diagnostic awaiting independent review. The byte-planned
candidate remains selector-ineligible and unpromoted.

## Boundary

This matrix compares two explicit immutable children of
`analysis/bout_classification_runs`:

- a complete, selected compact-v2 source opened through the maintained public
  `resolve_bout_classification_run()` and `load_bout_classification_table()`
  boundary; and
- a complete byte-planned candidate opened only by this private diagnostic
  after `validate_staged_bout_classification_run()` accepts it.

There is no maintained public candidate consumer. The matrix freezes that fact
as `candidate_public_consumer_implemented=false`; it must not use the staged
activation helper to imply otherwise. It also hard-codes
`profile_promoted=false` and `candidate_selector_eligible=false`.

The matrix opens the archive read-only and writes evidence only into a new,
nonsymlinked, benchmark-named directory outside the archive. It never changes
an archive, selector, registry, profile, production default, or canonical
artifact.

## Exact schema

Both roles must be `analysis.bout_classification_runs` v2 and expose exactly
one `per_bout` group with these 20 ordered arrays:

```text
source_bout_id                  int64[n_bouts]
start_frame                     int64[n_bouts]
end_frame                       int64[n_bouts]
window_start_frame              int64[n_bouts]
window_end_frame                int64[n_bouts]
HB1_frame                       int64[n_bouts]
HB1_offset_frames               int32[n_bouts]
category_id                     int32[n_bouts]
category_label_bytes            uint8[n_bouts,64]
subcategory_id                  int32[n_bouts]
sign                            int32[n_bouts]
probability                     float32[n_bouts]
tail_valid_fraction             float32[n_bouts]
traj_valid_fraction             float32[n_bouts]
max_consecutive_tail_invalid    int32[n_bouts]
max_consecutive_traj_invalid    int32[n_bouts]
source_window_valid             bool[n_bouts]
classified                      bool[n_bouts]
valid                           bool[n_bouts]
failure_reason_bytes            uint8[n_bouts,128]
```

The two byte matrices are NUL-terminated UTF-8 with zero padding; their
logical columnar dtypes remain exactly `|S64` and `|S128`. The validator
reconstructs and exactly compares the executable array manifest, ordered field
names, logical dtypes, semantic fill/null descriptions, dimensions, and all 20
physical candidate fill values.

Semantic validation requires unique nonnegative source-bout identities,
well-ordered frame intervals, consistent validity bitmaps, finite fractions in
`[0,1]`, nonnegative invalid-run lengths, and the exact unclassified sentinels:
`HB1_frame=-1`, `HB1_offset_frames=-1`, category/subcategory `-1`, sign `0`,
probability `NaN`, and category label `skipped_invalid_window`. Classified rows
require a nonnegative category, finite probability, and failure reason `ok`.
Their `HB1_offset_frames` must be nonnegative,
`HB1_frame = window_start_frame + HB1_offset_frames`, and the resulting frame
must lie inside the inclusive classification window.

## Scientific and dependency identity

Source and candidate must produce one byte-identical canonical scientific
identity projection. It binds the tail-posture and subject-shape publication
manifest paths and SHA-256 digests; track-motion run, arrays, coordinate
descriptor, and manifest digest; swim-bout run, level, table, and bound source
track-motion digest; and the `per_bout.source_swim_bout_path` attribute. Modern
run, array, and manifest-reference syntax is validated rather than accepted as
arbitrary strings.

The projection also includes the complete classification parameters, tail and
trajectory conversion records, invalid-frame policy, method, adapter, and
classifier identities, input mode, frame rate, window semantics, time-sampling
semantics, and source/valid/invalid/classified counts. Redundant attribute and
parameter values must agree. Only physical storage, writer-owner and
publication timing/completion state, and per-publication provenance instance
details are excluded. A digest-preserving change to a source reference or
scientific parameter therefore invalidates the pair even when both outer
receipts are re-signed.

## Publication and storage proof

The selected source must be the exact `latest` and `latest_complete` child. Its
owner must match the exact guarded-direct lease, generation, parent publication
policy, and selected run path. The byte-planned candidate must have a different
valid owner, complete status, false selector eligibility, the
`explicit_unpromoted_candidate` role, and no pointer mutation.

The candidate root profile and parsed receipt must be exactly the registered
`published_http_v1` profile with the explicit unpromoted-candidate role. The
receipt must round-trip byte-for-JSON, contain exactly 20 entries, and bind the
exact executable candidate declaration for every array. Production schema
validation then replays the declarations, dimensions, access semantics, fixed
fills, physical metadata, Zarr v3 chunks/shards, and codec policy. Merely
recomputing the receipt digest after changing a plan or substituting another
registered profile does not pass.

Both run subtrees require exact direct/root-inline consolidated metadata
equivalence. The validator rejects missing/extra arrays, aliases, groups,
metadata declarations, stale consolidated entries, and descendant symlinks.

## Workload and evidence

Every array is visited twice per trial:

1. its declared eager workload reads the complete array once; and
2. a deterministic bounded row workload reads first, middle, and final spans
   (deduplicated for short arrays).

`windows_per_array` is an exact integer with a minimum of three. Both the
Python API and CLI reject smaller values, so a saved matrix cannot call a
first-only or first/middle-only workload complete. Empty and short arrays still
deduplicate coincident first/middle/final spans without weakening that requested
workload contract.

Each array receipt carries its exact path, dtype, shape, declared access class,
half-open spans, operation count, decoded bytes, per-operation hashes, eager
hash, and aggregate window hash. Complete decoded source/candidate equality is
required. The matrix also exercises and hashes the complete structured public
source table and private candidate reconstruction.

Source/candidate order rotates by repetition. Every role/order position runs
in a new direct child process and binds controller PID, child PID, and live
parent PID. The default complete matrix has five repetitions and ten distinct
children.

The result records wall time, CPU time, peak process RSS, logical decoded
bytes, operation counts, current payload/metadata object counts, apparent
bytes, allocated bytes, environment, and deterministic thread settings. These
runtime observations are digest-protected but not externally attested.
Physical file reads, byte ranges, and transfer bytes remain null until an
OS/filesystem trace exists. Writer and publication phases, representative
scale, consumer promotion, and a promotion gate also remain explicitly
unmeasured.

The diagnostic does not claim to exercise node-local staging, atomic copy, or
production publication. Both fixture runs are created through the genuine
family writer solely to obtain valid logical artifacts; writer and publication
performance remain outside this read matrix.

The pair receipt is deterministic. Final validation rebuilds it from the live
archive, including logical payloads, manifests, storage receipt, metadata,
selector lease, filesystem facts, and workload. It then replays every trial's
array workload. Consequently, coordinated edits followed by re-signing pair,
trial, matrix, storage, metadata, publication, or workload envelopes fail.

## Checklist

- [x] Use the genuine maintained Megabouts writer for both roles.
- [x] Use the public source consumer and truthfully private candidate adapter.
- [x] Freeze all 20 arrays, fixed widths, semantic fills, and exact manifests.
- [x] Replay completion, eligibility, guarded-direct selectors, and storage.
- [x] Bind exact scientific/dependency identities across source and candidate.
- [x] Pin the candidate to the registered `published_http_v1` profile.
- [x] Enforce classified HB1 frame/offset/window arithmetic.
- [x] Require exact direct/consolidated metadata equivalence.
- [x] Visit every array with deterministic eager and bounded row reads.
- [x] Require the frozen first/middle/final workload at API and CLI boundaries.
- [x] Require complete decoded and structured-table equality.
- [x] Rotate five balanced repetitions in ten fresh child processes.
- [x] Record wall/CPU/RSS/object/apparent-byte facts without inventing I/O.
- [x] Live-replay archive, storage, selectors, metadata, and workload.
- [x] Reject aliases, symlinks, unsafe output paths, extra arrays, re-signing,
  forged selectors, dependency/parameter/profile/HB1 tampering, and false
  consumer/I/O/promotion claims.
- [ ] Receive independent ACCEPT before commit.
- [ ] Run a representative immutable source/candidate matrix.
- [ ] Add filesystem tracing before making physical-transfer claims.
- [ ] Implement or explicitly decline a maintained public candidate consumer.

## Invocation

```bash
scripts/py -m fisheye.diagnostics.benchmark_bout_classification_v2_reads matrix \
  /path/to/recording_analysis.zarr \
  --source-run EXPLICIT_SELECTED_V2_RUN \
  --candidate-run EXPLICIT_INELIGIBLE_BYTE_CANDIDATE \
  --output /path/to/.palette_benchmarks/bout_classification_v2_reads_20260803 \
  --repetitions 5
```

The output contains `pair_validation.json`, ten trial receipts, and
`matrix_result.json`. It is diagnostic evidence only.
