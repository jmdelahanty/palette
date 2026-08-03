# Stimulus-response compact-tabular-v3 contract

Status: implementation checkpoint, selector-ineligible until a publication
canary is reviewed. Date: 2026-08-03.

## Goal

`palette.stimulus_response` schema version 3 freezes a cross-language table
surface for derived protocol responses. Its physical layout identifier is
`compact_tabular_v3`. It is an opt-in writer layout at this checkpoint; the
existing compact-v2 production default and production selectors are unchanged.

The executable source is
`fisheye.shared.zarr.stimulus_response_schema`. Every persisted field has an
`AnalysisArrayDeclaration` containing its exact path, dtype, axes, authority,
access intent, write mode, fill/null semantics, and physical-policy owner. The
run embeds the exact declaration document in
`attrs["stimulus_response_array_schema"]` and declares its closed optional
bundle set in `attrs["stimulus_response_v3_bundles"]`.

The complete standard surface has 19 tables and at most 310 arrays: 34 core,
12 moving-grating, 136 moving-OMR including the global table, 13 concentric,
90 radial-OMR, and 25 looming arrays. Family and metric identity are encoded by
the closed table path rather than duplicated as fixed strings on every row.

The existing non-byte-planned compact-v3 compatibility surface retains array
declaration schema version 1 exactly. The opt-in byte-planned candidate uses
array declaration schema version 2, which gives every independent table its
own symbolic row axis. Tables such as moving-OMR bouts and looming trials no
longer claim false cardinality equality merely because both are joined to a
protocol step. Candidate markers are an exact all-or-none set; deleting the
profile-role marker cannot downgrade v2 arrays into the compatibility path.

## Required core

Every v3 run contains these columnar tables, including typed zero-row tables
when their row axis is empty:

- `step_index`: exact step identity, int32 mode/index, int64 half-open camera
  frame bounds, float32 duration, fixed-width UTF-8 names and canonical
  stimulus-parameter JSON.
- `global_per_fish`: int32 `fish_id` and float32 recording summaries.
- `step_per_fish`: the exact step identity, int32 `fish_id`, and float32 base
  movement/coverage summaries.

The v3 writer rejects unexpected fields, missing fields, inconsistent row
counts, non-1D table inputs, and dtype substitutions. It never silently drops
a field or widens int32/float32 values to int64/float64. Fixed text is stored
as exact-width two-dimensional uint8 arrays and overflow is an error rather
than truncation.

## Optional all-or-none bundles

The declared bundle list controls the complete table set:

- `frame_annotations`: `frame_annotations`.
- `step_bouts`: the three bout-summary columns in `step_per_fish` plus
  `step_per_bout`.
- `moving_grating`: `grating_per_fish`.
- `moving_grating_omr`: global per-fish, step per-fish, per-bout, windows, and
  early-windows tables.
- `concentric_grating`: `concentric_per_fish`.
- `concentric_radial_omr`: per-fish, per-bout, windows, and early-windows
  tables.
- `looming`: trials, per-trial-per-fish, and per-fish tables.

If a bundle is declared, every table and every declared field is present even
when it has zero rows. A partial bundle is invalid. Dense per-frame and binned
trace duplicates remain intentionally omitted; consumers reconstruct them
from the bound track/stimulus sources unless a future named cache contract is
approved.

## Identity and looming flattening

All per-fish moving-grating, concentric-grating, and looming summaries carry
an explicit int32 `fish_id`. Looming metrics computed as `[fish, trial]` are
persisted as a flat table with the Cartesian row keys `(fish_id, trial_index)`.
The writer verifies every input metric has the exact `[n_fish, n_trials]`
shape and dtype before flattening. This prevents the compact-v2 behavior where
two-dimensional looming fields were silently omitted.

`fish_id` is the input track identity for this analysis. It is not inferred
from table row order and is not a longitudinal animal identity unless the
bound upstream track contract makes that stronger claim.

## Reader and materializer boundaries

The new strict `resolve_stimulus_response_v3_tables()` reader validates schema
identity, layout, bundle set, embedded array manifest, exact table/field sets,
array rank, fixed text width, dtype, and row-count agreement before resolving
logical tables. The existing `resolve_stimulus_response_tables()` entry point
retains its compact-v2/hierarchical-v1 compatibility behavior so this
checkpoint does not break production consumers. Maintained consumers must
migrate to the strict v3 entry point before v3 can become the default.

When explicitly asked to materialize v3, the node-local stimulus-response
materializer accepts only an exact v3 run before atomic publication. Its v2
default retains the existing compatibility validation and selector behavior.
The compatibility reader continues to accept declaration-schema-v1 compact-v3
runs. It does not reinterpret their historical Zarr default fills as the
candidate's stronger semantic-fill contract.

## Physical policy and promotion

The default compact-v2 writer and compact-v3 compatibility writer remain
unchanged. A new explicit candidate is selected only by the complete pair:

```text
--layout compact_tabular_v3 --storage-profile published_http_v1 \
  --no-write-zarr-artifacts
```

The materializer supplies the no-artifact flag itself. A direct CLI invocation
must supply it explicitly. Review PNG/plot arrays are intentionally outside the
scientific table contract and cannot be written beneath a candidate after it
is complete; candidate validation rejects a `visualizations` group. Review
artifacts can remain external sidecars until they receive their own closed
byte-planned cache contract.

That candidate resolves every concrete shape and dtype through
`plan_analysis_storage()` and creates every array through
`create_array_from_plan()`. It therefore derives first-axis chunk and shard
extents from uncompressed bytes rather than a family-wide row constant. The
profile starts from approximately 1 MiB inner chunks and 32 MiB outer shards;
small eager tables become a single whole-array access unit when they fit. A
fixed-width text record is indivisible: its full byte width remains the trailing
access-unit axis. Exact Zarr-v3 codec and indexed-sharding declarations come
from the shared versioned profile and factory.

Physical fills are part of the declaration contract rather than Zarr defaults:

- float metrics use float32 NaN for unavailable values;
- fixed-width text uses uint8 zero padding;
- booleans use false;
- counts use integer zero;
- int8 labels use zero and `quality_flag` uses one;
- identities, indexes, and frame coordinates use integer -1.

The run persists the complete planner receipt, its digest, the exact profile,
and an explicit unpromoted-candidate envelope. Publication reparses and
replans that receipt from live arrays, compares the physical chunks, shards,
codecs, fills, and metadata, and rejects recomputed-digest tampering. Candidate
writes are serial and every sharded plan must declare whole-shard single-writer
ownership; parallel logical row writers are not authorized by this contract.

After all payload, provenance, and completion metadata are final, both the
direct writer and node-local materializer consolidate the archive root and
prove direct versus consolidated equality for the run group, all table groups,
and all arrays. The atomic publisher repeats consolidation and equality on the
destination after its final metadata writes. The exact receipt contains a
digest of normalized current metadata plus its own canonical payload digest;
strict candidate reads recompute it and reject missing, forged, or stale
evidence. Mutable/incomplete reads continue to use direct metadata.

The candidate always remains `stage_selector_eligible=false`, never writes
`latest` or `latest_complete`, and does not register or activate anything. It
cannot overwrite an existing immutable candidate of the same name. It
transitions to terminal `failed` and removes its completion timestamp if final
metadata consolidation or equality validation raises, so a failed seal cannot
leave a direct run marked complete.
It does not change production defaults, selectors, codec profiles, or registry
state. Promotion still requires a selector-ineligible canary, Palette
round-trip evidence, consumer review, and a workload-specific producer/read
benchmark.

## Implementation checklist

- [x] Freeze schema ID/version and layout.
- [x] Freeze exact typed declarations for all 19 possible tables.
- [x] Preserve float32/int32/int64/bool/int8 and fixed-width UTF-8.
- [x] Add `fish_id` to family summaries.
- [x] Flatten looming `[fish, trial]` metrics without loss.
- [x] Reject unexpected, missing, inconsistent, non-1D, and wrong-dtype input.
- [x] Enforce all-or-none optional bundles and typed empty tables.
- [x] Deepen reader and materializer validation.
- [ ] Migrate every maintained downstream consumer to explicit v3 or explicit
  legacy compatibility.
- [x] Add explicit shared-byte-planner and array-factory adoption.
- [x] Freeze semantic fill values and whole-shard-safe serial ownership.
- [x] Persist and executably validate the full physical-plan receipt.
- [x] Prove direct/consolidated metadata equality at local and destination
  publication boundaries.
- [ ] Benchmark the complete v3 writer and its maintained read patterns.
- [ ] Publish and review one selector-ineligible canary.
- [ ] Change the production writer default only after the canary gate.
