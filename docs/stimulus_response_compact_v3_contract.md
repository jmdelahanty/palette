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

## Physical policy and promotion

This checkpoint deliberately retains the existing columnar physical writer;
the declarations say `byte_planner_adopted=false`. It does not change the
production layout default, selectors, codec profile, or registry state.
Promotion requires a selector-ineligible canary, direct/consolidated metadata
equivalence, exact Palette round-trip validation, consumer review, and a
separate storage-plan benchmark. The shared byte planner should replace the
legacy columnar physical owner only after that evidence exists.

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
- [ ] Adopt the shared byte planner and benchmark the complete v3 writer.
- [ ] Publish and review one selector-ineligible canary.
- [ ] Change the production writer default only after the canary gate.
