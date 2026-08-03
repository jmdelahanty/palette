# Stimulus-step-summary exact Arrow contract — 2026-08-03

Status: implementation checkpoint; exact physical Arrow v1 only. This does not
change a production selector, exporter default, source authority, registry,
recording-local Zarr schema, or physical publication profile.

Coordination base: `f53c0d7a`.

Implementation lane:

- branch: `agent/palette/stimulus-step-summary-arrow-contract-20260803`;
- worktree: `/tmp/palette-stimulus-step-summary-arrow-contract-20260803`;
- owned implementation: the `stimulus_step_summary` logical contract and exact
  Arrow declaration;
- owned tests: exact-Arrow publication, selected-reader, current-producer, and
  two-fish key tests;
- selectors, registries, source writers, exporter defaults, shared catalogs,
  and recording-local contracts remain outside this lane.

## Producer, authority, and lifecycle census

`_load_stimulus_response_tables()` produces `stimulus_step_summary`. It resolves
one recording-local `analysis/stimulus_response_runs/<run>`, iterates each
stimulus step's `per_fish` table, and emits one row per fish per step. Each row
combines:

1. five shared export identity fields;
2. 14 fixed response-run and step fields;
3. six optional protocol-signature fields;
4. the seven-field maintained `STEP_PER_FISH_BASE` bundle;
5. the optional three-field `STEP_BOUT_SUMMARY` bundle; and
6. three optional virtual-collection fields.

The source resolver accepts the maintained compact response schema and legacy
compatible hierarchical representations. Its generic run resolution can still
use the existing latest/sorted compatibility fallback, and the real producer
fixture emits the existing legacy-complete warning. Exact Arrow v1 freezes the
published physical representation; it does **not** promote that source
selection to a new authority contract.

The logical table contract previously claimed one row per recording and step,
with primary key `(recording_id, step_index)`. That declaration was incompatible
with the producer: two fish in one step create two rows with the same old key.
The corrected grain is recording × fish × stimulus step, with primary key
`(recording_id, fish_id, step_index)`. A real two-fish source fixture proves the
two rows retain distinct keys. Selected-generation validation rejects a
manifest that attempts to restore the old two-field declaration. The current
validator does not scan all Parquet values for duplicate keys, so this
checkpoint does not overclaim value-level uniqueness enforcement.

## Exact 38-field physical schema

Order is authoritative. Source `float32` scientific fields become Arrow
`float64`; source `int32` identifiers and counts become Arrow `int64`.

| # | Field | Arrow type | Nullable |
| ---: | --- | --- | :---: |
| 1 | `export_schema_version` | `int32` | no |
| 2 | `table_name` | `string` | no |
| 3 | `recording_id` | `string` | no |
| 4 | `zarr_path` | `string` | no |
| 5 | `source_lineage_hash` | `string` | no |
| 6 | `stimulus_response_run` | `string` | no |
| 7 | `source_stimulus_run` | `string` | no |
| 8 | `source_track_kinematics_run` | `string` | no |
| 9 | `source_track_kinematics_type` | `string` | no |
| 10 | `source_bout_run` | `string` | yes |
| 11 | `step_index` | `int64` | no |
| 12 | `step_name` | `string` | no |
| 13 | `stimulus_mode` | `string` | no |
| 14 | `stimulus_mode_id` | `int64` | no |
| 15 | `start_frame` | `int64` | no |
| 16 | `end_frame` | `int64` | no |
| 17 | `start_camera_frame` | `int64` | no |
| 18 | `end_camera_frame` | `int64` | no |
| 19 | `duration_s` | `float64` | no |
| 20 | `protocol_signature_schema` | `string` | yes |
| 21 | `protocol_signature_hash` | `string` | yes |
| 22 | `derived_protocol_hash` | `string` | yes |
| 23 | `protocol_mode_sequence` | `string` | yes |
| 24 | `protocol_duration_sequence_s` | `string` | yes |
| 25 | `protocol_step_count` | `int64` | yes |
| 26 | `fish_id` | `int64` | no |
| 27 | `total_distance_mm` | `float64` | yes |
| 28 | `mean_speed_mm_s` | `float64` | yes |
| 29 | `median_speed_mm_s` | `float64` | yes |
| 30 | `max_speed_mm_s` | `float64` | yes |
| 31 | `fraction_moving` | `float64` | yes |
| 32 | `coverage` | `float64` | yes |
| 33 | `num_bouts` | `int64` | yes |
| 34 | `mean_bout_duration_s` | `float64` | yes |
| 35 | `mean_interbout_interval_s` | `float64` | yes |
| 36 | `collection_id` | `string` | yes |
| 37 | `collection_manifest_sha256` | `string` | yes |
| 38 | `collection_manifest_path` | `string` | yes |

## Null, fill, units, and interval semantics

- Null is the only portable missing-value representation. There is no generic
  numeric sentinel. Zero, negative scientific values where defined, and empty
  strings are not reinterpreted as missing.
- Non-finite source floating-point metrics are normalized to Arrow null before
  publication. The exact Parquet contract does not publish NaN or infinity as
  missing-value encodings.
- `source_bout_run` and all three bout-summary fields are nullable because the
  step-bout bundle is optional. `num_bouts == 0` is a real zero-bout result;
  `num_bouts == null` means that optional summary was absent.
- The six protocol fields are nullable together because a compatible response
  source can be exported when no recording-local protocol signature was
  resolved. The maintained signed path writes matching
  `protocol_signature_hash` and `derived_protocol_hash` values.
- Scientific per-fish metrics remain nullable because source validity and
  coverage can make a measurement non-finite or unavailable. `fish_id` itself
  is always required.
- `total_distance_mm` is millimetres; speed fields are millimetres per second;
  bout durations and inter-bout intervals are seconds; `fraction_moving` and
  `coverage` are fractions.
- `start_frame`, `end_frame`, `start_camera_frame`, and `end_camera_frame` are
  source-camera frame indexes. Step spans are half-open `[start, end)`. This
  contract does not rebase frames or create a new time axis.
- Deferred producer hardening: the legacy compatibility path currently derives
  camera-frame bounds with `value or step_frame`. An explicit camera frame zero
  is therefore treated as false and replaced by the step-frame fallback. The
  exact schema records the emitted integer but does not endorse that fallback;
  the producer should eventually distinguish `None` from `0` explicitly.
- `source_lineage_hash` binds the selected locator, response run, referenced
  source runs, step index, and fish ID used by the exporter. It is lineage
  evidence, not a digest of every source array value.

## Dynamic and compatibility boundary

The recording-local `step_per_fish` table has a versioned maintained field
vocabulary. Exact Arrow v1 accepts only `STEP_PER_FISH_BASE` and the optional
`STEP_BOUT_SUMMARY` fields. An unknown source array such as `future_metric`
would otherwise flow through `row.update(base)` and silently create a Parquet
column. It now fails before the export manifest or final generation is
published.

Historical exports with inferred schemas remain compatibility artifacts. New
metrics require a reviewed Arrow schema version or a separate normalized table;
they are not admitted opportunistically by observing one recording.

## Publication and consumer boundary

The existing immutable publisher binds this table to:

- the digest-bound exact Arrow envelope in the run manifest;
- exact ordered fields, types, and nullability in every Parquet footer;
- the Arrow schema ID, version, and declaration digest footer metadata;
- the corrected logical table grain and primary key in both manifest and
  footer contracts;
- the manifest-exclusive selected part inventory and per-part digest; and
- a zero-row representation with no placeholder part but the exact declaration
  retained in the immutable manifest.

Tests reject reordered, wrong-type, wrong-nullability, unexpected, missing, and
changed-footer schemas even after the physical part digest and inventory size
are recomputed. Rehashed Arrow-envelope declaration tampering also fails against
the installed exact contract.

There is no dedicated maintained semantic Parquet consumer for this table in
Palette today. Reporting and stimulus-response analysis read recording-local
Zarr. `manifest_selected_part_files()` is only a path resolver for the parts
named by a manifest; it does not validate manifest integrity, part digests, or
Parquet footers. Maintained callers must run `validate_export_run()` before
resolving and reading selected paths. The tests in this checkpoint exercise
the validation, path-selection, and exact-footer/read links without treating
the selector helper as a validating reader; they do not assert that caller
ordering contract themselves.

Parquet publication remains one part per contributing source recording. This
checkpoint does not benchmark or promote a new row-group, compression,
part-sizing, or access-aware profile. Those choices require representative
multi-recording queries and are independent of dtype closure.

## Implementation checklist

- [x] Start from clean integrated commit `f53c0d7a` in a dedicated worktree.
- [x] Census producer, source resolution, maintained field bundles, dynamic
      leak, null/fill semantics, and consumers before editing.
- [x] Correct logical grain and primary key to include `fish_id`.
- [x] Freeze all 38 maintained fields with exact order, Arrow type, and
      nullability.
- [x] Add exact writer, footer, envelope, every-required-field, zero-row, and
      manifest-selected representation tests.
- [x] Add a real two-fish same-step fixture and reject the old logical key in
      selected-generation validation.
- [x] Reject unowned dynamic source metrics before publication.
- [x] Add recomputed-envelope and recomputed-part-inventory tamper tests.
- [x] Complete the relevant outside-sandbox combined suites and static gates.
- [ ] Complete independent read-only review before commit.
- [ ] Commit without pushing, merging, changing selectors, or modifying the
      shared checkout.

## Validation evidence

Outside-sandbox validation:

- analytics logical-contract plus exact-Arrow suites: **118 passed**, with ten
  expected legacy-complete source-selection warnings;
- the initial focused 51-test run had one test-only recording-ID expectation
  mismatch; the producer emitted the filename-derived `two_fish` identity as
  designed, and the corrected focused rerun passed **51/51**;
- four broader exporter integrations produced **2 passed, 2 failed** in both
  the lane and clean pinned parent `f53c0d7a`. Exact node IDs were:
  - `tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_writes_first_tables`
    — failed;
  - `tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_reads_compact_stimulus_response`
    — failed;
  - `tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_can_limit_tables`
    — passed;
  - `tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_can_index_registry`
    — passed;
- the lane command, run from
  `/tmp/palette-stimulus-step-summary-arrow-contract-20260803`, was:

  ```text
  scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_writes_first_tables tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_reads_compact_stimulus_response tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_can_limit_tables tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_can_index_registry -q
  ```

  Result: **2 passed, 2 failed, 4 warnings**.
- the base command, run from the clean
  `/home/delahantyj@hhmi.org/gitrepos/palette` checkout at `f53c0d7a`, was:

  ```text
  scripts/py -m pytest -p no:cacheprovider tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_writes_first_tables tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_reads_compact_stimulus_response tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_can_limit_tables tests/unit/fisheye/test_export_cross_recording_analytics.py::test_export_cross_recording_analytics_can_index_registry -q
  ```

  Result: **2 passed, 2 failed, 4 warnings**. Both failures have the same
  exception and location before this table's Arrow publication:
  `ValueError: Hierarchical bout-kinematics runs require
  legacy_compatibility=True.` from
  `src/fisheye/analysis/bout_kinematics.py:349`, reached through
  `_load_bout_kinematics_metrics()` at
  `src/fisheye/utils/export_cross_recording_analytics.py:1291`. This document
  therefore does not describe the broader run as green;
- Black check, Ruff, `py_compile`, `git diff --check`, and the executable field
  census passed;
- installed exact contract: **38 fields**, **19 non-null**, digest
  `0f51ecb1dd7d73b8fa96c91d1dc5d0f0213bdfa30a6abbfb8a563e40508adadf`.

The independent read-only review and commit remain intentionally pending.
