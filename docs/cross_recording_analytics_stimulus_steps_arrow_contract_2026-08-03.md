# Stimulus-steps exact Arrow contract — 2026-08-03

Status: implementation checkpoint; exact physical Arrow v1 only. This does not
change a production selector, exporter default, source authority, registry,
recording-local Zarr schema, or physical publication profile.

Coordination base: `359f072b`.

Implementation lane:

- branch: `agent/palette/stimulus-steps-arrow-contract-20260803`;
- worktree: `/tmp/palette-stimulus-steps-arrow-contract-20260803`;
- owned implementation: `src/fisheye/analytics_exports/arrow_contracts.py`;
- owned tests: the exact-Arrow tests and the current-maintained stimulus source
  fixture in the cross-recording exporter tests;
- shared catalogs, selectors, registries, exporter behavior, and recording-local
  contracts remain outside this lane.

## Producer, authority, and lifecycle census

`stimulus_steps` is produced by `_load_stimulus_steps()` from
`analysis/stimulus_runs/<run>/steps/step_<index>`. It emits one row per named
step, orders steps numerically, uses `step_index` with the group suffix as a
fallback, derives a protocol signature over the ordered source-step metadata,
and optionally receives three collection fields during export publication.

The source selection is not newly authoritative here. The generic `_latest_run`
path still uses `resolve_zarr_run()` with latest and sorted compatibility
fallbacks. During the producer census, an unmarked source emitted the existing
legacy-complete warning. Exact Arrow v1 therefore freezes the exported physical
representation; it does **not** claim that source stimulus selection has become
manifest-bound, selector-strict, or newly approved.

The maintained source writer has two closed child metadata vocabularies:

- moving grating: the 15 attributes written by
  `_write_moving_grating_step_metadata()`;
- concentric grating: the 19 attributes written by
  `_write_concentric_grating_step_metadata()`.

The exporter currently flattens child attributes into prefixed row fields. That
mechanism was formerly unbounded: any new child attribute could silently create
a new inferred Parquet column. Physical v1 closes that leak. Only the maintained
moving and concentric vocabularies below are accepted.

## Exact 60-field physical schema

Order is authoritative. `int32`, `int64`, `float64`, `bool`, and `string` are
Arrow physical types, independently of values observed in one export.

| # | Field | Arrow type | Nullable |
| ---: | --- | --- | :---: |
| 1 | `export_schema_version` | `int32` | no |
| 2 | `table_name` | `string` | no |
| 3 | `recording_id` | `string` | no |
| 4 | `zarr_path` | `string` | no |
| 5 | `source_lineage_hash` | `string` | no |
| 6 | `stimulus_run` | `string` | no |
| 7 | `step_index` | `int64` | no |
| 8 | `step_group` | `string` | no |
| 9 | `step_name` | `string` | yes |
| 10 | `stimulus_mode` | `string` | yes |
| 11 | `stimulus_mode_id` | `int64` | yes |
| 12 | `start_frame` | `int64` | yes |
| 13 | `end_frame` | `int64` | yes |
| 14 | `start_camera_frame` | `int64` | yes |
| 15 | `end_camera_frame` | `int64` | yes |
| 16 | `duration_s` | `float64` | yes |
| 17 | `stimulus_params_json` | `string` | yes |
| 18 | `moving_grating_metadata_schema_version` | `int64` | yes |
| 19 | `moving_grating_source` | `string` | yes |
| 20 | `moving_grating_orientation_degrees_authored` | `float64` | yes |
| 21 | `moving_grating_grating_direction_camera_deg` | `float64` | yes |
| 22 | `moving_grating_camera_to_projector_offset_deg` | `float64` | yes |
| 23 | `moving_grating_direction_mapping_source` | `string` | yes |
| 24 | `moving_grating_direction_mapping_status` | `string` | yes |
| 25 | `moving_grating_direction_mapping_validated` | `bool` | yes |
| 26 | `moving_grating_speed_mm_s` | `float64` | yes |
| 27 | `moving_grating_speed_pps` | `float64` | yes |
| 28 | `moving_grating_spatial_freq_cycles_per_mm` | `float64` | yes |
| 29 | `moving_grating_spatial_freq_rpp` | `float64` | yes |
| 30 | `moving_grating_temporal_frequency_hz` | `float64` | yes |
| 31 | `moving_grating_actual_rendered_temporal_frequency_hz` | `float64` | yes |
| 32 | `moving_grating_duty_cycle` | `float64` | yes |
| 33 | `concentric_grating_metadata_schema_version` | `int64` | yes |
| 34 | `concentric_grating_source` | `string` | yes |
| 35 | `concentric_grating_stimulus_role` | `string` | yes |
| 36 | `concentric_grating_radial_polarity_authored` | `string` | yes |
| 37 | `concentric_grating_radial_sign_authored` | `int64` | yes |
| 38 | `concentric_grating_radial_polarity_source` | `string` | yes |
| 39 | `concentric_grating_radial_polarity_validated` | `bool` | yes |
| 40 | `concentric_grating_speed_mm_s` | `float64` | yes |
| 41 | `concentric_grating_speed_pps` | `float64` | yes |
| 42 | `concentric_grating_spatial_freq_cycles_per_mm` | `float64` | yes |
| 43 | `concentric_grating_spatial_freq_rpp` | `float64` | yes |
| 44 | `concentric_grating_temporal_frequency_hz` | `float64` | yes |
| 45 | `concentric_grating_actual_rendered_temporal_frequency_hz` | `float64` | yes |
| 46 | `concentric_grating_duty_cycle` | `float64` | yes |
| 47 | `concentric_grating_target_radius_min_mm` | `float64` | yes |
| 48 | `concentric_grating_target_radius_max_mm` | `float64` | yes |
| 49 | `concentric_grating_target_radius_source` | `string` | yes |
| 50 | `concentric_grating_centering_success_fraction_threshold` | `float64` | yes |
| 51 | `concentric_grating_coordinate_geometry_status` | `string` | yes |
| 52 | `protocol_signature_schema` | `string` | no |
| 53 | `protocol_signature_hash` | `string` | no |
| 54 | `derived_protocol_hash` | `string` | no |
| 55 | `protocol_mode_sequence` | `string` | yes |
| 56 | `protocol_duration_sequence_s` | `string` | yes |
| 57 | `protocol_step_count` | `int64` | no |
| 58 | `collection_id` | `string` | yes |
| 59 | `collection_manifest_sha256` | `string` | yes |
| 60 | `collection_manifest_path` | `string` | yes |

The logical V2 table declares `(recording_id, step_index)` as its primary key,
and exact Arrow v1 makes both fields non-null with fixed types. The current
publication validator does not scan Parquet values for duplicate logical keys,
so this checkpoint does not overclaim value-level uniqueness enforcement. The
physical row records the selected source path and run through `zarr_path`,
`stimulus_run`, and `source_lineage_hash`. That lineage hash covers the source
locator, run, step index, and source run schema version; it is not a digest of
the step payload or source manifest. `step_group` records the source group name
but does not replace `step_index` as the declared key.

## Null, fill, coordinate, and interval semantics

- Null is the only missing-value representation in declared nullable fields.
  Empty strings, zero, `-1`, false, and NaN are not generic missing sentinels.
- Step label, mode, numeric mode ID, frame bounds, duration, and parameter JSON
  remain nullable because the current compatibility source reader can emit a
  row from incomplete historical step attributes.
- Child fields are nullable because moving and concentric metadata are
  mode-specific. A null moving flag on a concentric row means “not present for
  this mode”; it is not false. Conversely, persisted false validation flags are
  real values and stay false.
- `concentric_grating_radial_sign_authored == -1` is the scientific value for
  contracting, not a missing sentinel.
- `stimulus_params_json` is serialized JSON text when a source mapping/list is
  present. Physical v1 does not reinterpret its nested schema.
- `start_camera_frame` and `end_camera_frame` are source-camera frame indices.
  Downstream step assignment currently interprets step spans as half-open
  `[start_frame, end_frame)`. This Arrow checkpoint does not introduce a new
  frame axis or rebase source frames.
- `duration_s` is seconds. Grating direction fields are degrees; speed fields
  state either millimetres per second or pixels per second in their names;
  spatial frequencies state cycles per millimetre or rendered reciprocal
  pixels; temporal frequencies are hertz.
- `moving_grating_grating_direction_camera_deg` is the authored projector
  orientation plus the declared camera/projector offset. The mapping source,
  status, and validation flag remain explicit; the physical schema does not
  upgrade an unvalidated zero offset into coordinate authority.
- The protocol signature and deprecated `derived_protocol_hash` alias are both
  non-null for every emitted step row and must match in the maintained producer.
  Mode and duration sequences remain nullable when those source values are
  unavailable.

## Dynamic and compatibility boundary

The following are intentionally outside current exact v1:

- legacy fixture field `moving_grating_direction_degrees`;
- legacy fixture field
  `concentric_grating_stimulus_radial_polarity_authored`;
- every `looming_dot_*` field; there is no maintained looming-step metadata
  writer vocabulary to freeze;
- any future child attribute not added through a reviewed schema version.

Rows containing any of these fields fail with an unexpected-field error before
an export manifest or final generation is published. This is an explicit
compatibility quarantine, not silent column removal. A present but empty
`looming_dot` group has no row representation in the current producer because
there are no attributes to flatten. That fact is tested and documented as
current behavior; it is not a supported looming semantic contract.

Historical stimulus exports that require those columns remain historical
inferred-schema artifacts. They do not weaken newly written exact v1. A future
maintained looming producer requires a versioned Arrow schema change or a
separate normalized table, not opportunistic nullable columns.

## Publication and consumer boundary

The existing immutable export publisher now binds this table to:

- the digest-bound exact Arrow envelope in the run manifest;
- the exact ordered fields, types, and nullability in each Parquet footer;
- the Arrow schema ID, version, and payload digest footer metadata;
- the exact manifest-selected part inventory and per-part content digest; and
- a zero-row representation containing no placeholder part but retaining the
  exact table contract in the manifest.

There is no dedicated maintained semantic `stimulus_steps` Parquet consumer in
Palette today. Reporting and stimulus-response processing read recording-local
stimulus Zarr metadata instead. The maintained cross-recording boundary is the
generic manifest-selected Parquet reader. Tests therefore validate the export,
resolve only manifest-selected parts, validate the physical footer, and read
the exact maintained representation. That evidence proves representation and
publication selection; it does not overclaim a new scientific consumer or
source authority.

Parquet part partitioning remains one part per contributing source recording,
as implemented by the existing publisher. This checkpoint did not benchmark or
promote a new row-group, compression, part-sizing, or access-aware physical
profile. Those performance decisions require representative multi-recording
query evidence and remain separate from dtype closure.

## Implementation checklist

- [x] Start from clean integrated commit `359f072b` in a dedicated worktree.
- [x] Census fixed row order, maintained child producers, compatibility source
      selection, dynamic-column risks, and current consumers before editing.
- [x] Freeze all 60 maintained fields with exact order, Arrow type, and
      nullability.
- [x] Exclude fixture-only legacy fields and unknown looming metadata.
- [x] Update the current-maintained export fixture to the exact maintained
      moving and concentric vocabularies.
- [x] Add exact writer, footer, envelope, every-required-field, zero-row, and
      manifest-selected representation tests.
- [x] Add recomputed-envelope and recomputed-part-inventory tests for reordered,
      wrong-type, wrong-nullability, unexpected, missing, and footer-metadata
      tampering.
- [x] Prove legacy moving, legacy concentric, and nonempty looming metadata fail
      before publication; prove empty looming has no current representation.
- [x] Complete focused validation and record base-versus-lane evidence.
- [x] Complete independent read-only review before commit: ACCEPT.
- [x] Commit without pushing, merging, changing selectors, or modifying the
      shared checkout.

## Validation evidence

Focused outside-sandbox validation:

- `test_analytics_export_arrow_contracts.py`: **94 passed**, with seven expected
  legacy-complete source-selection warnings from the representation fixtures;
- current-maintained fixture checks for limited export and registry indexing:
  **2 passed**, with two of the same warnings;
- Ruff, `py_compile`, `git diff --check`, and the executable 60-field/12-required
  field-count check passed;
- installed exact contract digest:
  `c51275fac206f71aa9bc3a6dfcbf1dd9f45a2e92c8df8ea090f4729309bf2916`.

The combined exact-Arrow plus broad exporter run was **not** clean: 98 passed
and four failed. The exact same four nodes and errors were then run in a
temporary detached worktree at untouched base `359f072b`; all four failed there
before this lane's fixture or Arrow changes:

| Test node | Base and lane error |
| --- | --- |
| `test_export_cross_recording_analytics_writes_first_tables` | `Hierarchical bout-kinematics runs require legacy_compatibility=True.` |
| `test_export_cross_recording_analytics_uses_bout_kinematics_source_refs_fallback` | `Unmanifested compact bout-kinematics runs require legacy_compatibility=True.` |
| `test_export_cross_recording_analytics_reads_compact_stimulus_response` | `Hierarchical bout-kinematics runs require legacy_compatibility=True.` |
| `test_export_cross_recording_analytics_reads_compact_bout_kinematics` | `Unmanifested compact bout-kinematics runs require legacy_compatibility=True.` |

Those failures occur while loading bout-kinematics, before Arrow publication.
They are recorded as pinned-base compatibility debt rather than counted as a
clean combined-suite result or repaired in this stimulus-only lane.
