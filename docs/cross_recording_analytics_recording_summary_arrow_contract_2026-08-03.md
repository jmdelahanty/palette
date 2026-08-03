# Recording-summary exact Arrow contract — 2026-08-03

## Decision

`recording_summary` uses exact physical Arrow schema v1 for new immutable
analytics exports. Its 32 payload fields have a closed order, type, and
nullability contract. The logical analytics table remains
`palette.analytics.table.recording_summary` v1 inside analytics export v2;
the independently versioned physical schema is
`palette.analytics_export.arrow_table.recording_summary` v1.

This change moves only `recording_summary` from
`inferred_v2_compatibility_tables` into the manifest's digest-bound
`exact_tables` envelope. It does not change the recording-local analysis
writers, exported row values, default table selection, registries, production
selectors, or any Zarr storage profile.

## Why this table is exact now

The exporter builds one row per recording from a closed set of named fields.
Stimulus-response, swim-bout, and virtual-collection capabilities change which
values are available, but do not define additional columns. By contrast,
other still-inferred default tables currently merge source-defined child
attributes or structured-array columns and require a separate vocabulary
checkpoint before their physical schemas can be frozen.

The former inferred writer selected the union of observed columns and allowed
PyArrow to infer every field as nullable. As a result, two valid exports could
have different physical schemas merely because one cohort lacked an optional
analysis stage. Exact v1 always writes the complete field inventory and uses
nulls for unavailable capability-dependent values.

## Ordered physical schema

| # | Field | Arrow type | Nullable | Meaning |
|---:|---|---|:---:|---|
| 1 | `export_schema_version` | `int32` | no | Analytics export schema version |
| 2 | `table_name` | `string` | no | Exact table name |
| 3 | `recording_id` | `string` | no | Recording identity |
| 4 | `zarr_path` | `string` | no | Source analysis archive path |
| 5 | `source_lineage_hash` | `string` | no | Digest of selected source-run lineage |
| 6 | `stimulus_run` | `string` | yes | Selected stimulus run |
| 7 | `stimulus_response_run` | `string` | yes | Selected stimulus-response run |
| 8 | `swim_bout_run` | `string` | yes | Selected swim-bout run |
| 9 | `stimulus_step_count` | `int64` | no | Resolved step count; zero is valid |
| 10 | `protocol_signature_schema` | `string` | yes | Protocol-signature schema |
| 11 | `protocol_signature_hash` | `string` | yes | Canonical protocol digest |
| 12 | `derived_protocol_hash` | `string` | yes | Deprecated v1 alias of the protocol digest |
| 13 | `protocol_mode_sequence` | `string` | yes | Ordered protocol modes |
| 14 | `protocol_duration_sequence_s` | `string` | yes | Ordered step durations |
| 15 | `protocol_step_count` | `int64` | yes | Protocol-signature step count |
| 16 | `source_track_kinematics_run` | `string` | yes | Response source run |
| 17 | `source_track_kinematics_type` | `string` | yes | Response source scope/type |
| 18 | `source_bout_run` | `string` | yes | Response source bout run |
| 19 | `n_fish` | `int64` | yes | Declared response fish count |
| 20 | `n_steps` | `int64` | yes | Declared response step count |
| 21 | `global_fish_count` | `int64` | yes | Resolved global response rows |
| 22 | `total_distance_mm_sum` | `float64` | yes | Sum of finite global distances |
| 23 | `mean_speed_mm_s_mean` | `float64` | yes | Mean of finite per-fish mean speeds |
| 24 | `fraction_moving_mean` | `float64` | yes | Mean finite moving fraction |
| 25 | `total_active_s_sum` | `float64` | yes | Sum of finite active durations |
| 26 | `swim_bout_default_level` | `string` | yes | Selected default bout level |
| 27 | `swim_bout_default_n_bouts` | `int64` | yes | Default-level bout count |
| 28 | `swim_bout_default_mean_duration_s` | `float64` | yes | Default-level mean duration |
| 29 | `swim_bout_default_total_path_length_mm` | `float64` | yes | Default-level path length |
| 30 | `collection_id` | `string` | yes | Optional virtual-collection identity |
| 31 | `collection_manifest_sha256` | `string` | yes | Optional collection digest |
| 32 | `collection_manifest_path` | `string` | yes | Optional collection manifest path |

`derived_protocol_hash` is retained because the current producer emits it as a
temporary compatibility alias. Exact v1 marks it deprecated but does not
reinterpret or remove it. Removing the field requires a new Arrow table schema
version.

## Empty and missing-capability behavior

- A recording row always has the six non-null fields above.
- Missing stimulus, response, swim-bout, or collection capabilities produce
  nulls in their declared columns; they do not remove columns.
- A recording with no resolved stimulus steps writes
  `stimulus_step_count = 0` and nullable protocol fields.
- An export containing zero `recording_summary` rows publishes zero Parquet
  parts, a zero row count, and the exact table declaration in its manifest.
  No empty placeholder Parquet file is authoritative.

## Validation boundary

The existing exact-schema pipeline applies unchanged:

1. the writer normalizes rows into declared order and rejects unexpected
   fields or null/missing required fields;
2. staged publication validates footer schema, metadata, contract digest,
   inventory, row counts, and cross-part equality before visibility;
3. manifest-selected readers reconstruct the installed contract and reject
   reordered, missing, additional, wrong-type, or wrong-nullability fields;
4. recomputing manifest or file digests cannot legitimize a changed installed
   contract or tampered footer contract metadata.

Historical exports remain readable only through the existing explicit legacy
compatibility path. They are not silently reclassified as exact v1.

## Implementation checklist

- [x] Freeze all 32 fields in producer order.
- [x] Freeze Arrow types and nullability independently of observed values.
- [x] Retain and label the deprecated protocol-hash alias.
- [x] Register `recording_summary` in the exact/inferred manifest partition.
- [x] Keep the generic writer, table defaults, selectors, and registries unchanged.
- [x] Cover exact writer order, type, nullability, and footer digest.
- [x] Cover real stimulus/response/swim and collection-backed export values.
- [x] Cover zero-row publication without placeholder parts.
- [x] Reject unexpected and missing required writer fields.
- [x] Reject rehashed envelope order/type/nullability/membership tampering.
- [x] Run focused exact-schema, exporter, atomic-publication, registry, and
  catalog tests outside the sandbox: 88 combined focused tests plus 11 catalog
  tests passed.
- [x] Complete independent read-only review before commit: ACCEPT.

Four broader exporter tests are currently blocked earlier by strict
bout-kinematics legacy-layout gates. The same four failures reproduce unchanged
at the clean coordination base `451b4af1`, before this Arrow contract is
installed; they are not recording-summary regressions.
