# Swim-bout and bout-kinematics exact storage contract — 2026-08-03

## Outcome

Maintained compact swim-bout and bout-kinematics writers now persist a closed,
canonical whole-run array declaration in
`attrs["array_schema_manifest"]`. The declaration is an immutable logical
contract, not a physical-profile promotion. It uses the shared
`AnalysisArrayDeclaration` and `ArrayContract` types and leaves chunk/shard
selection with `shared_columnar_storage_policy`.

No selector, registry, codec, chunk, or shard default changes as part of this
checkpoint.

## Maintained swim-bout v8 contract

- Run: `analysis/swim_bout_runs/<run>`.
- Identity: `palette.swim_bout_runs` version 8.
- Layout: `compact_tabular_v2` only.
- Required: 132 exact column arrays spanning candidate/signal indexes, bout,
  peak, interval, summary, histogram, point tables, and the detector trace.
- Optional all-or-none bundle: `signals/frame_indices` when the frame-axis
  contract selects an embedded copy. Reference-mode runs omit it.
- Detector samples are `float32[detector_signal, frame]`; detector IDs are
  `int32[detector_signal]`.
- Columnar fixed strings are physical `uint8[row, utf8_byte]` arrays. Their
  exact physical width is bound in the immutable run manifest. The logical
  column dtype/order stored in each table's `field_dtypes`/`field_names` attrs
  is bound separately, so changing only the decoder declaration also fails.
- Previously value-dependent candidate/parameter/summary text dtypes are fixed
  for new exact v8 outputs (`S256`, `S8192`, and `S64` respectively); decoded
  scientific values and algorithms are unchanged.
- Report-only `visualizations/` and `report_tables/` are explicitly outside the
  scientific array manifest.

The writer builds and validates the manifest before marking the run complete.
The materializer repeats the complete exact validation before atomic
publication. Current v8 readers reject absent or malformed manifests.
Hierarchical and earlier compact layouts require the existing explicit
`legacy_compatibility=True` reader policy.

## Maintained bout-kinematics v7 contract

- Run: `analysis/bout_kinematics_runs/<run>`.
- Identity: `analysis.bout_kinematics_runs` version 7.
- Layout: `compact_tabular_v2` only.
- Required: 111 exact column arrays in `level_index`, `movement_metrics`, and
  `heading_metrics`.
- Optional all-or-none bundle: 45 `eye_gaze_metrics` columns.
- Movement, heading, and gaze floating calculations remain `float64`; the
  checkpoint does not change scientific precision or formulas.
- Fixed strings are the same physical `uint8[row, utf8_byte]` representation.

The producer writes the exact manifest only after all compact tables are
complete and before run completion. Fresh-compute materialization requires it.
The older rematerialization path may still copy a legacy candidate, but its
logical resolver invokes the legacy compatibility path explicitly and does not
claim that candidate satisfies the new exact contract.

## Manifest and validation rules

The manifest envelope has an exact field set:

- `schema_id`, `schema_version`, and `persisted_attribute`;
- `digest_algorithm = sha256_canonical_json_v1`;
- `payload`;
- `payload_digest`.

The canonical payload freezes the run schema/layout, enabled optional bundles,
every exact path, physical dtype, rank and axes, run-specific dimensions,
units, access/write/authority classifications, fill/null semantics, physical
policy owner, and byte-planner adoption state.

Validation fails closed on:

- a missing required or partially present optional bundle;
- an unexpected scientific array;
- wrong dtype, rank, or fixed dimension;
- malformed envelope or digest;
- nested manifest edits even when an attacker recomputes the payload digest;
- a current schema/layout without the exact manifest.

The validator reconstructs the canonical declaration using executable table
dtypes and compares canonical JSON bytes. This makes the digest a corruption
check while the executable reconstruction prevents a recomputed digest from
authorizing an invented contract.

## Implementation checklist

- [x] Reuse `AnalysisArrayDeclaration` and `ArrayContract`.
- [x] Freeze maintained compact path inventories.
- [x] Record exact physical dtypes and ranks, including uint8 string matrices.
- [x] Record axes, units, null/fill semantics, access, write mode, and authority.
- [x] Encode optional bundles as all-or-none.
- [x] Emit strict canonical JSON with a SHA-256 payload digest.
- [x] Reject missing, unexpected, wrong-dtype, wrong-rank, and nested tampering.
- [x] Write and validate manifests before completion.
- [x] Validate fresh materializations before atomic publication.
- [x] Keep old layouts on explicit compatibility paths.
- [ ] Adopt the shared byte planner after its family-wide policy lands.
- [ ] Publish selector-ineligible full-duration canaries for Crimson.
- [ ] Benchmark eager indexes, indexed bout windows, publication time, object
  count, and mounted-reader behavior before physical-profile promotion.
