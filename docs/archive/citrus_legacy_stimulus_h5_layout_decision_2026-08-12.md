# Archived decision: legacy Citrus stimulus H5 layout

**Decision date:** 2026-08-12
**Status:** Archived historical layout; not an acceptance contract for new recordings.

## Decision

The legacy Citrus stimulus H5 layout is retained only for provenance and
explicit compatibility migration. It must not be treated as the source
contract for newly transferred recordings, and the strict Palette importer
must not silently reinterpret it.

The legacy surfaces include, among other historical fields:

- `/tracking_data/bounding_boxes` as a non-empty camera-native dataset;
- `/tracking_data/chaser_states` with schema-v4-style rows and legacy
  coordinate attributes; and
- implicit or incomplete state-to-acquisition identity rather than the
  canonical array-level mappings.

The authoritative contract for future coordinate-bearing output is
[`docs/citrus_stimulus_coordinate_output_contract.md`](../citrus_stimulus_coordinate_output_contract.md).
The executable validation and acceptance logic is
`fisheye.shared.stimulus_coordinate_contract`.

## Why this was archived

The 2026-08-10 17:20:55Z `goodbatbadbat` transfer contains four recordings
with the legacy layout. Each has a non-empty `/tracking_data/bounding_boxes`
dataset and lacks the canonical tracking datasets required to bind stimulus
rows to the authoritative acquisition stream:

- `/tracking_data/stimulus_state_key`
- `/tracking_data/source_acquisition_frame_index`
- `/tracking_data/target_source_acquisition_frame_index`
- `/tracking_data/target_source_acquisition_frame_valid`

The transfer itself completed, but stimulus import failed as designed when the
strict contract guard encountered the legacy bounding-box dataset. A renderer
snapshot alone does not make a coordinate-bearing H5 canonical.

Camera-native bounding boxes remain valid observational data, but they are not
stimulus-coordinate surfaces. If they must be preserved for an older consumer,
they belong in an explicitly named compatibility or camera-observation
artifact, with provenance, rather than in the canonical stimulus-coordinate
surface contract.

## Required shape for new coordinate-bearing output

New acquisition output must provide the contract-defined array-level metadata,
including:

1. a stable `int64 [N, 2]` row key using
   `[chaser_index, stimulus_frame_num]`;
2. explicit source and held-target acquisition-index arrays, including their
   validity mask, derived from Orange's authoritative acquisition identity;
3. a schema-v5 chaser-state surface manifest and coordinate descriptor with
   digests;
4. sealed arena geometry and pixel-frame authority records; and
5. one canonical stimulus row and corresponding frame-metadata identity per
   stimulus state, with shutdown rows represented according to the contract.

No importer fallback, field-name alias, or reinterpretation of
`triggering_camera_frame_id` can substitute for those records.

## Metadata generation and migration path

The repository has two explicitly gated metadata-generation paths. The older
`fisheye.utils.migrate_legacy_batman_stimulus_h5` remains restricted to
`v1.2.1-1491-g5ddcc39-dirty` and its one-row-plus-shutdown layout. The dedicated
`fisheye.utils.migrate_legacy_goodbatbadbat_stimulus_h5` handles the audited
2026-08-10 `v1.2.1-1529-g6827d7c` layout with two chaser rows per stimulus
frame.

Both commands default to a read-only dry run. With `--apply`, they read the raw
H5 and Orange sidecars, build an evidence-bound canonical derivative, and write
a migration receipt under the recording's `derived` tree. The raw artifact is
not replaced. Future acquisition output should still emit the canonical
records at source; the migration utility is a recovery boundary for this exact
historical producer/layout pair, not a permissive importer fallback.

### 2026-08-12 arena-1 recovery canary

The migration was applied to only
`2026-08-10T17-20-55Z_arena_1_goodbatbadbat`. The raw H5 remained at SHA-256
`a9c869aabd3da7b904adb96408347c6e900f02be37cd452ed1aa6dd7791eaee8`.
The immutable canonical derivative has SHA-256
`c287dc413a61ad6c1bd51b96e71e2d0653570c6761a80eaa5b68967508e6565a`.

An isolated import copied all 359,968 chaser rows and passed strict coordinate
evidence loading plus a fresh canonical target-position handoff. That canary
also exposed and corrected a Palette chaser-reader omission: descriptors with
the separate target-source acquisition mapping must resolve that mapping as
part of their exact lineage. The production analysis Zarr was not modified,
and arenas 2--4 were not migrated during this canary.

For immediate partial processing, the organized-recording importer exposes
`--stimulus-metadata-and-calibration-only`. That mode imports stimulus events,
protocol metadata, and selected calibration while omitting coordinate surfaces
that lack canonical array-level identity. It does not generate or bless the
missing coordinate metadata, so it is not a substitute when coordinate-bearing
analysis is required.

## Operational rule

Keep the transferred raw H5 and its failure receipt unchanged. Any repaired
or migrated H5 must be a separately named, digest-bound derivative with an
explicit receipt. New agents should start from the current output contract,
not from the historical reference at
[`src/fisheye/docs/citrus_data_structure_documentation.md`](../../src/fisheye/docs/citrus_data_structure_documentation.md).
