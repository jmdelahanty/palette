# Cross-recording chaser Arrow contracts — 2026-08-04

## Outcome

All 34 tables in the canonical cross-recording analytics catalog now have an
installed, ordered, digest-bound Arrow schema. The canonical envelope contains
zero `inferred_v2_compatibility` tables.

This checkpoint changes the immutable Parquet query representation only. It
does not activate a chaser component, select a production run, modify a
recording Zarr, or promote a storage profile. Seventeen chaser tables remain
unavailable because their independent scientific component authorities are not
yet adopted by the exporter. The existing preflight continues to remove those
tables before any legacy raw-Zarr navigation occurs. Only
`chaser_epoch_spatial_occupancy_zones` is currently reachable.

## Exact schemas

| Table | Fields | Declaration SHA-256 |
|---|---:|---|
| `chaser_epoch_spatial_occupancy_zones` | 59 | `e12735d88de75733f324b81583478fd731d66df6810a28dcc8427bb31ee66e63` |
| `chaser_epoch_distance_summary` | 46 | `865e58c1bfc1ff5fbf815768c6e1d659b2e2337e95fb4c3680052c48951af71c` |
| `chaser_epoch_behavior_summary` | 94 | `0f80c237b92ef75693d08c91dec0b45941ec22b00d72c5d81330b64c91505c86` |
| `chaser_epoch_bout_events` | 66 | `478b692f0881b69220d47c06b3420049aed856fc01819b58eebca0df53961942` |
| `chaser_epoch_bout_histogram` | 68 | `053d07c0b6542e56488de1ba9abb87af5b5d53c832c708ea58380c03af8cd834` |
| `chaser_epoch_inter_bout_interval_histogram` | 68 | `13f6b0d48892b3587d76afaff208b5a5e46a9211c876cb0006194e7314b30a1f` |
| `chaser_epoch_center_distance_histogram` | 58 | `c4ee2fe6a2aad0e8ccf20486ee50d532927aea7f743179ab3c071c93016561af` |
| `chaser_speed_distance_bins` | 50 | `b73b5ae53dcee90ea202540157a38d55aab323e343c4f6be30b4f86978bb9b36` |
| `chaser_epoch_distance_histogram` | 47 | `012c17b2416c31fe9aa7bcd6b92151a84ac42a90e53af8060100830d2f48065e` |
| `chaser_quadrant_occupancy_summary` | 60 | `534f5897a428868acbfd6f312937dd384a2081a913b2c54d4ea4b09738b65a7b` |
| `chaser_quadrant_occupancy_chaser_phase` | 87 | `37217f6f7af7da7198d9da4bbee74bb7d95225a433e574e54f3473fd72a7b42f` |
| `chaser_quadrant_occupancy_density` | 86 | `9deb29be65e020eea3cf1578cf281d2f8fe47f7c882559d0f6ee9e33b2bdbb6e` |
| `chaser_near_field_occupancy_summary` | 72 | `0421a343b18ed67aaae32b3c3908799403f22192f0e6b58cfc97f6b16c4916d2` |
| `chaser_near_field_occupancy_chaser_phase` | 90 | `e5e15edeed73738ad8cf45d66bc8e15c8285e4824dde73d9410d30067d476c68` |
| `chaser_near_field_occupancy_radial_density` | 87 | `484801f9fe6cb876eaee191863572c4b6ea51b2d97c4e0f9633ee8f3583f4601` |
| `chaser_near_field_occupancy_distance_cdf` | 76 | `8e85725ff2bbba58629bb3dbebe033930c81e47387ec1ddc58ef1de3b8b4e450` |
| `chaser_egocentric_epoch_summary` | 69 | `131d96cc42557c3f0d99f727f27c9efab27510789bd66eebac2ab008c29e3d4a` |
| `chaser_egocentric_distance_bearing_histogram` | 70 | `d8d163be8f24a1ca2d2c4c85fcfd02b4d72020f91a646e09ca582b1530a3d010` |

The implementation uses exact shared blocks for export identity, selected
chaser-run lineage, component provenance, and optional collection identity.
Every table then appends a closed table-specific field vocabulary. All logical
primary-key fields are present and non-nullable.

## Decisions

- Structured component tables no longer have an Arrow schema that can expand
  by iterating an observed NumPy dtype. An undeclared persisted field now fails
  exact publication.
- Quadrant and near-field summary mappings are closed projections. Additional
  historical or future summary keys are compatibility data, not automatic
  canonical columns.
- Near-field physical v1 freezes the maintained approach-percentile axis to
  `[5, 10]`, represented by `approach_p05_mm` and `approach_p10_mm` plus their
  percentile-value fields. A future arbitrary percentile contract must use a
  row axis instead of generating column names.
- The egocentric histogram has one `distance_bin_width_mm` and one
  `bearing_bin_width_deg` column. Their component-wide values and per-bin
  values are the same row keys; they are not duplicated in the Arrow schema.
- Nullable compatibility lineage remains visibly nullable. Fields promised by
  independently sealed component manifests remain required, so a dormant
  legacy reader cannot become publishable merely because the Arrow layer was
  frozen.

## Validation

- `226/226` focused Arrow contract, writer, publication, and tampering tests
  passed outside the Codex sandbox.
- `77/77` cross-recording exporter, chaser preflight, and logical-contract
  tests passed outside the sandbox.
- The combined canonical export, atomic publication, group-statistics,
  viewer, baseline-strategy, and training-response matrix passed `424/424`
  active tests with five expected strict xfails for the still-unsealed chaser
  viewer surfaces.
- Every canonical table is present in both `TABLE_CONTRACTS` and
  `ARROW_TABLE_CONTRACTS`; the full canonical envelope has no inferred member.
- Every chaser schema has unique ordered fields, exact types and nullability,
  non-nullable primary keys, exact Parquet footer metadata, and rejection for
  unexpected fields, missing required fields, and duplicate keys.
- The export preflight still reports every unsealed chaser table unavailable
  and never falls through to legacy raw-group navigation.
- Python compilation, Ruff, and `git diff --check` passed.

## Remaining boundary

This does not complete all Arrow work in Palette. Baseline-strategy v1 still
has four inferred tables and whole-training-response v1 still has three. They
need their own closed envelopes and exact schemas. Chaser component consumers
also still need to migrate to manifest-bound component handles before the 17
dormant tables may be re-enabled.
