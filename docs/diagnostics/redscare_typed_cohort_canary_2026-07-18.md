# RedScare typed cohort canary — 2026-07-18

This was a read-only query of
`/groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite` using
normalized stimulus mode `CHASER` and exact protocol hash
`578a2cd8b3aa5762994b61a2405b94e1cf5012d68c1fa6bfcb76a5e04eb45492`.

Results:

- 168 active source analysis datasets formed the base candidate universe.
- 28 datasets matched both `CHASER` and the exact hash.
- The frozen cohort contains all 28, with no limit or sampling.
- No duplicate active analysis datasets occurred among those recordings.
- A render-only release produced the four-stage dependency DAG without reading
  Zarr array values or mutating the registry.

Normalized biological metadata is not yet present for these 28 recordings. The
coverage audit found one consistent scalar legacy-provenance candidate for each
recording:

- DPF: `7`;
- line/strain: `AB [AB IC] SEPT25`;
- genotype: `AB [AB IC] SEPT25`;
- cross: `18482`.

These compatibility values were deliberately not treated as normalized subject
metadata. A DPF-7 direct-selector canary stopped with 28
`missing_dpf_metadata` blockers and submitted no jobs. The next safe action, if
these values are authoritative, is a separately reviewed dry-run of the existing
subject/dish/cross registry backfill followed by another coverage query.

No production registry rows, recording Zarrs, analytics exports, or LSF jobs
were changed by this canary.
