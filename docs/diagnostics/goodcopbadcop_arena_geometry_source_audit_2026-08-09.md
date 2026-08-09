# GoodCopBadCop arena-geometry source audit — 2026-08-09

<!-- contract-meta
version: 1
status: active
last_verified: 2026-08-09
implementation: implemented
-->

## Verdict

Every active GoodCopBadCop analysis archive that currently has a selectable
chaser-distance run resolves the fitted dish mask:

- active analysis recordings in the registry: **40**;
- recordings with a selectable chaser-distance run: **33**;
- resolved from `analysis_metadata.dish_mask`: **33/33**;
- nominal-circle or other geometry fallbacks: **0**;
- recordings without `analysis/chaser_distance_runs`: **7**;
- unexpected audit errors: **0**.

The seven unavailable recordings are not evidence of nominal fallback. They have no
chaser-distance parent to audit and therefore cannot currently contribute selected
escape, pursuit, gaze, or bout-response results.

This clears the narrow W1.3 question for the **currently selected chaser-distance
runs**: none resolves nominal geometry today. It does not prove which geometry an
older, previously published derived component used at compute time. W1.2 remains
required so future virtual-control analyses fail closed instead of silently accepting a
nominal fallback.

## Method

Audited at `2026-08-09T22:13:25Z` from Palette `29598056` with:

```bash
scripts/py -m fisheye.diagnostics.audit_arena_geometry_sources \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --protocol-name GoodCopBadCop
```

The diagnostic:

1. opens the registry through SQLite URI `mode=ro` with `query_only=ON`;
2. selects exactly one active analysis archive for each recording whose registered
   protocol is `GoodCopBadCop`;
3. opens each Zarr with direct metadata (`use_consolidated=False`) because this is an
   audit of live source truth, not a published consumer benchmark;
4. resolves only `authoritative_run`, or `latest_complete` when no authority is set;
5. calls the shared `resolve_arena_geometry()` implementation used by analytics; and
6. records the selected run, geometry status/source, QC notes, and any failure.

No registry, Zarr, selector, or production metadata was changed.

## Recording results

All successful rows used selector `latest_complete` and produced no geometry QC notes.

| Recording ID | Selected chaser-distance run | Geometry source |
| --- | --- | --- |
| `2026-05-29T18-11-16Z_arena_1_GoodCopBadCop` | `chaser_distance_gaze_default_canary_20260717_01` | `analysis_metadata.dish_mask` |
| `2026-05-29T18-11-16Z_arena_2_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |
| `2026-05-29T18-11-16Z_arena_3_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |
| `2026-05-29T18-11-16Z_arena_4_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |
| `2026-06-14T21-12-08Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-12-08Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-12-08Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-12-08Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-50-10Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-50-10Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-50-10Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T21-50-10Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T22-33-50Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T22-33-50Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T22-33-50Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-14T22-33-50Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-18-31Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-18-31Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-18-31Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-18-32Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-56-34Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-56-34Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-56-34Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T18-56-34Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T19-28-57Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T19-28-57Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T19-28-57Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T19-28-57Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T21-29-13Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T21-29-13Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T21-29-13Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T21-29-13Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T22-03-24Z_arena_1_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T22-03-24Z_arena_2_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T22-03-24Z_arena_3_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-06-21T22-03-24Z_arena_4_GoodCopBadCop` | `chaser_distance_v1_20260718` | `analysis_metadata.dish_mask` |
| `2026-07-02T15-06-50Z_arena_1_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |
| `2026-07-02T15-06-50Z_arena_2_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |
| `2026-07-02T15-06-50Z_arena_3_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |
| `2026-07-02T15-06-50Z_arena_4_GoodCopBadCop` | unavailable | no `analysis/chaser_distance_runs` |

## Consequences

1. Do not re-run the 33 currently selected chaser-distance sources solely because of
   nominal-circle fallback; this audit found none.
2. Keep W1.2 open. The notes-discarding API still permits a future or different cohort
   to use nominal geometry without a hard failure.
3. Treat the seven unavailable recordings as missing analysis prerequisites, not as
   zero-valued observations and not as geometry failures.
4. Before asserting that historical published escape/pursuit artifacts are clean,
   inspect their persisted geometry-source provenance or regenerate them after W1.2.

