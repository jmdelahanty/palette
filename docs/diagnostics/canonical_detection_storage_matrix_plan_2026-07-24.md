# Canonical Detection Storage Matrix Plan — 2026-07-24

Status: plan-only implementation checkpoint; no cluster submission and no
production-profile promotion.

## Scope

The shared matrix planner now accepts only logical dimensions and byte budgets.
It resolves each label through the canonical detection storage planner,
fingerprints the effective physical stage plan, removes duplicate layouts, and
emits exact exclusive destinations plus deterministic balanced trial orders.
It does not expose per-array chunk-row or shard-row overrides.

The initial representative scales are:

| Scale | Frames | Detection instances | Source image |
| --- | ---: | ---: | ---: |
| `frames_200k` | 200,000 | 199,734 | 4512 x 4512 |
| `frames_full` | 1,188,000 | 1,187,087 | 4512 x 4512 |

The declared sweep contains four regular candidates (`128 KiB`, `512 KiB`,
`1 MiB`, and `2 MiB` inner targets) and sixteen indexed-sharded candidates
(the same inner targets crossed with `8 MiB`, `32 MiB`, `128 MiB`, and
`512 MiB` shard targets).

## Plan Census

The complete two-scale plan produced:

| Scale | Requested labels | Unique physical plans | Removed duplicates | Trials at five repetitions |
| --- | ---: | ---: | ---: | ---: |
| `frames_200k` | 20 | 8 | 12 | 40 |
| `frames_full` | 20 | 12 | 8 | 60 |
| Total | 40 | 20 | 20 | 100 |

Duplicates are expected because a requested byte target can resolve to the
same complete-array chunk or shard shape at a finite logical scale. Every
removed label remains in `matrix.json` with its retained candidate ID, common
physical fingerprint, and reason. No redundant LSF work should be submitted.

## Predeclared Gates

Every measured candidate must preserve exact decoded values, logical schema,
dtypes, frame offsets, immutable source identity, exclusive destinations,
whole physical-unit writes, and both direct and consolidated metadata opens.
Crimson codec compatibility remains mandatory before promotion.

The `1 MiB` regular layout is the same-host, same-tier control. Initial
reduction limits are 1.25x control median write/publication time, 1.10x median
and 1.20x p95 required-read latency, and 1.25x peak RSS. Among candidates that
pass every gate, object count is minimized. These cluster limits do not replace
the required HTTP Range and Crimson/Mac/VPN validation.

## Validation Performed

- The pure two-scale plan rendered as strict JSON with zero destination
  collisions and `payload_io_performed=false`.
- Focused benchmark-contract, diagnostic, and matrix tests passed: 13 tests.
- A fresh 1,000-frame disposable real-Zarr smoke ran outside the sandbox after
  the kernel refactor; all nine array digests were exact, and both direct and
  consolidated opens succeeded.
- The smoke destination was benchmark-only under
  `/tmp/palette-zarr-benchmarks`; no archive, registry, selector, or training
  artifact was changed.

## Next Gate

Create and freeze the cluster-visible noncanonical source fixture, then add the
stage-in/local-canonical/publish lifecycle around these unique candidates. The
first LSF run remains one bounded `frames_200k` repetition block; the full
100-trial matrix is not yet authorized by this checkpoint.
