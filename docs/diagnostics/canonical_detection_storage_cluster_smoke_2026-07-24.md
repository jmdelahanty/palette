# Canonical Detection Storage Cluster Smoke

Status: Phase 8 lifecycle smoke complete; no storage profile selected

Date: 2026-07-24

## Outcome

The first commit-pinned LSF smoke completed successfully for one `200,000`-
frame repetition of the canonical nine-array detection schema. All eight
distinct physical candidates:

- used the same node-local canonical staging store;
- decoded to the exact planned schema, dtypes, shapes, and array digests;
- published through exclusive copy-back into the benchmark-only shared
  namespace;
- reopened through direct and consolidated metadata;
- passed all `72` published-array digest reads; and
- remained immutable at directory mode `0555` and file mode `0444`.

The finalizer published an aggregate with eight candidates and zero registry,
selector, training-artifact, or storage-profile changes. This is one
uncontrolled-cache smoke observation, not enough evidence to select a default.

## Reproducibility

- Palette commit: `421b5ff235add2edc03dce4496702ce4aa236855`
- locked cluster worktree:
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/shared-zarr-storage-policy-20260723-421b5ff2`
- workflow ID: `sleepyfish_det_storage_smoke_20260724_01`
- matrix fingerprint:
  `aad3ca2789feae4d97c3545541cf931275a7f4f4c90eb32fc94df0a7dc9dde6e`
- workflow root:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/canonical_detection_storage/workflows/sleepyfish_det_storage_smoke_20260724_01`
- block job: `153169586[1]`, `DONE`, host `h07u31`
- success-gated finalizer: `153169587`, `DONE`

The block verified that `fisheye` imported from the locked worktree and that
the checkout was clean at the planned commit.

## Lifecycle Evidence

The frozen fixture was copied to:

```text
/scratch/delahantyj/153169586_1/canonical_detection_storage/
sleepyfish_det_storage_smoke_20260724_01/frames_200k_repetition_000
```

Stage-in took `12.325 s` and was excluded from candidate timings. The local
copy matched all `5,809` files, `8,317,265` apparent bytes, and tree SHA-256
`7dbe2bf7b5517990609024923ddede439f61614d582918beeab56ce49c81657d`.
Canonical conversion, exact validation, consolidation, and reopen took
`0.480 s`. Candidate subprocesses then ran sequentially in the recorded
balanced order. The complete block took `36.632 s`; LSF reported `44 s` wall
runtime, `20.54 s` CPU, and `241 MB` maximum RSS.

The exact scratch work-unit directory was absent after the job. An independent
post-run inventory recomputed the same frozen-fixture digest. Both LSF stderr
files were empty.

## Candidate Smoke Measurements

The logical payload was `15,181,920` bytes and every candidate occupied about
`5.61 MB` apparent after Zstd compression. `local total` includes local write,
exact validation, consolidation, and the initial local reads. `publish` is the
exclusive shared copy, tree validation, freeze, rename, and consolidated
reopen. PRFS values sum the nine per-array reads.

| Inner target | Layout | Payload files | Local total (s) | Publish (s) | PRFS 1,024-row windows (ms) | PRFS full arrays (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 KiB | regular | 104 | 0.674 | 1.345 | 33.0 | 70.6 |
| 128 KiB | sharded, 8 MiB target | 8 | 0.614 | 0.323 | 52.8 | 72.8 |
| 512 KiB | regular | 31 | 0.435 | 0.537 | 40.8 | 46.5 |
| 512 KiB | sharded, 8 MiB target | 8 | 0.534 | 0.316 | 59.0 | 54.7 |
| 1 MiB | regular | 17 | 0.528 | 0.436 | 50.1 | 45.4 |
| 1 MiB | sharded, 8 MiB target | 8 | 0.520 | 0.312 | 61.3 | 53.5 |
| 2 MiB | regular | 10 | 0.531 | 0.328 | 58.5 | 47.6 |
| 2 MiB | sharded, 8 MiB target | 8 | 0.572 | 0.339 | 77.7 | 55.9 |

At this scale the larger shard targets collapsed to the same concrete shard
shapes, so the planner correctly removed twelve duplicate labels. Sharding
clearly reduced payload objects, but regular chunks were faster for every
summed PRFS window observation in this single ordered pass. The first regular
candidate also paid a conspicuously larger publication cost. Repetitions and
cache-state separation are required before interpreting either timing pattern.

Direct PRFS group opens took `1.59-1.84 ms`; consolidated opens took
`2.29-2.97 ms`. This tiny nine-array, warm/shared-filesystem result says
nothing about HTTP request count or complete-archive startup. Zarr Python
`3.1.3` also repeated its warning that consolidated metadata is not formally
part of the Zarr v3 specification, so Crimson support remains an explicit
compatibility gate.

## Resource And Observability Follow-up

The smoke requested two CPU slots with `8 GB` per slot, but the site policy
raised the effective allocation to `30 GB` total. CPU peak was only `0.47` and
average CPU efficiency was `11.63%`. Future blocks therefore default to one
slot and a `30`-minute limit; the site minimum is then about `15 GB`, still far
above the observed RSS.

LSF observed `34` threads because this first smoke did not constrain native
library thread pools. The post-smoke planner now supplies and the worker
validates one-thread settings for OpenMP, OpenBLAS, MKL, BLIS, NumExpr, and
vecLib.

The first run also wrote successful runtime status to filenames containing
literal `%J` and `%I`. Those tokens are valid only for LSF log paths, not
Palette runtime paths. The post-smoke planner now uses Palette's explicit
runtime job-ID and array-index tokens. The evidence contents were complete;
the correction makes future filenames operationally useful.

## Next Gate

Do not promote a profile from this smoke. Next:

1. run four more balanced `200,000`-frame repetitions with the fixed thread and
   status environment;
2. add process-cold/warm reads, adjacent-offset plus frame-slice reads, random
   frames, sequential windows, and 700-FPS traversal;
3. carry only nondominated candidates to the full `1,188,000`-frame scale;
4. instrument HTTP Range request count and transferred bytes; and
5. validate finalists through Crimson on the actual Mac/VPN path.
