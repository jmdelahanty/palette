# Sleepyfish Keypoint Sharding Canary

**Date:** 2026-07-12
**Status:** clone and direct YOLO-writer canaries complete; candidate passed; default enabled

> Superseded default note (2026-07-14): the `65,536`-row measurements below
> remain historical canary evidence, but the production immutable-keypoint
> default is now `262,144` outer rows. Canonical clipped collection finalization
> uses the same larger grid.

## Scope

The canary cloned one completed immutable clipped-collection keypoint run
without modifying its source:

```text
keypoint_shard_runs/
  keypoint_shard_sleepyfish_kp_allclips_20260708_01_clip_000000
```

The source contains `54,000` ROI rows, five keypoints, and 24 numeric arrays.
Its physical size is only `8,832,358` bytes, but its ordinary Zarr chunks
occupy 742 files, including 717 payload files.

## Candidate Layout

- ROI-domain arrays retain their existing `1,024`- or `16,384`-row inner
  chunks and use aligned `65,536`-row outer shards.
- The three frame-domain arrays retain `1,024`-row inner chunks and use aligned
  `262,144`-row outer shards.
- Every destination outer shard is written in one operation and then reread.
- Every array is validated using a decoded-byte SHA-256 digest.

One complete ROI-domain shard across all 21 ROI arrays is approximately
`25.95 MB`; two writer buffers would require approximately `51.9 MB`. One
frame-domain shard across all three frame arrays is approximately `3.15 MB`.

## Execution

LSF job `153064781` ran on `h07u22` in the `short` queue. The workload ran from
shared-checkout commit `1cb2aaf`; the login host was used only for submission
and status queries. LSF runtime was 21 seconds and maximum accounted memory was
84 MB.

Benchmark artifact and report:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/keypoint_sharding_canary/
  sleepyfish_clip000000_20260712_01/
```

## Results

All 24 arrays passed exact decoded-byte equality. Run attributes and completion
metadata were copied into an isolated group stamped `benchmark_only=true`.

| Measure | Ordinary chunks | Indexed shards | Change |
| --- | ---: | ---: | ---: |
| Total files | 742 | 45 | `16.5x` fewer |
| Payload files | 717 | 20 | `35.9x` fewer |
| Apparent bytes | 8,832,358 | 8,867,727 | `+0.4%` |

Clone plus exhaustive destination validation took `4.26 s`. Sparse all-fill
arrays remained payload-free; each populated ROI array used one physical shard,
and each populated frame-domain array also used one physical shard for this
clip's nonzero range.

Seven repeated warm PRFS reads of `keypoints_roi` were stable:

| Read | Ordinary median | Sharded median | Ratio |
| --- | ---: | ---: | ---: |
| 256 random rows | `266 ms` | `501 ms` | `1.88x` |
| contiguous 1,024 rows | `1.48 ms` | `3.16 ms` | `2.14x` |
| full 54,000-row scan | `22.5 ms` | `49.0 ms` | `2.18x` |

Indexed-sharding codec/index handling roughly doubled these small-array read
times, matching the direction observed for probability-mask sharding. Absolute
latencies remain small, while filesystem-object pressure falls sharply.

## Direct YOLO Writer

The serial YOLO writer has a double-buffered sharded path. It was implemented
in commits `227ebc3` and `0cf0132`. The validated/default layout is:

```text
--keypoint-roi-shard-rows 65536
--keypoint-frame-shard-rows 262144
```

Those values are now defaults. Use `--no-keypoint-sharding` only for an
explicit ordinary-chunk compatibility or benchmark run.

The writer retains the existing inner chunk grid. It accumulates the 13
inference-produced ROI arrays until it owns a complete outer shard, writes that
shard in one operation, and overlaps the write with continued inference using
exactly two buffers. Copied ROI lineage arrays use the same `65,536`-row outer
grid; frame-count arrays use the independent `262,144`-row grid. Variable-width
string arrays are not sharded.

Before a buffer is reused, the writer hashes its decoded source bytes. It then
rereads the published destination slice and requires an identical SHA-256
digest. Run attrs record the requested/effective layout and a
`keypoint_shard_write` summary containing buffer sizes, write and validation
times, per-array hashes, and the aggregate exact-match result.

### Cluster execution

Both full-clip writer jobs ran through LSF from the shared checkout; no workload
ran on the login host.

| Job | Host | Result | Purpose |
| --- | --- | --- | --- |
| `153064806` | `h08u06` | inference/output passed; wrapper exited 1 | exposed stale consolidated metadata during post-run provenance update |
| `153064813` | `h08u12` | `DONE` | corrected end-to-end canary after commit `9bc5fa1` |

The first job produced a complete, internally validated output, but its wrapper
could not rediscover the newly created run through stale consolidated metadata.
Commit `9bc5fa1` made the mutable post-run provenance update open with
`use_consolidated=False`. The second job then completed publication and model
resolution provenance normally.

Corrected canary output:

```text
keypoint_shard_runs/
  keypoint_shard_sleepyfish_kp_sharded_writer_canary_20260712_02_clip_000000
```

Logs:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/keypoint_sharding_writer_canary/
  sleepyfish_clip000000_20260712_01/canary2.153064813.{out,err}
```

### Writer and parity results

- All 24 numeric arrays matched the ordinary reference run exactly in an
  independent cross-run decoded-value audit.
- The run reported `53,993` successful and 7 failed ROI predictions, identical
  to the reference (`99.99%` success).
- All 13 buffered arrays passed the writer's own destination-reread SHA-256
  validation.
- This 54,000-row clip fit in one `65,536`-row outer ROI shard. Each of the two
  buffers occupied `17,928,000` bytes, for `35,856,000` bytes total
  (approximately `34.2 MiB`).
- Physical shard writing took `0.632 s`; exhaustive writer reread/validation
  took `0.692 s`.
- Completion state, publication, registry model resolution, and recorded
  provenance all completed successfully.

### Runtime and storage

The closest ordinary-layout reference took `269.1 s` for inference, `360 s`
of accounted LSF runtime, and `1,917 MB` maximum memory. The corrected sharded
writer took `257.3 s` for inference, `331 s` accounted runtime, and `2,026 MB`
maximum memory. The approximately 4.4% faster inference is best treated as node
variance, not a sharding speedup. More importantly, the sharded writer showed no
runtime regression, and its measured write-plus-validation cost was only
`1.32 s`.

| Measure | Ordinary reference | Direct sharded writer | Change |
| --- | ---: | ---: | ---: |
| Total files | 742 | 45 | `16.5x` fewer |
| Payload files | 717 | 20 | `35.9x` fewer |
| Apparent bytes | 8,832,358 | 8,865,645 | `+0.38%` |
| Allocated bytes | 9,102,848 | 8,881,152 | `-2.44%` |

## Default-Path And Refinement Canary

After indexed sharding became the serial YOLO writer default in commit
`e6d25940`, an end-to-end canary exercised the registry wrapper without
`--keypoint-roi-shard-rows`, `--keypoint-frame-shard-rows`, or the opt-out flag.
The canary used an isolated analysis Zarr and a copied registry so canonical
`keypoints_runs`/`refined_keypoints_runs` publication could be tested without
changing production `latest` pointers or registry rows.

LSF inference job `153065020` ran on `h08u12` in `gpu_l4`. It staged the
existing `14,155,776,000`-byte flat ROI cache to node scratch in `23.30 s`,
completed 54,000-row inference in `201.3 s` at `268.3 poses/s`, and used
1,856 MB maximum accounted memory. Both stderr logs were empty.

The default-path validation confirmed:

- `keypoint_storage_policy=default_indexed_sharding_v1`
- ROI-domain arrays use `65,536` outer rows and frame-domain arrays use
  `262,144` outer rows
- `53,993 / 54,000` predictions succeeded; coordinates contained `539,930`
  nonzero values and confidences contained `269,965` positive values
- parent `latest` and `latest_complete`, run completion, provenance validation,
  stage-array validation, model-resolution provenance, and copied-registry
  keypoint status all passed

LSF refinement job `153065029` then ran on `h07u18` in `short`, reading that
indexed-sharded canonical source through the normal `refine_keypoints` path.
Refinement took `23.38 s`, retained all `53,993` source-success rows, marked
`53,984` rows usable, and used 369 MB maximum accounted memory. Its validation
confirmed all 46 refined arrays had `shards=null`; the refined run was complete,
nonempty, linked to the exact source run, and recorded `ok` in the copied
registry.

Canary archive, scripts, LSF logs, and machine-readable validation reports:

```text
/groups/johnson/johnsonlab/jeremy/recordings/logs/keypoint_default_path_canary/
  sleepyfish_clip000000_20260712_01/
```

## Decision And Next Step

The direct YOLO-writer candidate passed output parity, completion/publication,
file-count, memory, and runtime gates. Indexed sharding is therefore the
default for immutable outputs from the serial YOLO keypoint writer. The
ordinary-chunk path remains available through `--no-keypoint-sharding`.

Refined keypoint runs remain ordinarily chunked editable/review outputs. Their
readers use the standard Zarr array interface and accept either ordinary or
indexed-sharded source keypoint arrays.

Do not infer that this result makes the traditional/Dask writer shard-safe.
Traditional/Dask workers must own complete physical shards, or write
worker-local outputs followed by a deterministic merge; disjoint logical row
slices within one shard are not safe concurrent writes.
