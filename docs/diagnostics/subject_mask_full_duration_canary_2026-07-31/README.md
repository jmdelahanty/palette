# Subject-Mask Full-Duration Canary

Date: 2026-07-31

Status: complete selector-ineligible correctness and publication evidence; not
production activation or Crimson performance evidence

## Immutable execution

- Worker Palette commit: `73f7bb5e9bf840f7a5ce697857f8130a6115bff4`
- Corrected publisher Palette commit:
  `58a010aa9482919026137a969a6b41cfb75d3ddf`
- Plan digest:
  `0fb4451df97b871f6eb187becf053305bf9d1139b817ef7d24e063306fc559a8`
- Run root:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/subject_mask_storage/full_duration/sleepyfish_cam2010095_20260731_73f7bb5e`
- Inference array: `153237178`
- Original refinement array: `153237179`
- Exact refinement recovery array: `153237659`
- Corrected recording publisher: `153237730`
- Independent persisted-bundle validator: `153238285`

The plan covers 22 real clip windows, 1,188,000 acquisition frames, and
1,169,010 crop/keypoint rows. It is benchmark-only and prohibits selector,
registry, authority, and production-path mutation.

## Cluster-outage recovery

All 22 inference partitions completed before the outage. The original
refinement array completed 18 partitions; array elements 2, 3, 5, and 6 exited
after their execution hosts became unavailable. Their LSF failures were
infrastructure/run-limit outcomes, not Palette tracebacks. The original
all-success-dependent publisher `153237180` exited without publishing.

Only two hidden temporary destinations were left by interrupted atomic copies:

- `.clip_000001.291ee5f13f5640cdaea0766e3be7faca.tmp`
- `.clip_000004.66f7ee23d8b848cfbbdee71bbdc0958f.tmp`

Neither was consumable. The four missing canonical partitions were recomputed
as exact recovery array `153237659`; all four completed. The resulting 22
refinement receipts cover `[0, 1,169,010)` exactly once and preserve the
original immutable plan identity.

The first fresh publisher `153237675` then failed closed before exposing any
planned target. It revealed a real contract bug: the raw model profile has
three components (`subject_body`, `eyes_union`, `swim_bladder`), while the
editable refined authority has four (`subject_body`, `eye_left`, `eye_right`,
`swim_bladder`). Bundle v1 had incorrectly required the two registries and
channel dimensions to be equal.

Commit `58a010aa` introduced bundle manifest v2, which binds the raw and refined
component registries independently while requiring exact frame, row, crop, and
instance identity. It retains read validation for legacy v1 bundles. The final
receipt separately records the worker and publisher commits, so the correction
does not misattribute previously completed scientific work.

The inference workers processed 308,075,820,594 source-video bytes and
materialized 306,448,957,440 logical ROI-cache bytes. Median per-window cache
materialization was 171.5 rows/s, and median U-Net inference was 82.4 rows/s.
These are correctness-canary timings: this driver uses one decoder per GPU task
and does not exercise the maintained production DAG's separately benchmarked
multi-session NVDEC cache bundles.

## Completed publication

Corrected publisher `153237730` completed on `h06u01` in 14,680 seconds. The
result payload reports 14,673.299 seconds for finalization itself. LSF recorded
17,920 CPU seconds, 4,896 MiB maximum memory, 2,270.48 MiB average memory, and no
swap. The long phase was a bounded single-writer rematerialization plus a full
scientific-quality pass; memory did not scale with the logical dense surface.

Persisted lifecycle timestamps separate the dominant phases:

| Phase | Approximate wall time |
| --- | ---: |
| Raw probability rematerialization | 3,510.090 s |
| Refined dense-authority rematerialization | 4,878.009 s |
| Full refined-source QC computation | 6,085.959 s |
| Recording-level atomic imports and bundle gate | 181.273 s |

The raw probability tensor is 919,346,872,320 logical bytes (0.836 TiB), and
the refined dense authority is 1,225,795,829,760 logical bytes (1.115 TiB).
Quality intentionally reads the complete refined logical surface once while
deriving metrics and recomputing its source digest.

| Member | Physical bytes | Files | Atomic copy |
| --- | ---: | ---: | ---: |
| Raw probabilities | 5,467,856,548 | 3,457 | 87.216 s |
| Refined dense authority | 929,290,971 | 4,119 | 73.971 s |
| Scientific quality | 43,688,406 | 30 | 0.629 s |

The complete benchmark analysis tree has 7,689 files and 6,728,697,933
apparent bytes. Recording-level member import, cross-binding, repeated
validation, and bundle publication took 181.273 seconds. Every physical import
receipt reports exact source/target file-inventory equality and final
validation success.

The published members are:

- `subject_mask_runs/subject_masks_sleepyfish_subject_mask_full_duration_20260731_73f7bb5e`
- `refined_subject_masks_runs/refined_subject_masks_sleepyfish_subject_mask_full_duration_20260731_73f7bb5e`
- `subject_mask_quality_runs/subject_mask_quality_sleepyfish_subject_mask_full_duration_20260731_73f7bb5e`
- `subject_mask_bundle_runs/subject_mask_bundle_sleepyfish_subject_mask_full_duration_20260731_73f7bb5e`

The final result payload digest is
`9522ffbb8e15039f00a5835bfa34a2dae0653564ae3f2bac767a398cef1b4a30`.
The bundle manifest digest is
`cfcfd2297985bd097546d8642f02cea5d5556df0487dad0ad1d9a9bafea3ed0b`.

## Persisted contract

- Raw `mask_probs_roi` is `uint8[1,169,010,3,512,512]`. It uses independently
  decodable `[4,1,512,512]` inner chunks inside `[1024,1,512,512]` indexed
  shards, with bytes + Zstd and a bytes + CRC32C end index.
- Refined `masks_roi` is the dense editable
  `uint8[1,169,010,4,512,512]` authority. It uses the same inner chunk and
  codec chain inside `[1144,1,512,512]` indexed shards.
- Raw and refined `frame_row_offsets` are `int64[1,188,001]`, with 131,072-row
  inner chunks inside a single 1,310,720-row shard.
- Quality is source-bound to the exact refined manifest and dense-array hash.
  Its component metrics are `float32[1,169,010,4,8]`, with 8,192-row inner
  chunks inside 262,144-row indexed shards.
- Bundle v2 records
  `component_registry_policy=raw_and_refined_bound_independently_v1` and exact
  hashes for shared instance, frame, offset, and crop arrays.

## Independent validation

The final result digest was recomputed with Palette's canonical JSON serializer
and matched exactly. All 44 worker result documents were independently hashed:

- 22/22 inference and 22/22 refinement results are complete;
- both stages cover `[0, 1,169,010)` contiguously and exactly;
- every result matches the plan and the final receipt's ordered digest list;
- each stage has 22 distinct scientific-identity, attempt, and receipt
  digests; and
- every worker result is correctly bound to commit `73f7bb5e`.

Read-only LSF validation job `153238285` reopened and deeply validated the
complete persisted bundle in 29 seconds. It returned `status=valid`, matched all
raw/refined/quality/bundle manifest digests, and confirmed selector
ineligibility. It used 2,716 MiB peak RSS.

A first attempt from a low-memory login process failed to allocate a 156 MiB
raw shard and a 1.12 GiB refined shard while checking bounded row bands. This
is not corruption—the same validator passed unchanged with adequate memory—but
it exposes Python Zarr validator read amplification: a bounded logical sample
may allocate an entire outer shard. This should be optimized or explicitly
resource-sized before using the gate in lightweight coordinator processes.

Direct and inline-consolidated declarations passed the deep validator. All
four runs have completion status `complete`, all member and bundle selector
flags are false, the bundle activation state is `deferred`, and root
`subject_mask_authority` remains absent. No registry or production path was
changed.

## Remaining gates

- Obtain Crimson correctness and mounted-reader performance evidence for this
  full-duration candidate.
- Optimize or resource-bound Python validation of large sharded mask samples.
- Treat the two hidden outage temporary directories as retained forensic
  evidence until an explicit cleanup decision; they are not consumable.
- Do not activate a production profile until the external consumer and
  promotion gates pass.
