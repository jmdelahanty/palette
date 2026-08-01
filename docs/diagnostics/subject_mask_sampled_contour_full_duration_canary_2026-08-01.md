# Subject-Mask Sampled-Contour Full-Duration Canary

Date: 2026-08-01

Verdict: **Palette publication PASS; Crimson mounted-read and visual gate
pending.**

## Immutable Identity

- Palette implementation commit:
  `9082949197cfc2733af8e551e5b22c5de3d01586`
- LSF job: `153239611`, queue `local`, host `h07u21`, four core slots
- Locked compute worktree:
  `/groups/johnson/johnsonlab/jeremy/gitrepos/palette-worktrees/crop-storage-publication-integration-20260729-90829491`
- Source analysis archive:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/subject_mask_storage/full_duration/sleepyfish_cam2010095_20260731_73f7bb5e/analysis.zarr`
- Source refined run:
  `refined_subject_masks_runs/refined_subject_masks_sleepyfish_subject_mask_full_duration_20260731_73f7bb5e`
- Published cache artifact:
  `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/subject_mask_storage/sampled_contours/sleepyfish_cam2010095_sampled_contours_20260801_90829491`
- Published cache run:
  `cache.zarr/subject_mask_cache_runs/subject_mask_sampled_contours_sleepyfish_20260801_90829491`

The corresponding macOS cache path is:

```text
/Volumes/johnsonlab/jeremy/recordings/.palette_benchmarks/subject_mask_storage/sampled_contours/sleepyfish_cam2010095_sampled_contours_20260801_90829491/cache.zarr
```

This is a selector-ineligible standalone cache canary bound to the already
accepted refined bundle-v2 source. It does not duplicate or mutate the source
archive. Bundle-v3 import/cross-binding/activation is covered by unit tests; a
second full physical copy of every bundle member was intentionally not made.

## Result

- Status: `complete`; fresh post-publication validation returned zero errors.
- Rows: `1,169,010`; camera frames: `1,188,000`.
- Arrays: 12 exact Zarr v3 arrays: `points_xy`, `valid`, and
  `source_point_count` for four components.
- Logical bytes: `2,716,779,240` (2.53 GiB).
- Physical bytes including metadata: `1,321,856,788` (1.23 GiB), about 2.06x
  smaller than logical.
- Files: 355 total, of which the planner predicts 331 payload shard objects.
- Inner chunks: 128 KiB uncompressed for every array.
- Outer shards: at most 8 MiB uncompressed, observation axis only.
- Cache logical-content digest:
  `f119c16841a45005b8617a93ab3a86930af3f6dc9d33e7b4ea4748a02891c725`.
- Cache manifest document digest:
  `fdf724f6b57faea74092ceb8e67ecf6cd2fd05c0951e9bf2e28bde34fdccd3c8`.
- Cache manifest payload digest:
  `c04a3f9283da0bd9bd16497b7ecc9ca9871c816ad517b1e5d40b092bde8c6861`.
- Publication receipt digest:
  `c63cc9508bcb6a86485fc6cf1fc0bbd9e2a1fd924f478033b1cfd5094dfe179c`.
- File SHA-256 values:
  - `canary_manifest.json`:
    `3895f387702255d6c7431a930a33d0c418c839144718779b3694759416e36205`
  - `publication_receipt.json`:
    `f7d016fcc7428736ea3936ee2295aecec9ccfa7aa4cea46dee1c74788b414d99`
  - `cache.zarr/zarr.json`:
    `ff41d6d338c0536df4efdcefa56f3c7761ddfedda967b7dcf49de2074fa874bb`

Exact coverage from the complete `valid` and `source_point_count` companions:

| Component | Valid rows | Invalid rows | Source-point range |
|---|---:|---:|---:|
| subject body | 1,169,010 | 0 | 107–1,553 |
| eye left | 1,168,786 | 224 | 0–74 |
| eye right | 1,168,437 | 573 | 0–70 |
| swim bladder | 1,169,004 | 6 | 0–754 |

Every invalid row in this artifact has `valid=false`,
`source_point_count=0`, and all-NaN sampled points. The general contract retains
the observed source-vertex count even when it is below the two-point validity
minimum.

## Timing and Resources

| Phase | Seconds |
|---|---:|
| PRFS refined source to node-local scratch | 58.89 |
| Source validation and storage planning | 8.29 |
| Four-worker dense contour generation | 1,504.25 |
| Single-owner physical-shard publication | 21.07 |
| Final metadata and bounded sample gate | 0.37 |
| Validated node-local tree to PRFS atomic publication | 7.53 |
| End to end before atomic rename | 1,612.20 |
| LSF runtime | 1,619 (26m 59s) |

LSF measured 10,811.6 CPU-seconds, 5.4 GiB aggregate peak RSS, zero swap,
and a peak of 7.22 logical CPUs (90.21% of the eight hardware threads exposed
by four physical core slots). The cluster couples memory to CPU slots, so the
four-core job carried a 60 GiB reservation even though the process did not need
it.

The first canary also exposed 189 software threads because every worker's
OpenCV import created a host-sized native pool. LSF confined execution to the
assigned CPUs, so correctness and the stored artifact are unaffected. The
post-canary launcher explicitly limits OpenCV and native numerical libraries to
one thread per process; any publication-performance comparison must identify
which threading policy produced it.

## Safety and Validation

- The complete refined run was copied to node-local scratch before compute.
- Contour workers owned disjoint row blocks in node-local memmaps and never
  wrote Zarr.
- One parent process wrote every complete physical output shard.
- The local cache passed logical, manifest, direct/consolidated metadata,
  boundary-sample, codec, and completion checks.
- The hidden PRFS publication copy was validated before one sibling rename.
- A fresh read-only post-publication validator returned `errors=[]`.
- The source archive, production selectors, registries, and the accepted
  subject-mask bundle-v2 canary were not changed.
- Job scratch was removed after successful publication.

Expected Zarr warnings concerned the experimental v3 consolidated-metadata
envelope and the source validation sidecar not being a Zarr hierarchy node.
There were no data, CRC, contract, or lifecycle errors.

## Crimson Gate

Crimson should use the exact refined source and cache paths above and:

1. validate all 12 direct and consolidated declarations with exact dtypes;
2. verify the cache manifest binds the exact refined manifest, dense hash,
   component registry, and row-identity hashes;
3. retain the refined run's `frame_row_offsets` exactly once and use its row
   ranges for every component, including empty and multi-subject frames;
4. open `points_xy` and `valid` for visible contours while leaving
   `source_point_count` lazy during ordinary playback;
5. measure random-frame, 70-frame window, sequential, cancellation, physical
   byte-range, cache, and RSS behavior on the mounted Mac path;
6. prove zero stale visible results and zero post-warmup deadline misses;
7. compare a stratified set of sampled outlines against the dense masks in a
   Metal smoke, including invalid eye/bladder rows; and
8. confirm contour-only overlay reads do not force dense-mask pixel reads.

This canary establishes the physical candidate and publication contract. It
does not yet promote `subject_mask_presentation_candidate_v1`, activate bundle
v3, or claim that the largest external contour preserves holes and disconnected
islands.
