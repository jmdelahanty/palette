# Refined Detection Local Compaction Integration Checkpoint

Status: **PASS**

This checkpoint exercised the refined-detection delta rollover and immutable
compactor on a copied, selector-ineligible Sleepyfish refined-v1 fixture. It is
an integration and safety result. It is not a full-duration scaling result, a
GUI-concurrency result, or authorization to publish or select compacted runs.

## Scope

- Palette revision: `a0a9056d2e563b52af8859b5556259928b19714b`
  from a clean worktree.
- Source: `/groups/johnson/johnsonlab/jeremy/recordings/.palette_benchmarks/refined_detection_storage/integration/20260128_refined_v1_crimson_20260727_v1/refined.zarr`
- Source run: `refined_detect_shadow_crimson_20260727_v1`
- Local work root: `/tmp/palette-refined-detection-compaction-benchmarks/sleepyfish_refined_local_20260727_v2`
- Source dimensions: 23,287 frames, 22,926 refined instances, and 22,938
  source detections.
- Edit: add one manual instance to previously empty frame 672 using allocated
  `refined_row_id=23287` and the canonical manual `instance_key` allocator.
- Result: a fresh, immutable 22,927-instance refined-v1 snapshot using
  `detection_published_access_aware_v1`.

The existing arrays were copied without coordinate conversion. Normalized
bounding boxes remain authoritative; image boxes and centers remain their
exact derived image-space projections. The source-audit arrays were unchanged.
No camera-to-canvas, homography, physical-coordinate, or subject/track identity
contract participated in this operation.

## Result

| Phase | Seconds |
| --- | ---: |
| Source logical hash | 0.313 |
| Source-to-local copy | 0.462 |
| Open, hash, and validate local copy | 0.186 |
| Source immutability recheck | 0.287 |
| Author delta partition | 0.231 |
| Freeze generation 0 and open generation 1 | 0.210 |
| Compactor proper | 1.042 |
| End to end before driver receipt | 3.029 |

Inside the compactor, base validation took 0.061 seconds, frozen-prefix
verification 0.072 seconds, resolution/sort/offset rebuilding 0.206 seconds,
and immutable publication plus validation 0.701 seconds. The compacted store
contains 61 files and 1,959,950 apparent bytes, compared with 58 files and
1,963,995 apparent bytes for the copied base.

The rollover returned `heavy_compaction_may_begin=true` only after generation
0 was frozen and generation 1 was open. This demonstrates the intended short
edit-path boundary: a live editor can target generation 1 while heavier
compaction consumes the immutable generation-0 prefix. This run did not place
real GUI writes under concurrent compactor load, so latency and responsiveness
under that workload remain a later test.

## Validation

A fresh process reopened the source, copied base, and compacted output and
reported zero errors. It recomputed logical-content digests, checked that only
the declared empty-frame addition changed instance membership, proved the
source-audit arrays unchanged, reconstructed storage plans, reran the complete
refined publication validator, and validated the nested compaction receipt.

```bash
scripts/py -m fisheye.diagnostics.benchmark_refined_detection_compaction \
  --verify-receipt \
  /tmp/palette-refined-detection-compaction-benchmarks/sleepyfish_refined_local_20260727_v2/benchmark_receipt.json
```

The command returned `{"status":"pass","errors":[]}`. The source and local
copy share logical-content digest
`a59b1f720095f304cf210a364f7eb76abe4f0ca55ef29f15d535c090e22494a3`.
The output logical-content digest is
`bee3dce3aa2994ec5601ea8032a3e28a6a8b506336ae0e71680100e4c64bd60a`.

Evidence file SHA-256 values:

- `benchmark_receipt.json`:
  `6580bc88484ee69cd3cd8b68b6727f2ee85aa7a571de0b7586b0fa1d1cc5d800`
- `compaction_benchmark_receipt.json`:
  `67f7c6ea65da6861578911b2d7f97ea60e1fa792e6737c8e16c32a1021a2613f`
- `snapshot_publication_receipt.json`:
  `661da16573b87906cf3265e2bf6f98e2ae669beeb009e8d3dbe4a61089fb5822`

## Safety And Limits

- The shared source was opened read-only and rehashed after copying.
- All writes stayed under the fresh `/tmp` work root.
- Copy-back was not performed.
- No production archive, registry, selector, training artifact, or `/groups`
  checkout was changed.
- The source fixture was used whole rather than sliced to 2,048 frames. It was
  already an exact refined-v1 authority and only about 1.9 MB; slicing it would
  have required inventing a new source authority and lineage contract.
- Timings are one local integration repetition with uncontrolled filesystem
  caches. The receipts record process-lifetime peak RSS, not isolated phase RSS.
- Full-duration behavior, crash injection, concurrent GUI editing, and an
  atomic benchmark copy-back path remain open checkpoints.
