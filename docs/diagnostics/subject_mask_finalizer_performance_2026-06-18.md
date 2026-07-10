# Subject Mask Finalizer Performance Notes - 2026-06-18

## Context

GoodCopBadCop subject-mask inference exposed that GPU inference can complete much faster than the CPU finalization step. The current subject-mask workflow writes U-Net probability masks into `subject_mask_runs/<run>/mask_probs_roi`, then `finalize_subject_masks` materializes canonical refined outputs under `refined_subject_masks_runs/<run>`.

The finalized surface is row-major:

```text
masks_roi[row, component, y, x]
```

One row is one ROI/detection instance. Components share the same row-lineage, crop geometry, keypoint context, and review status context. This is why row-sharded execution is the safest parallelism axis: one worker owns a contiguous row range across all components and writes whole, non-overlapping row chunks.

## What Changed Before This Note

The workflow already had row-sharded CPU finalization through `process_shards`, plus `dask_worker_chunks` and `serial_driver` backends. The slow path was not missing parallelism entirely; it was a mix of:

- core per-row mask cleanup and QC metric work;
- dense Zarr writes for `masks_roi`, optional `source_seed_masks_roi`, and metrics;
- serial postcompute for eye geometry and body/swim contours;
- limited progress/phase telemetry for comparing these pieces.

Recent instrumentation added workflow-level JSONL profile events and a full-finalizer benchmark diagnostic that copies a real subject-mask row slice into a temporary zarr, runs finalization there, and reports phase timings without mutating canonical recording stores.

## Benchmark Evidence

Source recording:

```text
/groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr
```

Source run:

```text
subject_masks_unet_registry_subject_masks_handoff_probe_apply_20260618_02
```

Settings:

```text
execution_backend=process_shards
num_workers=8
chunk_size=256
metric_level=cheap
write_eye_geometry=true
write_component_contours=true
```

Results:

| Rows | Mode | Archive Total | Finalizer Run | Postcompute | Effective Finalizer |
|---:|---|---:|---:|---:|---:|
| 2048 | production | 45.9s | 42.0s | 22.5s | 48.9 rows/s |
| 2048 | sharded postcompute diagnostic | 25.3s | 19.4s | 1.34s | 106.1 rows/s |
| 4096 | production | 88.6s | 81.0s | 43.6s | 50.6 rows/s |
| 4096 | sharded postcompute diagnostic | 44.8s | 36.2s | 2.23s | 113.4 rows/s |
| 4096 | sharded + hole-after shortcut | 41.3s | 32.2s | 2.23s | 127.3 rows/s |
| 4096 | sharded + hole-after shortcut + metric reuse | 39.0s | 30.2s | 2.15s | 135.8 rows/s |
| 4096 | sharded + metric reuse + vectorized spatial metrics/timing split | 38.7s | 29.5s | 2.23s | 139.5 rows/s |
| 4096 | default no source seeds + sharded + metric reuse + vectorized spatial metrics | 37.3s | 28.5s | 2.34s | 144.3 rows/s |
| 4096 | default no source seeds + derived-eye topology seed | 37.5s | 28.6s | 2.27s | 144.0 rows/s |
| 4096 | default no source seeds + cv2 hole filling | 22.7s | 14.1s | 2.12s | 293.2 rows/s |
| 4096 | default no source seeds + vectorized eye split | 21.6s | 13.5s | 1.44s | 305.3 rows/s |
| 4096 | default no source seeds + fast select-components | 21.6s | 12.9s | 1.48s | 321.0 rows/s |
| 4096 | shared mask-geometry refactor | 20.8s | 12.8s | 1.39s | 323.9 rows/s |

Interpretation:

- Sharding postcompute is the biggest demonstrated win. It turns eye geometry + body/swim contour materialization from roughly 40+ seconds per 4096 rows into roughly 2 seconds.
- After postcompute is sharded, the next bottleneck is core `process_shard_compute`.
- A narrow production shortcut that skips impossible `hole_count_after` recomputation when the policy already fills holes improved the 4096-row sharded run from 36.2s to 32.2s for the finalizer run.
- Reusing raw-component topology metrics from `finalize_component_mask()` improved the 4096-row sharded run again, from 32.2s to 30.2s for the finalizer run.
- Replacing the finalizer's per-mask centroid/bbox loop with chunk-vectorized NumPy spatial metrics improved the 4096-row sharded run modestly, from 30.2s to a two-run mean of 29.5s for the finalizer run. This is useful, but it is not the main remaining bottleneck.
- The phase timing values are summed across worker chunks/components, so they are attribution counters rather than direct wall-time percentages. They are most useful for ranking repeated CPU/I/O work inside the finalizer.
- In the two-run vectorized-spatial benchmark, the largest remaining attribution buckets were core `process_shard_compute`, eye assignment, dense `masks_roi`/`source_seed_masks_roi` writes, source-row fingerprinting, and derived eye topology metrics. After the follow-up policy change, `source_seed_masks_roi` is omitted by default and only appears when `--retain-source-seeds` is requested.
- A single 4096-row default no-seed benchmark improved the effective finalizer rate to 144.3 rows/s. This is a modest improvement over the retained-seed two-run mean of 139.5 rows/s, and confirms the policy change removes the expected dense write phases without changing the canonical output surface.
- Seeding derived `eye_left/right` component topology removed full connected-component topology recomputation for assigned eyes. It replaced `compute_topology_metrics_eye_left/right` with narrower `compute_hole_metrics_eye_left/right` phases and preserved assignment status counts, but the 4096-row wall-time benchmark was neutral (`28.5s` -> `28.6s` finalizer run). Keep this as a redundancy/attribution cleanup, not as a major performance win.
- Replacing `scipy.ndimage.binary_fill_holes` with an OpenCV flood-fill based hole helper is the next major demonstrated win. It reduced raw component finalization attribution from roughly `53s` per source component to `15-16s`, cut `process_shard_compute` from `25.9s` to `11.6s`, and improved the 4096-row finalizer run from `28.5s` to `14.1s`. Assignment status counts were unchanged.
- Vectorizing the keypoint split in subbatches and delaying `_measure_mask()`'s pixel-centroid scan until ellipse-fit failure reduced `eye_assignment` attribution from `17.96s` to `13.02s`. The finalizer run improved modestly from `14.1s` to `13.5s` because process-shard wall time is now governed by several similarly sized phases. Assignment status counts were unchanged.
- Optimizing component selection for the common single-component split-mask case reduced `eye_assignment_select_components` from `6.55s` to `1.52s` and total `eye_assignment` attribution from `13.02s` to `7.48s`. The full finalizer run improved from `13.5s` to `12.9s`; status counts remained unchanged.

## Full Recording Validation - 2026-06-19

Validation run:

```text
recording: 2026-06-14T21-12-08Z_arena_1_GoodCopBadCop
subject_run: subject_masks_unet_registry_subject_masks_full_validate_20260619_01
refined_run: refined_subject_masks_smart_finalizer_subject_masks_full_validate_20260619_01
LSF finalization job: 151434577[1] on h06u29
rows: 120,221
```

Finalization settings:

```text
execution_backend=process_shards
num_workers=8
chunk_size=256
postcompute_backend=process_shards
postcompute_chunk_size=256
write_eye_geometry=true
write_component_contours=true
retain_source_seeds=false
stage_finalization_input_to_scratch=true
publish_to_prfs=true
```

Observed workflow timings:

| Phase | Duration |
|---|---:|
| Prepare local staged input | 4.0s |
| Finalizer subprocess | 468.3s |
| Output validation | 0.03s |
| Publish staged refined run to PRFS | 698.5s |
| Validate published output | 0.04s |
| Cleanup local staging | 3.4s |
| Consolidate metadata | 1.3s |
| Total finalization workflow | 1175.7s |

Finalizer attrs on the published refined run confirm:

```text
smart_finalizer_execution_backend=process_shards
smart_finalizer_postcompute_backend=process_shards
smart_finalizer_postcompute_chunk_size=256
smart_finalizer_postcompute_num_workers=8
source_seed_masks_status=omitted
cluster_output_staging.policy=node_local_write_publish_to_prfs
```

The finalizer itself processed the full recording at `260.8 rows/s`.
Sharded postcompute completed in `31.5s` (`3814.9 rows/s`) and wrote eye
geometry plus body/swim contours. The long pole for this completed cluster run
was not postcompute; it was durable publication of the staged refined Zarr back
to PRFS. This is the evidence used to promote `process_shards` as the
subject-mask batch workflow's default postcompute backend.

## Current Optimization Tiers

### Tier 1: Low Risk

1. Promote sharded postcompute from benchmark-only into production behind an explicit option, then make it the batch-workflow default after full-run validation. Implemented behind `--postcompute-backend process_shards`; the subject-mask batch workflow now defaults to `process_shards` while the lower-level finalizer CLI keeps `serial` available for historical-path debugging. Workers compute fixed eye geometry and local contour packs; the parent process merges variable-length contour arrays and writes them deterministically.
2. Reuse cheap topology metrics already computed by `finalize_component_mask()` for raw finalized components instead of recomputing them in `_write_component_metrics_chunk()`. This is now implemented for raw finalized components.
3. Seed assignment-derived `eye_left/right` `component_count` and `largest_component_fraction` from the eye-assignment contract. Assigned eye masks are either empty or one selected foreground component, so only hole metrics still need a mask pass.
4. Vectorize mask-present, area, centroid, and bounding-box metrics for one component row chunk. This replaces the finalizer's old per-mask `np.nonzero()` centroid/bbox loop while preserving the existing output arrays.
5. Use a cv2 flood-fill based hole helper instead of scipy `binary_fill_holes`. This keeps the same filled-mask semantics while avoiding the dominant scipy morphology cost in the per-ROI path.
6. Split finalizer write timing into dense mask writes, source-seed writes, source-row fingerprinting, spatial metric compute/write, component metric writes, QC reason generation, and finalization metric writes. This makes the next bottleneck visible in `smart_finalizer_timing_summary.phase_seconds` and per-chunk timing attrs.
7. Preserve row-sharded process execution. Component-sharding is possible but less attractive because components share row identity and dense chunk writes; row-sharding better matches Zarr write-safety constraints.

### Tier 2: Medium Risk

1. Vectorize centroid and bounding-box metrics across a row chunk. These are currently per-mask loops around `np.nonzero()` and can be expressed with chunk-level reductions.
2. Split pure compute timing from dense Zarr write timing inside worker chunks. Today `process_shard_compute` includes both, which makes CPU and I/O bottlenecks harder to separate.
3. Make `source_seed_masks_roi` diagnostic-only. Implemented behind `--retain-source-seeds`; default production finalization records `source_seed_masks_status="omitted"` and skips the dense seed-mask arrays.

### Tier 3: Research / Larger Design

1. Batch connected-components, hole filling, and contour extraction with a lower-level implementation. OpenCV/scipy APIs are per-image, so true vectorization likely needs numba, C++, CUDA/cuCIM, or a custom kernel.
2. Explore GPU finalization. This is plausible for threshold/morphology, but connected components, holes, contours, and packed variable-length outputs make this a separate project.
3. Materialize subject-mask finalization outputs through temporary local/NRS staging and publish back to PRFS only after validation. This should be considered together with cluster staging and handoff design.

## Implemented Metric-Reuse Slice

Metric reuse for raw finalized components now:

- map `component_count_after` -> persisted `component_count`;
- map `largest_component_fraction_after` -> persisted `largest_component_fraction`;
- map `hole_count_after` -> persisted `hole_count`;
- map `hole_area_fraction_after` -> persisted `hole_area_fraction`;
- use this only when the values came from `finalize_component_mask()`;
- seed assignment-derived `eye_left/right` `component_count` and `largest_component_fraction` from the invariant that assignment outputs are empty or one selected component;
- compute only `hole_count` and `hole_area_fraction` for assignment-derived eyes;
- keep the existing recomputation path for any missing metrics;
- validates with focused unit tests and real-zarr benchmark comparison.

## Implemented Production Postcompute Option

`fisheye.refinement.finalize_subject_masks` now accepts:

```bash
--postcompute-backend serial|process_shards
--postcompute-chunk-size <rows>
--postcompute-num-workers <workers>
```

The option applies only to expensive derived artifacts requested with `--write-eye-geometry` and/or `--write-component-contours`. It does not change canonical mask finalization. The production path keeps deterministic parent-side packed-contour writes: workers compute fixed-shape eye geometry arrays and local contour packs over row shards, then the parent merges variable-length `points_xy` and writes the final contour arrays.

Validation added:

- focused unit coverage checks that serial and `process_shards` postcompute produce matching eye geometry, eye-pair metrics, and body/swim/eye contour arrays on a filesystem zarr;
- summary serialization now accepts both legacy summary objects and sharded dict summaries;
- run-level attrs from the sharded zarr handle are mirrored back onto the main run handle before later attrs are written, preventing zarr attr handoff loss.

## Implemented Spatial Metrics / Timing Attribution Slice

The finalizer now computes these mask-local spatial metrics with a vectorized NumPy row-chunk pass:

- `mask_present`
- `area_px`
- `centroid_xy`
- `centroid_valid`
- `bbox_xyxy`
- `bbox_valid`

This targets the active `metric_level=cheap` workflow. It does not change full shape-QC metrics (`sigma_noise`, `curvature_var`, `ipr`, `solidity`), which remain opt-in through `--metric-level full`.

Chunk timings now include detailed subphases such as:

- `write_masks_roi_<component>`
- `write_source_seed_masks_<component>`
- `compute_source_row_fingerprint_<component>`
- `compute_spatial_metrics_<component>`
- `write_run_spatial_metrics_<component>`
- `write_component_spatial_metrics_<component>`
- `write_component_metric_arrays_<component>`
- `compute_metric_qc_reasons_<component>`
- `write_finalization_metrics_<component>`

Benchmark rerun:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_full_finalizer \
  /groups/johnson/johnsonlab/jeremy/recordings/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop/zarr/2026-06-14T21-12-08Z_arena_1_GoodCopBadCop_analysis.zarr \
  --source-run subject_masks_unet_registry_subject_masks_handoff_probe_apply_20260618_02 \
  --roi-count 4096 \
  --chunk-size 256 \
  --metric-level cheap \
  --execution-backend process_shards \
  --num-workers 8 \
  --postcompute-mode sharded \
  --postcompute-chunk-size 256 \
  --postcompute-num-workers 8
```

`write_source_seed_masks_<component>` appears only when `--retain-source-seeds`
is enabled. Default production runs keep `mask_probs_roi` as the raw durable
model output and skip these dense intermediate arrays.

Next safe step: focus on remaining dense write/fingerprint cost and core
component finalization before making sharded postcompute the cluster default.

## Implemented cv2 Hole-Fill Slice

`subject_mask_finalization._fill_holes()` and `_hole_stats()` now use a padded
OpenCV flood-fill from outside the mask to identify enclosed background pixels.
The padded border preserves the scipy behavior even when foreground touches the
top-left image corner or other image edges. Unit tests compare the new helper
against `scipy.ndimage.binary_fill_holes` for empty, solid, holed,
border-touching, and two-hole masks.

Synthetic cProfile smoke:

- old scipy hole path: 60 synthetic 512x512 masks in `1.58s`;
- cv2 flood-fill path: same masks in `0.44s`.

Real 4096-row benchmark:

- `finalizer_wall_seconds`: `28.5s` -> `14.1s`;
- `rows_per_second`: `144.3` -> `293.2`;
- `process_shard_compute`: `25.9s` -> `11.6s`;
- assignment status counts unchanged: `assigned=4067`,
  `assigned_needs_review=20`, `failed_keypoint_status=9`.

## Eye Assignment Fast-Path Attempt

An attempted `eyes_union` assignment fast path labeled disconnected union
components once and assigned whole components to the nearest left/right
keypoints. The permissive version improved the 4096-row benchmark modestly:

- `eye_assignment`: 17.67s -> 12.34s
- effective finalizer rate: 144.3 -> 147.2 rows/s

It was rejected because it changed review semantics: 20 rows that previously
landed in `assigned_needs_review` became `assigned`. Those rows appear to depend
on the historical pixel-split behavior, likely because the keypoint split plane
cuts ambiguous eye-union components.

An equivalence-guarded version preserved status counts by falling back whenever
a union component crossed the keypoint split plane, but it was slower than the
baseline:

- `eye_assignment`: 17.67s -> 17.98s
- effective finalizer rate: 144.3 -> 142.7 rows/s

Conclusion: do not optimize assignment by bypassing the pixel-split semantics
unless the output/review contract is intentionally changed. Derived
`eye_left`/`eye_right` topology seeding has since removed one redundant
post-assignment topology pass, but the assignment split itself remains expensive
because it still performs per-row pixel splitting, component selection, contour
extraction, and ellipse QC.

## Current Remaining Bottlenecks

After the cv2 hole-fill change, the 4096-row benchmark moved from a finalizer
rate of roughly `144 rows/s` to roughly `293 rows/s`. The next largest phase
attribution buckets are:

- `eye_assignment`: pixel split plus per-eye component selection and ellipse QC;
- raw component finalization: thresholding, connected components, morphology,
  hole stats, and cleanup/review metrics for `subject_body`, `swim_bladder`, and
  `eyes_union`;
- dense writes and row fingerprints: `masks_roi`, component metrics, spatial
  metrics, and source-row fingerprints;
- postcompute: now mostly controlled by sharded eye-geometry/contour
  materialization and no longer the dominant phase in the benchmarked setup.

The `eye_assignment` phase should be read carefully: it is not just assigning
left/right labels. For each successful row it:

- enumerates foreground pixels in the `eyes_union` mask;
- splits pixels by squared distance to left/right eye keypoints, with ties going
  left;
- runs connected-component selection on the left and right split masks;
- runs `_measure_mask()` on each selected eye, which extracts contours and fits
  OpenCV ellipses;
- routes rows to `assigned` or `assigned_needs_review` based on ellipse success.

That review-routing contract is why the faster whole-component fast path was
rejected: it changed review labels on ambiguous rows.

## Suggested Improvement Ladder

### Low Risk: Keep Python/OpenCV Semantics

1. Add finer subphase timing inside `assign_eyes_union_to_lr()`:
   `split_by_keypoint`, `select_components`, `measure_ellipse`, and
   `reason_labels`. Implemented.
2. Avoid redundant scans in `_measure_mask()`. It computes a nonzero centroid
   before `cv2.fitEllipse()`, then overwrites the centroid with the ellipse
   center on success. Implemented by delaying the nonzero centroid scan until
   failure cases.
3. Vectorize the keypoint split in small subbatches with precomputed `x/y`
   coordinate grids:

   ```text
   assign_left = dist(pixel, left_eye) <= dist(pixel, right_eye)
   left_mask = union_mask & assign_left
   right_mask = union_mask & ~assign_left
   ```

   This preserves the current pixel-split and tie-to-left semantics. It should
   be subbatched to avoid large temporary `(N,H,W)` arrays when multiple process
   workers are active. Implemented with strict split-equivalence tests against
   the row-wise reference split.

Current 4096-row subphase attribution after this slice:

- `eye_assignment_split_by_keypoint`: `3.27s`;
- `eye_assignment_select_components`: `6.55s`;
- `eye_assignment_measure_ellipse`: `2.08s`;
- `eye_assignment_reason_labels`: `0.02s`.

The follow-up select-components optimization changed the subphase attribution to:

- `eye_assignment_split_by_keypoint`: `3.03s`;
- `eye_assignment_select_components`: `1.52s`;
- `eye_assignment_measure_ellipse`: `1.93s`;
- `eye_assignment_reason_labels`: `0.02s`.

The selector improvement uses `cv2.connectedComponents` and returns immediately
when a split mask has one foreground component. In the real 4096-row slice, about
`90.8%` of split-eye masks had exactly one component (`7421 / 8174` selector
calls), so the old `connectedComponentsWithStats` centroid path was usually
unnecessary. Multi-component masks still use nearest-centroid selection and are
covered by parity tests against the legacy selector.

### Medium Risk: Shared Compiled Mask-Geometry Kernel

A reusable compiled extension is plausible and likely higher ROI than a bespoke
GPU rewrite. The target should be a shared mask-geometry module, not a
subject-mask-only implementation. Inputs would be row-major `uint8` masks; fixed
outputs would include:

- component count and largest-component fraction;
- selected largest or top-K component masks;
- hole count and hole area fraction;
- area, centroid, and bounding box;
- external contour summary;
- ellipse fit parameters and success/failure codes.

The first version could be a C++/pybind11 wrapper around OpenCV loops. That
would remove Python loop overhead and allow buffer reuse while keeping OpenCV's
connected-component, contour, and ellipse semantics. It should be protected by
strict parity tests against the current Python/OpenCV implementation before
becoming a production dependency.

### High Risk: CUDA / True Batched Geometry

CUDA could accelerate thresholding, morphology, reductions, and maybe connected
components, but contour extraction and ellipse fitting have variable-length
outputs and subtle review semantics. A CUDA path should only be considered after
the Python/OpenCV and C++/OpenCV paths are exhausted. The likely engineering
cost is multi-week and the main risk is silent drift in QC/review labels.

## Recommended Next Step

Do not start with a low-level rewrite. After the select-components optimization,
the next incremental target is the vectorized split itself or the remaining raw
component finalization/write phases. If those are not enough for production
throughput, define the C++/pybind11 mask-geometry kernel as a separate,
parity-tested project.

## Shared Mask-Geometry Surface

The reusable, policy-free pieces have been factored into
`fisheye.shared.mask_geometry` so future segmentation-mask consumers can share
the optimized primitives without inheriting subject-mask review policy:

- `fill_holes()` / `hole_mask()` / `hole_stats()`;
- `connected_component_labels()`;
- `select_component_near_point()`;
- `coordinate_grids()`;
- `mask_pixel_centroid()`.

Subject-mask finalization, subject eye assignment, and refined-eye-mask
measurement now import these helpers. `subject_body_mask_qc` also uses the
shared hole metric for its hole-area QC. Keep anatomical assignment, quality
thresholds, review recommendations, and reason-tag construction in the stage
modules; the shared module should remain geometry-only.

Post-refactor benchmark confirmation:

- `finalizer_wall_seconds`: `12.87s` -> `12.76s`;
- `rows_per_second`: `321.0` -> `323.9`;
- `eye_assignment`: `7.48s` -> `7.23s`;
- assignment status counts unchanged.
> Historical performance log. The current backend decision is documented in
> `subject_mask_finalizer_publication_status_2026-07-09.md`: use
> `process_shards` for production parallel finalization; the Dask finalizer
> implementations have been retired.
