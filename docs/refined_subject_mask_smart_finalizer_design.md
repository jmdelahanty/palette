# Refined Subject-Mask Smart Finalizer Design

Purpose: define the batch refinement step that turns raw subject-mask model
probabilities into canonical refined subject-mask candidates with component
cleanup, eye identity assignment, QC metrics, and review-routing reasons.

Date anchored: 2026-04-26.

## Problem

The unified U-Net subject-mask path now writes raw `subject_mask_runs/<run>`
probability surfaces for:

- `subject_body`
- `eyes_union`
- `swim_bladder`

Those probabilities are the immutable model output. They are not the final
review/edit surface.

The current probability-only compatibility path can read `mask_probs_roi`,
threshold it, and expose a binary seed mask to refined-subject assembly. That
is enough to create a mechanically valid refined run, but it is not enough to
handle common refinement problems:

- body masks can contain holes, gaps, or detached islands
- swim-bladder masks can contain holes or multiple plausible blobs
- eye-union masks can contain one, two, or more components
- eye identity must be assigned after segmentation, not assumed by the model
- suspicious cleanup needs review-routing reasons instead of silent acceptance

## Boundary

The smart finalizer owns this conversion:

```text
subject_mask_runs/<run>/mask_probs_roi
  -> decoded per-component probability surfaces
  -> thresholded candidates
  -> component-specific cleanup
  -> canonical refined binary components
  -> metrics, reasons, review routing, provenance
  -> refined_subject_masks_runs/<run>
```

It must not mutate or overwrite the raw `subject_mask_runs` evidence.

It must not auto-approve refined components. Approval remains an explicit human
or separately documented auto-approval action.

## Current Versus Target Behavior

Current behavior:

- read `mask_probs_roi` when `masks_roi` is absent
- decode `uint8` probabilities as `0..255 -> 0..1`
- threshold at the run/component threshold, defaulting to `0.5`
- write/copy thresholded binary masks into a refined run
- compute existing refined metrics from the resulting binary masks
- for `eyes_union`, split into `eye_left` and `eye_right` when keypoint
  assignment evidence is available

Target V1 behavior:

- do all current behavior
- apply component-specific spatial cleanup before writing the refined mask
- preserve cleanup provenance and policy details
- write per-row/component reason tags that explain cleanup and uncertainty
- route suspicious rows/components to `needs_review`
- by default, write canonical masks, provenance, reason tags, finalization
  metrics, and cheap topology metrics first
- record phase and chunk timings so scheduler/write changes have a measurable
  baseline
- optionally use Dask worker chunks for disjoint row-range zarr writes
- optionally write expensive shape QC metrics and eye geometry after
  `eye_left` and `eye_right` assignment

## Component Policies

### `subject_body`

Target semantics:

- one fish-body silhouette per ROI row
- holes are usually artifacts
- tiny detached islands are usually artifacts

V1 operations:

1. threshold or hysteresis threshold
2. small binary closing to bridge narrow gaps
3. fill enclosed holes
4. remove small detached components
5. keep the best body component, initially the largest component

Review routing:

- mark `needs_review_removed_high_prob_island` when removed islands carry too
  much high-probability evidence
- mark `needs_review_large_cleanup_delta` when cleanup changes too much area
- mark `needs_review_multiple_components` if component ambiguity remains

### `swim_bladder`

Target semantics:

- one compact internal organ per ROI row when visible
- small holes are usually artifacts
- multiple similarly plausible blobs are suspicious

V1 operations:

1. threshold or hysteresis threshold
2. fill small enclosed holes
3. remove tiny detached components
4. select one compact component when one clearly dominates

Review routing:

- mark `needs_review_ambiguous_component_selection` when no component clearly
  dominates
- mark `needs_review_removed_high_prob_island` when discarded probability mass
  is suspicious
- mark `needs_review_large_hole_fill` when hole filling changes too much area

### `eyes_union`

Target semantics:

- one or two eye components can be valid
- keeping only the largest component is wrong
- the union mask is an input to assignment, not a canonical refined eye output

V1 operations:

1. threshold or hysteresis threshold
2. remove tiny islands
3. keep up to two plausible eye components
4. pass remaining components to left/right assignment

Review routing:

- mark `needs_review_multiple_components` when more than two plausible
  components remain
- mark `needs_review_ambiguous_component_selection` when keypoint assignment
  cannot confidently split the union into left/right masks

### `eye_left` And `eye_right`

Target semantics:

- one component per anatomical eye channel
- identity must be justified by keypoints or another explicit assignment source

V1 operations:

1. derive candidate components from cleaned `eyes_union`
2. assign left/right using canonical eye keypoints
3. write refined `eye_left` and `eye_right` masks
4. run refined-subject eye geometry writing, including ellipse fitting

Ellipse fitting belongs after left/right assignment. A single ellipse fit on
raw `eyes_union` is not semantically meaningful because the union can contain
both eyes.

Review routing:

- mark rows where assignment fails or only one eye is found
- mark rows where ellipse fitting fails or produces poor geometry
- preserve assignment and geometry failure reasons component-locally

## Metrics And Reasons

The finalizer should write the existing refined masks metrics and add
component-local cleanup metrics where possible.

Existing refined metrics remain authoritative for the finalized mask and are
written by the default fast path:

- `metrics/mask_present`
- `metrics/area_px`
- `metrics/centroid_xy`
- `metrics/centroid_valid`
- `metrics/bbox_xyxy`
- `metrics/bbox_valid`
- component topology metrics such as component count and hole fraction

Recommended cleanup metrics are written as finalization metrics in the default
path where available:

- `component_count_before`
- `component_count_after`
- `removed_component_count`
- `removed_area_px`
- `removed_area_fraction`
- `removed_prob_mass`
- `removed_prob_mass_fraction`
- `removed_high_prob_area_px`
- `changed_area_px`
- `changed_area_fraction`
- `hole_count_before`
- `hole_count_after`
- `hole_area_fraction_before`
- `hole_area_fraction_after`
- `quality_code`
- `quality_score`

Expensive shape-QC metrics are optional during finalization:

- default `--metric-level cheap` writes topology metrics and leaves slower
  contour/shape metrics marked as deferred
- `--metric-level full` also computes `sigma_noise`, `curvature_var`, `ipr`,
  and `solidity`

Refined eye geometry/ellipse relations are also optional during finalization:

- default behavior records `eye_geometry_status=deferred`
- `--write-eye-geometry` computes the relation surfaces immediately

Recommended reason tags:

- `clean`
- `cleanup_thresholded_probability`
- `cleanup_closed_gaps`
- `cleanup_filled_holes`
- `cleanup_removed_small_islands`
- `cleanup_kept_largest_component`
- `assigned_from_eyes_union`
- `split_by_keypoint`
- `needs_review_removed_high_prob_island`
- `needs_review_multiple_components`
- `needs_review_large_hole_fill`
- `needs_review_ambiguous_component_selection`
- `needs_review_large_cleanup_delta`
- `needs_review_eye_assignment_failed`
- `needs_review_eye_ellipse_failed`

`quality_code`, `quality_score`, and reason tags are machine-generated
review-routing signals. They are not approval state.

## Dask Execution Model

Use Dask for chunk-level batch computation, not for interactive review events.

Natural task unit:

```text
ROI row chunk:
  read probabilities for rows [start:stop]
  finalize body/swim/eyes for those rows
  assign eyes_union -> eye_left/eye_right for those rows
  compute chunk metrics/reasons/review recommendations
  return chunk result
```

Driver responsibilities:

- open the zarr archive
- resolve source and target runs
- create target `refined_subject_masks_runs/<run>` arrays and attrs
- submit chunk tasks
- write returned chunk results in deterministic row ranges
- write aggregate attrs and provenance
- run refined eye geometry after left/right masks exist
- emit registry status

Current implementation note:

- the default `serial_driver` backend is still available and remains the
  deterministic debug path
- `--execution-backend dask_worker_chunks` lets Dask workers compute and write
  disjoint fixed-shape row chunks while the driver finalizes attrs, reasons,
  provenance, registry status, and optional eye geometry
- `--scheduler` and `--num-workers` control the Dask compute mode when the Dask
  backend is selected; in serial mode they are recorded as instrumentation only

Task responsibilities:

- no zarr metadata mutation
- no registry writes
- no global attrs
- no approval changes
- pure or near-pure computation over an input chunk

This avoids workers racing over zarr group metadata and keeps failed runs
recoverable.

Supported scheduler modes should mirror existing refined-subject tooling:

- `single-threaded` for deterministic debugging and tests
- `threads` for local low-overhead parallelism
- `processes` when CPU cleanup dominates and thread scaling is poor
- `distributed` local cluster for real batch execution when it wins on measured
  workload timing

Recommended initial default:

- keep `single-threaded` explicit for small dry-runs
- benchmark `threads`, `processes`, and local `distributed` before selecting
  the recommended operator path
- promote local `distributed` as the recommended mode if it is faster on real
  finalization workloads and produces byte-equivalent outputs

## Scheduler Benchmark Plan

Scheduler choice should be measured on the actual finalizer workload, not on a
synthetic Dask microbenchmark.

Benchmark modes:

- `single-threaded`
- `threads`
- `processes`
- `distributed` local cluster

Benchmark sizes:

- small subset: `256` or `512` ROI rows
- medium subset: `2k` to `5k` ROI rows
- full canary: the complete arena-2 U-Net subject-mask run, after subset output
  inspection passes

Benchmark controls:

- use the same zarr archive
- use the same source `subject_mask_runs/<run>`
- use the same target output mode
- use the same components
- use the same chunk size
- use the same cleanup policies and thresholds
- do not compare schedulers across different model outputs or source runs

Metrics to record:

- wall time
- rows per second
- scheduler startup time
- probability-read time
- finalizer compute time
- zarr write time
- eye-geometry time
- peak memory when easy to collect
- failed or retried chunk count

Correctness gate:

- masks must be byte-equivalent across schedulers for the benchmark subset
- numeric metrics must be exactly equal or within documented dtype tolerance
- reason labels must match exactly
- component review-routing states must match exactly
- registry emission should be skipped during benchmark runs or performed only
  for the selected final canary output

Output naming:

- benchmark target runs should include the scheduler and subset size in the run
  name
- failed or incomplete benchmark runs should not become `latest`
- the selected final canary run should use a clean operator-facing run name

Decision rule:

- keep `single-threaded` as the deterministic debug mode regardless of timing
- recommend the fastest scheduler that passes the correctness gate on the
  medium subset
- require a full-canary confirmation before making a scheduler the documented
  default for operator commands
- if local `distributed` wins, document it as the preferred real-run path rather
  than treating it as only a future scaling option

### Initial Diagnostic Results

Initial diagnostic tool:

```bash
scripts/py -m fisheye.diagnostics.benchmark_subject_mask_finalizer_schedulers \
  /nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr \
  --source-run subject_masks_unet_registry_gpu_metrics_profile_2026-04-26
```

This diagnostic reads real `mask_probs_roi` chunks and runs spatial
finalization plus `eyes_union -> eye_left/eye_right` assignment, but it does not
write refined zarr outputs or run refined eye ellipse geometry. It is a compute
and source-read benchmark, not a full production writer benchmark.

Before benchmarking schedulers, the body/swim hole-fill implementation was
changed from a pure-Python flood-fill loop to `scipy.ndimage.binary_fill_holes`.
On a 32-row single-threaded smoke, that reduced wall time from about `55.0s` to
about `1.86s`. Scheduler comparisons should use the vectorized hole-fill path.

256-row subset, chunk size `64`, all schedulers:

| Scheduler | Wall seconds | Rows/sec | Output parity |
| --- | ---: | ---: | --- |
| `single-threaded` | `13.89` | `18.43` | baseline |
| `threads` | `4.39` | `58.36` | matched |
| `processes` | `14.83` | `17.26` | matched |
| `distributed` local cluster | `9.10` | `28.12` | matched |

2,048-row subset, chunk size `256`, focused comparison:

| Scheduler | Wall seconds | Rows/sec | Output parity |
| --- | ---: | ---: | --- |
| `threads` | `26.81` | `76.39` | baseline |
| `distributed` local cluster | `33.05` | `61.97` | matched |

Current interpretation:

- `threads` is the fastest mode for the current spatial-finalizer diagnostic.
- local `distributed` is valid and output-equivalent, but did not win on these
  subsets.
- this does not yet settle the final production default because the diagnostic
  does not include refined zarr writes, registry emission, or eye ellipse
  geometry.
- production default selection should be revisited after the writer path exists.

### Full Writer Canary Results

Full real-zarr canary source:

```text
/nvme1/recordings/2026-01-28T23-15-10Z_arena_2_Feeding/zarr/2026-01-28T23-15-10Z_arena_2_Feeding_analysis.zarr
source run: subject_masks_unet_registry_gpu_metrics_profile_2026-04-26
rows: 19,235
mask_probs_roi chunks: (32, 1, 512, 512)
```

Worker-chunk writer, `--chunk-size 64`, `--metric-level cheap`, deferred eye
geometry:

| Scheduler | Workers | Wall seconds | Rows/sec | Notes |
| --- | ---: | ---: | ---: | --- |
| `processes` | `24` | `132.21` | `145.49` | one worker per physical core |
| `distributed` local cluster | `24` | `247.70` | `77.65` | valid, but higher overhead |
| `processes` | `48` | `108.51` | `177.26` | fastest canary on this workstation |

Current operator recommendation for this workstation:

- use `--execution-backend dask_worker_chunks`
- use `--scheduler processes`
- use `--num-workers 48` when the machine is otherwise available
- fall back to `--num-workers 24` if interactive responsiveness, I/O
  contention, or worker memory pressure becomes an issue
- keep `--scheduler distributed` as an explicit diagnostic/scaling option, not
  the default local operator path

The fastest canary was then refreshed with
`fisheye.utils.backfill_refined_subject_eye_geometry`. The latest refined run
contained all four canonical components with `masks_roi = (19235, 4, 512, 512)`
and `eye_geometry_status = computed`; eye-pair separation was valid for
`19233 / 19235` rows.

## Failure And Restart Semantics

V1 should prefer whole-run creation over in-place mutation.

Safe behavior:

- raw source run remains unchanged
- target refined run name is explicit
- existing target requires `--overwrite`
- `--dry-run` plans rows, components, scheduler, chunk count, and expected
  output arrays without writing
- small `--roi-indices` or `--roi-index` subsets can be finalized first for
  visual inspection

If a full run fails partway through:

- the failed target run should be treated as incomplete
- rerun with a new target name or explicit `--overwrite`
- registry emission should happen only after successful finalization

## Temporal QC

Temporal QC should be a second pass, not part of V1 mask repair.

V1 spatial finalization can operate independently per ROI row. Temporal flags
need row-lineage context and neighboring observations.

Later temporal pass:

```text
finalized refined masks
  -> compare neighboring rows within valid lineage
  -> write temporal metrics and reason tags
  -> do not change mask pixels
```

Recommended temporal reason tags:

- `needs_review_area_drop`
- `needs_review_area_spike`
- `needs_review_centroid_jump`
- `needs_review_temporal_gap`
- `needs_review_component_count_jump`

## CLI Shape

The current V1 CLI is a dedicated entrypoint:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  /path/to/analysis.zarr \
  --source-run subject_masks_unet_... \
  --refined-run refined_subject_masks_unet_finalized_... \
  --components subject_body eyes_union swim_bladder \
  --chunk-size 64 \
  --metric-level cheap \
  --execution-backend dask_worker_chunks \
  --scheduler processes \
  --num-workers 48 \
  --dry-run
```

Then create the candidate run:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  /path/to/analysis.zarr \
  --source-run subject_masks_unet_... \
  --refined-run refined_subject_masks_unet_finalized_... \
  --chunk-size 64 \
  --metric-level cheap \
  --execution-backend dask_worker_chunks \
  --scheduler processes \
  --num-workers 48
```

Use `--metric-level full --write-eye-geometry` when the operator explicitly
wants the expensive shape-QC and ellipse relation pass folded into the same
run creation command.

V1 supports explicit `--overwrite`, but does not support ROI-subset writes yet
because a refined subject-mask run is row-aligned with the full source run.

## Implementation Status

Implemented:

- `src/fisheye/refinement/subject_mask_finalization.py`
  - pure component finalization for `subject_body`, `swim_bladder`, and
    `eyes_union`
  - probability thresholding, spatial cleanup, source-seed mask retention,
    quality codes, quality scores, metrics, and reason tags
- `src/fisheye/refinement/finalize_subject_masks.py`
  - creates a new canonical `refined_subject_masks_runs/<run>` from one
    `subject_mask_runs/<run>`
  - reuses the existing refined-subject run schema and provenance writer
  - writes `subject_body`, `eye_left`, `eye_right`, and `swim_bladder` when the
    raw run exposes `subject_body`, `eyes_union`, and `swim_bladder`
  - assigns `eyes_union -> eye_left/eye_right` using canonical keypoints
  - writes refined masks, source-seed masks, run metrics, component metrics,
    and finalization metrics by deterministic row chunks
  - writes component reason labels and body/swim finalization metrics
  - defaults to `--metric-level cheap`, so expensive shape metrics are marked
    deferred instead of blocking canonical mask publication
  - defaults to deferred eye geometry; `--write-eye-geometry` computes ellipse
    relation surfaces during finalization
  - supports `--execution-backend dask_worker_chunks` for Dask worker-written
    disjoint row chunks
  - writes `smart_finalizer_timing_summary` and
    `smart_finalizer_chunk_timings` attrs to expose per-phase and per-chunk
    runtime
  - records Dask instrumentation attrs: `execution_backend`,
    `dask_execution_enabled`, `dask_scheduler`, `dask_num_workers`,
    `dask_chunk_size`, and `dask_version`
  - leaves component and run approval states `pending`
  - emits refined-subject registry status when run through the path-based CLI

Still open:

- visual inspection of the completed full canary before treating the generated
  run as biologically ready for review
- ROI-subset preview runs or an explicit preview artifact type
- finalization metrics for the intermediate `eyes_union` source surface without
  pretending it is a canonical refined component
- a dedicated second-pass command for deferred shape-QC metrics and eye
  geometry

## Implementation Plan

1. Add pure chunk finalization helpers. In progress.
   - input: probability chunk, labels, policies, keypoints
   - output: finalized masks, metrics, reasons, review recommendations

2. Extend `subject_mask_finalization.py`. Done for V1.
   - keep body policy
   - add swim-bladder policy
   - add eyes-union cleanup policy
   - avoid left/right assignment in this module; assignment stays separate

3. Add a batch driver. Sequential chunk-range writer and opt-in
   `dask_worker_chunks` execution are in place.
   - creates target refined run
   - writes deterministic chunks sequentially in the default debug path
   - can submit equivalent Dask worker chunks for disjoint row-range writes
   - can run eye geometry after LR masks exist when explicitly requested

4. Add tests. In progress.
   - in-memory unit tests for each component policy
   - chunk driver tests with fake/in-memory zarr groups
   - real-zarr focused validation outside the Codex sandbox

5. Canary.
   - dry-run full archive
   - run small ROI subset
   - inspect masks and reason tags
   - run full 19k ROI canary
   - rescan registry

## Non-Goals For V1

- no raw probability mutation
- no automatic approval
- no temporal mask repair
- no distributed cluster deployment requirement
- no interactive brush/save event through Dask
- no overwrite of existing curated refined runs unless explicitly requested
