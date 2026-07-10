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
- omit dense per-component `source_seed_masks_roi` intermediates by default;
  retain them only when `--retain-source-seeds` is requested for debugging or
  troubleshooting
- record phase and chunk timings so scheduler/write changes have a measurable
  baseline
- use contiguous process shards for production parallel finalization
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

Component metric groups should advertise their schema explicitly:

- `components/<component>/metrics.attrs["schema_id"] =
  "refined_subject_component_mask_metrics_v1"`
- `components/<component>/metrics.attrs["qc_schema_id"] =
  "refined_subject_component_metric_qc_reasons_v1"`
- `components/<component>/metrics.attrs["qc_policy"]` records the
  component-specific gates used to derive generated metric-QC reason tags

Generated metric-QC reason tags use the `needs_review_metric_*` prefix. This
lets refresh/backfill tools replace generated metric-QC tags without removing
manual/operator tags such as `manual_correction`.

`source_seed_masks_roi` is a diagnostic retention surface, not a canonical
production requirement. The default finalizer records
`source_seed_masks_status="omitted"` and keeps the raw model probabilities plus
finalized masks/metrics/reasons as durable evidence. Use
`--retain-source-seeds` to write `components/<component>/source_seed_masks_roi`
when a run needs seed-vs-final troubleshooting.

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

Existing refined-subject runs can refresh mask-local metrics and generated
metric-QC reason tags without recreating the masks:

```bash
scripts/py -m fisheye.utils.backfill_refined_subject_mask_metrics \
  /path/to/archive_analysis.zarr \
  --refined-run <run> \
  --metric-level cheap
```

Use `--metric-level full` when the expensive shape-QC metrics are needed, and
`--no-refresh-reason-tags` when only numeric arrays should be recomputed. For
large archives, metric refresh is currently a sealed serial maintenance pass;
the parallel production path applies when creating the refined run with
`--execution-backend process_shards`.

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

## Parallel Execution Model

Use `process_shards` for production parallel finalization. Keep
`serial_driver` as the deterministic correctness and debugging fallback.

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
- partition the run into contiguous, whole-physical-chunk-aligned shards
- merge sealed metrics and packed variable-length outputs deterministically
- write aggregate attrs and provenance
- run refined eye geometry after left/right masks exist
- emit registry status

Worker responsibilities:

- open the source and target Zarr once per contiguous shard
- own whole, non-overlapping physical chunks for every directly written array
- no registry writes
- no global attrs
- no approval changes
- finalize body, swim bladder, and eye masks for the owned rows
- return sealed finalization-metric payloads and optional packed postcompute
  payloads to the driver

This avoids workers racing over zarr group metadata and keeps failed runs
recoverable.

### Backend Decision

The retired delayed-task and Dask-array implementations were slower than
`process_shards` on production-matched real-data benchmarks. The Dask-array
thread canary was about 2.5 times slower on 4,096 rows; the process canary was
stopped after 536 seconds with low CPU utilization and high memory use. The
full staged delayed-task Dask process run was also slower than
`process_shards` (451.72 versus 279.18 seconds). Detailed evidence is preserved
in `docs/diagnostics/subject_mask_finalizer_publication_status_2026-07-09.md`.

The supported decision is therefore:

- `process_shards` is the only production parallel backend
- `serial_driver` is retained only for deterministic correctness/debugging
- Dask finalizer backends and scheduler options are not part of the active API
- a future alternative must begin as a separate benchmark canary and pass
  output-parity, throughput, memory, and publication-layout gates before it is
  added to production code

## Failure And Restart Semantics

V1 should prefer whole-run creation over in-place mutation.

Safe behavior:

- raw source run remains unchanged
- target refined run name is explicit
- existing target requires `--overwrite`
- `--dry-run` plans rows, components, backend, chunk count, and expected
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
  --chunk-size 256 \
  --metric-level cheap \
  --execution-backend process_shards \
  --num-workers 16 \
  --dry-run
```

Then create the candidate run:

```bash
scripts/py -m fisheye.refinement.finalize_subject_masks \
  /path/to/analysis.zarr \
  --source-run subject_masks_unet_... \
  --refined-run refined_subject_masks_unet_finalized_... \
  --chunk-size 256 \
  --metric-level cheap \
  --execution-backend process_shards \
  --num-workers 16
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
  - supports `--execution-backend process_shards` for row-sharded worker
    processes that open the zarr once per shard and write disjoint row chunks
  - supports `--postcompute-backend process_shards` for expensive derived
    artifacts requested by `--write-eye-geometry` and
    `--write-component-contours`; workers compute fixed eye geometry and local
    contour packs, and the parent merges packed variable-length contour arrays
    deterministically
  - writes `smart_finalizer_timing_summary` and
    `smart_finalizer_chunk_timings` attrs to expose per-phase and per-chunk
    runtime
  - records process-shard instrumentation attrs including `execution_backend`,
    `process_shard_execution_enabled`, `worker_process_count`, requested and
    effective worker chunk sizes, and the chunk-alignment policy
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

3. Add a batch driver. Sequential debug execution and production
   `process_shards` execution are in place.
   - creates target refined run
   - writes deterministic chunks sequentially in the default debug path
   - assigns contiguous, physical-chunk-aligned row shards to worker processes
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
- no interactive brush/save event through the batch finalizer
- no overwrite of existing curated refined runs unless explicitly requested
