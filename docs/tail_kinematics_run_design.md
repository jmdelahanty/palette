# Tail Kinematics Run Design
<!-- contract-meta
version: 2
status: active
last_verified: 2026-07-15
-->

Purpose: define the first Palette-native tail-angle, tail-deflection, and
tail-curvature metric surface derived from ordered subject-shape tail samples,
while keeping future Megabouts/ZebraZoom/Stytra/BEAST adapters compatible but
non-canonical.

## Design Decision

Palette has a dedicated frame-level run family:

```text
analysis/tail_kinematics_runs/<run>
```

This run consumes an exact tail-geometry source, usually:

```text
analysis/subject_shape_runs/<shape_run>/components/subject_body/
```

and produce tail posture metrics that are easier to plot, compare, export, and
summarize than raw geometry arrays.

`analysis/subject_shape_runs` remains the geometry authority:

- snout/head endpoint
- tail base and tail tip
- ordered centerline
- B-spline samples
- normalized tail samples
- tangents, normals, curvature

`analysis/tail_kinematics_runs` owns behavior-facing derived traces:

- tail angles
- lateral deflections
- scalar bend/curvature summaries
- optional temporal derivatives or frequency summaries in later versions

It should not mutate subject-shape runs, refined masks, swim-bout
segmentations, or external classifier outputs.

## Source Requirements

The current implementation requires a valid subject-shape run with:

```text
components/subject_body/bspline_valid
components/subject_body/tail_sample_s
components/subject_body/tail_sample_xy
components/subject_body/tail_tangent_xy
components/subject_body/tail_curvature_px_inv
components/subject_body/tail_sample_valid
components/subject_body/tail_base_xy
body_frame/forward_axis_xy
body_frame/left_axis_xy
body_frame/valid
```

Rows are valid only when the source row has a valid body frame and valid tail
geometry. The v1 behavior resamples from valid B-spline/tail geometry into the
lower-dimensional `tail_angle_sample_*` surface. Source failures propagate into
`failure_reason_bytes` rather than being
silently interpolated.

## Coordinate And Sign Convention

Palette's body frame currently defines:

- `forward_axis_xy`: posterior-to-anterior direction.
- `left_axis_xy`: anatomical left.

Tail samples are ordered from `tail_base_xy` to `tail_tip_xy`, so their natural
tail direction is caudal/posterior. Tail angles should therefore use:

```text
caudal_axis_xy = -forward_axis_xy
```

Recommended signed angle convention:

```text
tail_angle_rad =
  atan2(dot(tail_tangent_xy, left_axis_xy),
        dot(tail_tangent_xy, caudal_axis_xy))
```

This gives:

- `0`: tail tangent points straight caudally.
- positive values: bend toward anatomical left.
- negative values: bend toward anatomical right.

This convention should be recorded in attrs, not inferred from array names.
Degree arrays are useful for plotting, but the mathematical convention should
be defined once in radians.

## Sampling Dimensionality

Tail kinematics should be lower-dimensional than the dense subject-shape
geometry surface.

Subject-shape runs may store:

- dense `bspline_sample_xy` for whole-body geometry and visualization
- compact `bspline_control_points_xy`
- subject-shape `tail_sample_xy` geometry samples, currently schema-v3
  geometry outputs
- dense or moderately dense curvature samples for geometry/QC

Tail-kinematics runs store behavior-facing tail samples separately. The default
is:

```text
tail_angle_sample_count = 10
tail_angle_sample_s = linspace(0.0, 1.0, 10)
```

where `0.0` is the tail base and `1.0` is the tail tip. These samples are the
markers that should drive the default Palette tail-angle/deflection vectors
shown to users and used by Palette-native summaries. External adapters may use
a different sample count, but they must record it explicitly.

Megabouts keypoint input uses a related but different count: 11 ordered
tail-curve points produce 10 Megabouts cumulative angle segments. Palette may
therefore generate a K=11 tail-kinematics candidate for comparison or adapter
symmetry, but the Megabouts adapter can also resample directly from
`subject_shape_runs` without changing Palette's default K=10 behavior-facing
tail-angle surface.

This split avoids using hundreds of dense spline evaluation points as a
behavior feature vector while preserving dense geometry for measurements that
need it.

## Frame-Level Metric Set

The current run exposes one trace group for the selected geometry source:

```text
analysis/tail_kinematics_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_kinematics_runs"
    schema_version                    1
    method                            "tail_metrics_from_subject_shape"
    method_version                    1
    row_axis                          "roi_rows"
    source_subject_shape_run
    source_refined_subject_masks_run
    source_tail_geometry_kind         "subject_shape_bspline_tail_resample"
    body_frame_convention
    tail_angle_reference_axis         "caudal_axis=-forward_axis"
    tail_angle_positive_direction     "anatomical_left"
    tail_angle_units_primary          "rad"
    tail_sample_domain                "tail_segment_normalized_arclength"
    tail_angle_sample_count           10 by default
    source_geometry_tail_sample_count optional
    curvature_source                  "subject_shape.tail_curvature_px_inv"
    created_at_utc

  frame_index                         (N,)
  time_s                              (N,) optional
  valid                               (N,)
  failure_reason_bytes                (N, width)

  tail_angle_sample_s                 (K,)
  tail_angle_sample_xy                (N, K, 2)
  tail_angle_rad                      (N, K)
  tail_angle_deg                      (N, K) optional plotting mirror
  tail_tip_angle_rad                  (N,)
  tail_tip_angle_deg                  (N,) optional plotting mirror

  tail_lateral_deflection_px          (N, K)
  tail_tip_lateral_deflection_px      (N,)
  tail_lateral_deflection_mm          (N, K) optional when calibrated
  tail_tip_lateral_deflection_mm      (N,) optional when calibrated

  max_abs_tail_angle_rad              (N,)
  max_abs_tail_angle_deg              (N,) optional plotting mirror
  tail_angle_rms_rad                  (N,)
  tail_angle_rms_deg                  (N,) optional plotting mirror
  integrated_abs_tail_angle_rad       (N,)

  tail_curvature_px_inv               (N, K)
  max_abs_tail_curvature_px_inv       (N,)
  integrated_abs_tail_curvature       (N,)

  source_refined_subject_masks/       optional copied source-revision snapshot
    row_revision                      (N, C)
    row_revision_available            (C,)
```

`tail_lateral_deflection_px` should be computed from each behavior-facing tail
angle sample relative to the tail base in body-frame coordinates:

```text
tail_lateral_deflection_px =
  dot(tail_angle_sample_xy - tail_base_xy, left_axis_xy)
```

This is a signed spatial deflection, not an angle. It is useful because some
users reason about tail-tip displacement more naturally than tangent angle.

`integrated_abs_tail_angle_rad` should integrate over normalized tail arclength
using `tail_angle_sample_s`; it is a compact "how bent is the tail now?" scalar,
not a frequency or movement classifier.

If the source subject-shape run has
`source_refined_subject_masks/row_revision`, the tail-kinematics writer should
copy that snapshot into its own `source_refined_subject_masks/` group. The tail
run is still downstream of the subject-shape run, but this copied revision table
keeps the refined-mask lineage auditable even if a consumer only has the tail
run selected.

Schema policy:

- `analysis.tail_kinematics_runs` schema v1 should define this low-dimensional
  `tail_angle_sample_*` behavior-facing surface before the first implementation
  ships.
- Existing `analysis.subject_shape_runs` schema v3 does not need to change if
  tail kinematics resamples from valid subject-shape geometry.
- If subject-shape itself changes the meaning or default dimensionality of
  `components/subject_body/tail_sample_xy`, then subject-shape should bump to
  schema v4 and a new subject-shape method version.

## What Not To Add Yet

Do not add these to v1 unless there is an immediate analysis need:

- tail-beat frequency
- phase
- dominant frequency
- temporal derivatives
- tail vigor
- bout-aligned tail arrays

Those metrics depend on temporal windows, smoothing, gap policy, and bout
selection. They are valid and important, but they should be added as explicit
method-versioned extensions rather than smuggled into the first geometry-to-
kinematics pass.

## Bout-Level Relationship

`analysis/tail_kinematics_runs` should be frame-level and independent of bout
segmentation.

Bout summaries should be written downstream, linked to exact sources:

```text
analysis/bout_kinematics_runs/<run>/
  attrs:
    source_tail_kinematics_run
    source_swim_bout_run
    source_swim_bout_speed_level

  tail/per_bout_metrics/
    bout_id
    source_start_frame
    source_end_frame
    source_start_time_s
    source_end_time_s
    max_abs_tail_angle_deg
    tail_tip_angle_peak_to_peak_deg
    tail_tip_lateral_deflection_peak_to_peak_px
    integrated_abs_tail_angle_mean_rad
    max_abs_tail_curvature_px_inv
    valid
    failure_reason_bytes
```

This keeps raw bout segmentation, frame-level tail traces, and per-bout
biological summaries separable and re-runnable.

## Million-Frame Materialization Strategy

The subject-shape run is the immediate materialized input to native tail
kinematics:

```text
refined subject masks
  -> analysis/subject_shape_runs/<shape_run>
  -> analysis/tail_kinematics_runs/<tail_run>
```

The subject-shape run remains the geometry authority. Tail kinematics must not
return to dense masks, crops, or source video when the required subject-shape
geometry is already materialized. This makes tail kinematics a relatively
inexpensive, repeatable interpretation layer: its sample count, angle
convention, and summaries can change without rerunning segmentation, contour
extraction, or B-spline fitting.

For the current implementation, the required subject-shape staging surface is:

```text
components/subject_body/
  tail_sample_s
  tail_sample_xy
  tail_tangent_xy
  tail_curvature_px_inv
  tail_sample_valid
  bspline_valid
  tail_base_xy
  tail_sample_failure_reason_bytes
  bspline_failure_reason_bytes

body_frame/
  forward_axis_xy
  left_axis_xy
  valid
  failure_reason_bytes

row_index/
  frame_indices
  detection_indices
  source_refined_row_ids
  source_detect_row_index
  source_crop_row_ids
  instance_key

source_refined_subject_masks/         optional revision snapshot
```

The row-index and revision arrays preserve provenance; the body and body-frame
arrays drive the calculation. Unrelated subject-shape components, dense
refined masks, eye-angle data, video, and other analysis-run families are not
tail-kinematics staging inputs.

### Preferred cluster topology

The preferred large-recording topology is one LSF job per recording, not one
scheduler job per Zarr chunk:

```text
one recording job
  -> resolve and fingerprint the exact subject-shape source run
  -> stage its required physical shards to node-local scratch
  -> process bounded row blocks with a modest local worker pool
  -> assemble and validate a complete local tail-kinematics run
  -> copy the run to its authoritative Zarr location
  -> publish completion metadata and latest-complete pointers last
  -> clean ephemeral node-local data
```

Node-local staging is an execution cache, not another data authority. The
recording Zarr on shared storage remains authoritative. A failed or preempted
job may discard its local state without changing the selected subject-shape run
or exposing a partially completed tail run as current.

A reasonable first benchmark is 8, 16, and 32 local workers. Worker count must
remain configurable because vectorized tail calculations may become limited by
memory bandwidth before all CPUs are useful. For a cohort, the scheduler may
run several per-recording jobs concurrently while retaining this intra-node
topology within each job.

### Shard-aware staging

For an all-frame calculation that fits in node-local scratch, stage all
physical shards of the required arrays in one explicit, resumable transfer
before starting workers. "All" here means all shards of the narrow input
surface above, not all shards in the recording Zarr. This avoids repeated
network reads and keeps local workers off shared storage during computation.

The staging manifest must record at least:

- exact source Zarr and subject-shape run;
- source schema/method versions and a metadata fingerprint;
- selected array paths;
- requested frame or row interval;
- physical shards selected for each array;
- byte/file totals and verification outcome.

Selective shard staging is reserved for bounded time ranges, multi-node
partitioning, insufficient local capacity, or resuming unfinished output
ranges. Because one physical Zarr shard may contain several logical chunks,
selection and worker ownership must follow the physical shard/chunk layout
rather than arbitrary row boundaries. Touching one logical chunk can require
reading its enclosing physical shard.

A manifest-driven transfer such as a single resumable file-copy operation is
preferred to workers fetching shards independently. The transfer still opens
the physical files internally, but operationally it is one bounded and
auditable staging phase.

### Local parallel computation and safe publication

The writer now resolves lazy subject-shape array handles, copies row lineage
and revision snapshots in bounded slices, and computes/writes one row block at
a time. Requested block rows are rounded up to the physical output row-chunk
grid and both values are recorded in provenance. This single-writer streaming
backend is the first safe milestone: memory is bounded without requiring
parallel writes. The block-local kernel now vectorizes shared-grid
interpolation, body-frame projection, validity classification, and summary
reductions. New runs record `compute_kernel = "vectorized_shared_grid_v1"`.
Sparse failure-label normalization remains rowwise because it operates only on
invalid rows.

On the first 16,384-row Sleepyfish block, the scalar kernel took about 0.998
seconds and the vectorized kernel took about 0.070 seconds, approximately a
14-fold compute speedup. All 15 floating-point output arrays had zero maximum
absolute difference against the persisted scalar reference; validity and
failure bytes were also exact. Network source reads took roughly 1.4--2.4
seconds in those spot measurements, so node-local staging remains the larger
end-to-end optimization target.

Local parallel execution may then assign each worker one or more consecutive
blocks. Every worker must exclusively own complete, non-overlapping physical
output chunks for every array it writes. Disjoint logical row slices are not
sufficient when they share a physical Zarr chunk. Requested and effective
block boundaries, worker count, and physical chunk alignment must be recorded
in provenance. See [dask_zarr_write_safety.md](dask_zarr_write_safety.md) for
the repository-wide write rule.

The local result is not publishable until a single coordinator verifies:

- exact, gap-free row coverage;
- no overlapping worker ownership;
- expected shapes, dtypes, and sample count;
- validity and failure-reason accounting;
- source fingerprint equality;
- required row lineage and revision snapshots;
- successful completion of every output array.

The implemented materializer lives in
`fisheye.analysis_workflows.materializers.tail_kinematics`. Its default mode is
a read-only plan. With `--apply`, it:

1. resolves and inventories the exact source run without creating scratch;
2. copies only the required physical files through one `rsync --files-from`
   operation;
3. validates the staged file inventory and opens the staged logical run;
4. computes the complete result against the node-local subset Zarr, either
   serially or with process tasks that each own one complete output shard;
5. validates schema, shapes, row accounting, block accounting, and the compute
   kernel;
6. copies the completed run into a hidden authoritative sibling, validates the
   copied physical inventory and logical run, and atomically renames that
   sibling to the final run name;
7. reopens and validates the published run before changing any parent pointer,
   then persists that pre-pointer validation result in the run provenance;
8. marks the authoritative run complete, advances the latest and
   latest-complete pointers, performs final run and pointer validation, and
   persists that final validation result in the run provenance.

Existing authoritative run names are never replaced. Scratch is removed only
after successful publication and is retained after failure for diagnosis. The
authoritative run-group rename and parent latest-complete update are serialized
per recording with an advisory file lock, so two materializers cannot race the
same parent metadata. Any failure after the authoritative rename removes the
new target and restores the parent attrs captured before publication, including
the previous latest and latest-complete pointers. The rollback policy and both
validation results are recorded under `cluster_output_staging`. The
corresponding LSF entrypoint is
`scripts/submit_tail_kinematics_materialization_bsub.sh`; it pins a clean Palette
commit, refuses compute outside an LSF allocation, selects `/scratch` or a
node-local `$TMPDIR`, and records an atomic status file on shared storage.

New tail-kinematics outputs separate three storage/compute scales: 256-row
logical chunks, 16,384-row bounded compute blocks, and 262,144-row physical
output shards. A process task exclusively owns one complete physical shard and
computes/writes it serially in bounded sub-blocks. The driver remains the only
metadata/finalization writer. Requested and effective values for both compute
blocking and output sharding, plus block/shard/task counts, are recorded in run
attrs and stage provenance. Copied lineage arrays preserve their source logical
chunks, so their physical shard span is the requested output span or the
minimum larger span required to contain one source chunk.

The 262,144-row default is measured rather than arbitrary. On the 1,169,010-row
Sleepyfish tail run it reduced total files from 1,686 to 145 and reduced a
checksum-validated node-local-to-PRFS publication from 13.70 seconds to 2.41
seconds. Random-row and 1,024-row window reads were unchanged; the bounded full
scan increased from 3.95 to 6.26 seconds. See
`docs/archive/tail_kinematics_sharding_benchmark.md`.

Current subject-shape sources are still physically chunked, so the first
staging implementation copies all required source chunk files. It will
naturally copy fewer physical files when upstream subject-shape arrays are
sharded without changing the logical staging contract.

The node-local serial and process-shard topologies plus atomic publication are
implemented and unit tested. The million-frame staged canary validated source
transfer, bounded computation, local/final inventories, and completion-last
publication, so the general workflow DAG now enables `tail_kinematics`. Direct
use of the dedicated materializer remains available for focused operations.

The sharded-subject-shape downstream benchmark also measured the worker-count
tradeoff on the 1,169,010-row Sleepyfish source. Eight requested LSF slots
yielded five effective shard workers and completed local materialization in
32.04 seconds; two requested/effective workers completed it in 56.65 seconds.
The larger allocation saved about 25 seconds while reserving four times as many
slots. Two workers are therefore the routine operational recommendation for
this short, I/O-heavy stage; the larger allocation is an opt-in latency choice,
not a required correctness or throughput setting.

## Megabouts Compatibility

Megabouts should be treated as an adapter and classifier consumer, not as the
canonical Palette schema.

The key boundary is:

```text
Palette owns reusable tail primitives.
Megabouts consumes a mapped view of those primitives.
Megabouts outputs return as imported classifier results.
```

Palette should therefore compute the general signals that Megabouts-like tools
need, such as ordered tail points, body-frame tail angles, tail-tip deflection,
curvature, and validity masks. It should not rename its canonical arrays or
change its sign/unit conventions just to match a specific external package.
External-tool conventions belong in explicit adapter attrs and export manifests.

Palette-native inputs for Megabouts can be generated from:

- `analysis/track_kinematics_runs` for `head_x`, `head_y`, `head_yaw`,
  position, and trajectory.
- `analysis/subject_shape_runs` for `tail_x`, `tail_y`.
- `analysis/tail_kinematics_runs` for Palette-native tangent-angle review
  and comparison.
- `analysis/swim_bout_runs` when using Palette-selected bout windows.

Megabouts-compatible `tail_angle` should not be assumed to be Palette
`tail_angle_rad`. Palette `tail_angle_rad` is a local body-frame tangent angle
sampled along the tail. Megabouts `tail_angle` is a cumulative segment-angle
trace derived from ordered keypoints. The first Megabouts adapter should
therefore derive its `tail_angle` from `K=11` subject-shape tail keypoints via
Megabouts' own keypoint conversion, then compare it against Palette
`tail_angle_rad` only as an audit.

The implemented first step is a derived view, not duplicated permanent arrays
inside the native tail-kinematics run:

```text
Palette sources
  -> analysis/tail_posture_view_runs/<run>
  -> Megabouts runtime
  -> imported/classifier output run
```

`analysis/tail_posture_view_runs` is a compatibility artifact. It is
regenerated from Palette sources when needed and does not redefine Palette's
native `analysis/tail_kinematics_runs` schema.

Current v1 structure:

```text
analysis/tail_posture_view_runs/<run>/
  attrs:
    schema_id                         "analysis.tail_posture_view_runs"
    schema_version                    1
    method                            "tail_posture_view_from_subject_shape"
    method_version                    1
    row_axis                          "roi_rows"
    view_family                       "megabouts_compatible"
    compatible_tool                   "megabouts"
    dependency_policy                 "no_megabouts_dependency_required"
    source_subject_shape_run
    source_subject_shape_path
    source_refined_subject_masks_run
    source_tail_kinematics_run        optional comparison source
    source_tail_geometry_kind         "subject_shape_tail_curve_resample"
    head_source                       "head_endpoint_xy" | "snout_tip_xy"
    keypoint_count                    11
    angle_count                       10
    angle_convention                  "megabouts_cumulative_segment_angle"
    keypoint_order                    "tail_base_to_tail_tip"
    frame_index_source
    row_lineage_copied
    row_lineage_missing
    algorithm_provenance

  frame_index
  row_index/
    frame_indices                     copied when available
    detection_indices                 copied when available
    source_refined_row_ids            copied when available
    source_detect_row_index           copied when available
  valid
  failure_reason_bytes
  head_xy                             (N, 2)
  head_yaw_rad                        (N,)
  tail_keypoints_xy                   (N, 11, 2)
  tail_angle_rad                      (N, 10)
  tail_angle_deg                      (N, 10)
```

The first canary run was:

```text
tail_posture_view_megabouts_compatible_canary_20260501
source_subject_shape_run: subject_shape_v3_snout_medialjoin_canary_20260429
source_tail_kinematics_run: tail_kinematics_k10_canary_20260430
rows: 19,235
valid rows: 17,495
invalid rows: 1,740
duration: about 4.2 s
```

This run stores a Megabouts-compatible geometric view but does not run
Megabouts preprocessing, segmentation, or classification.

Megabouts outputs should land in classifier/import runs:

```text
analysis/bout_classification_runs/<run>/
  attrs:
    schema_id                         "analysis.bout_classification_runs"
    schema_version                    1
    classifier_family                 "megabouts"
    classifier_version
    source_tail_kinematics_run
    source_track_kinematics_run
    source_swim_bout_run              optional
    megabouts_config_json
    megabouts_export_hash

  per_bout/
    source_bout_id                    optional if Palette bouts used
    start_frame
    end_frame
    start_time_s
    end_time_s
    class_id
    class_label_bytes
    confidence                        optional
    valid
    failure_reason_bytes

  features/                           optional compact model-specific features
```

If Megabouts segments its own bouts, those boundaries belong in the
classification/import run. They should not overwrite `analysis/swim_bout_runs`.

This keeps Megabouts useful without making Palette dependent on Megabouts'
internal schema, model versions, dependency stack, or classifier taxonomy.

## Implementation Checklist

- [x] Implement `analysis/tail_kinematics_runs` writer that resamples valid
  subject-shape tail geometry into default `K=10` behavior-facing tail-angle
  samples.
- [x] Add unit tests for sign convention, straight-tail zero angle, left/right
  sign, and invalid-row propagation.
- [x] Replace whole-array source, lineage, and revision loading with a bounded
  single-writer block calculation and validate equivalent output in unit tests.
- [x] Vectorize block-local interpolation, projection, angle, and summary
  operations. Real-data comparison against the persisted scalar Sleepyfish
  reference was exact for floating outputs, validity, and failure bytes.
- [x] Add a manifest-driven node-local staging adapter that copies all physical
  shards of only the required subject-shape arrays for an all-frame run.
- [x] Add configurable intra-node workers with exclusive, complete output-shard
  ownership, bounded compute sub-blocks, and independent requested/effective
  compute and physical-shard provenance.
- [x] Add shared-storage copy validation and completion-last publication. The
  publisher refuses overwrite, validates a hidden sibling, atomically renames
  the run group, and advances completion metadata last.
- [x] Harden post-rename publication: validate the authoritative target before
  changing pointers, persist pre-pointer and final validation results, and
  remove the new target plus restore the exact prior parent attrs on any
  post-rename failure.
- [x] Add a fail-closed Citrus LSF wrapper that pins a clean shared checkout,
  allocates node-local scratch, and records submission, status, and report
  paths on shared storage.
- [x] Run the node-local materializer on the million-frame Sleepyfish source and
  inspect its staging, local-compute, publish, and final-validation reports
  before enabling `tail_kinematics` in the general workflow DAG.
- [x] The staged Sleepyfish canary `tail_kinematics_sleepyfish_node_local_canary_20260715_01`
  processed 1,169,010 rows on node-local scratch, validated 1,097,961 valid and
  71,049 invalid rows, and atomically published a 285 MB run. Source staging,
  local output, temporary publication, and final authoritative validation all
  matched their inventories.
- [x] The hardened two-worker downstream canary
  `tail_kinematics_hardened_w2_canary_20260715_01` completed all five physical
  shard tasks, persisted valid pre-pointer and final validation reports, and
  advanced both parent pointers only after validation. LSF job `153102493`
  completed in 74 seconds with about 1 GB peak memory and no swap.
- [x] Run the writer on the feeding canary subject-shape run:
  `tail_kinematics_k10_canary_20260430` from
  `subject_shape_v3_snout_medialjoin_canary_20260429` wrote 17,495 valid rows
  and 1,740 invalid rows from 19,235 ROI rows.
- [x] Preserve the first whole-array Sleepyfish reference run for streaming
  parity: `tail_kinematics_sleepyfish_core_canary_20260715_01` wrote 1,097,961
  valid and 71,049 invalid rows from 1,169,010 rows in about 500 seconds. A
  bounded dry run over the same source resolves 72 blocks at 16,384 effective
  rows with 256-row output chunks.
- [ ] Persist PNG summaries for tail angles, tail-tip deflection, curvature,
  and validity/failure reasons.
- [ ] Add tail traces to the Marimo kinematics explorer after the Zarr schema
  stabilizes.
- [ ] Add per-bout tail summaries under `analysis/bout_kinematics_runs`.
- [x] Prototype a Megabouts-compatible posture view from the canary:
  `tail_posture_view_megabouts_compatible_canary_20260501` wrote 17,495 valid
  rows and 1,740 invalid rows from 19,235 ROI rows.
- [x] Add a Palette-owned optional Megabouts classifier adapter CLI. It records
  classifier outputs into Palette-native `analysis/bout_classification_runs`
  while keeping Megabouts itself an optional dependency.
- [x] Define `analysis/bout_classification_runs` for the first Megabouts
  classifier output. See
  [bout_classification_runs_contract.md](bout_classification_runs_contract.md).

## Open Questions

- Resolved for v1: store radians canonically and also write degree mirrors for
  plotting/review convenience. Radians remain the primary units.
- Resolved for first Megabouts-compatible view: do not pass Palette native
  tangent-angle samples directly as Megabouts `tail_angle`. Instead,
  `analysis/tail_posture_view_runs` resamples subject-shape tail geometry to
  `K=11` ordered tail keypoints and writes the `K=10` cumulative segment-angle
  representation expected by Megabouts-like tooling.
- Resolved for v1: mirror curvature into `tail_kinematics_runs` at the same
  low-dimensional `tail_angle_sample_s` positions and record the source in attrs.
- Resolved for v1: persist a sibling `analysis/tail_posture_view_runs` family
  rather than nesting external-tool arrays under `tail_kinematics_runs` or
  requiring external export files.
