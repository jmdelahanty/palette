# Layout-neutral production DAG design

- Status: accepted direction; implementation not started
- Last reviewed: 2026-07-26
- Scope: production inference and refinement workflows using the structured LSF
  kernel

## Decision

Palette's structured LSF workflow implementation is the general production DAG
engine. `fisheye.cluster.clipped_inference` is one complete workflow recipe
built on that engine; it is not the definition or permanent boundary of the
engine.

Future production workflows must separate four concerns:

1. **Scientific recording identity**: one acquisition timeline and its
   canonical analysis Zarr.
2. **Input layout**: one whole-recording video, several recording-bound clips,
   or another explicitly indexed set of video work units.
3. **Workflow scope**: detection only, keypoints only, subject masks only, or a
   complete default analysis.
4. **Scheduler packaging**: ordinary LSF jobs, job arrays, or bounded
   in-allocation bundles.

Changing one of these concerns must not require inventing a different
scientific pipeline or execution engine.

The intended architecture is:

```text
structured production DAG engine
  ├─ recording-layout adapters
  │    ├─ whole-recording video
  │    └─ clipped recording collection
  ├─ reusable scientific stage fragments
  │    ├─ raw detection
  │    ├─ arena geometry and detection gate
  │    ├─ detection quality and refinement
  │    ├─ crop/cache
  │    ├─ keypoints and refinement
  │    ├─ subject masks and refinement
  │    └─ validation, registry publication, and cleanup
  └─ workflow recipes
       ├─ detection only
       ├─ keypoints only
       ├─ subject masks only
       └─ complete default analysis
```

This direction extends the existing `LsfWorkflow`, `LsfWorkflowFragment`, job
array, bundle, runtime-envelope, and submission implementations. It does not
introduce another DAG framework.

## Current state

The shared kernel under `fisheye.cluster.lsf` is already stage- and
layout-agnostic. It models resources, structured dependencies, jobs, task
groups, fragments, workflow composition, topological submission, runtime
status, and retained evidence.

The most complete client is `fisheye.cluster.clipped_inference`. It currently
plans:

```text
detect array
  -> recording-order quality source
  -> collection quality reconciliation
  -> refined-detection bundle
  -> finalized detection collection
  -> ROI-cache array
  -> proxy crop binding
       ├─ keypoint array -> keypoint finalization -> refinement
       └─ subject-mask array
                    \       /
                 mask packages
                      -> refined-mask import
                      -> exact validation
                      -> registry reconciliation
                      -> optional cleanup
```

Whole-recording keypoint and subject-mask workflows also use the shared LSF
models. The remaining limitation is primarily module shape: the complete
clipped planner accepts clip-specific targets, and the first extracted
detection fragment currently combines raw detection, quality, refinement, and
collection publication in one module.

The downstream `fisheye.analysis_workflows` YAML DAG is a separate system. It
selects and orders persisted analysis products and normally executes them
serially inside one LSF allocation. It is appropriate for downstream derived
analysis, but it is not the production inference scheduler and should not be
extended to duplicate the structured LSF kernel.

## Canonical target model

A production workflow should receive a layout-neutral recording target with
one or more video work units. Conceptually:

```text
RecordingTarget
  target_id
  recording_id
  recording_dir
  analysis_zarr
  canonical timeline authority
  expected subject/arena context
  video work units[]

VideoWorkUnit
  work_unit_id
  video path and immutable source identity
  camera serial
  optional arena identity
  local frame domain
  mapping to canonical recording frames
  acquisition-time domain
```

The concrete schema may use more strongly typed nested records, but it must
preserve these distinctions.

For a whole-recording video, `video_work_units` contains one element. For a
clipped recording, it contains every selected clip. A one-video recording may
therefore be treated as a one-member source collection internally. In this
context, *collection* means the complete set of physical source partitions that
cover one scientific recording; it does not imply that the recording must have
multiple video files.

The layout adapter owns proof that every work-unit frame maps correctly to the
canonical recording timeline. Scientific stages consume that declared mapping
rather than branching on “clipped” versus “whole recording.”

## Required invariants

Every target and work-unit implementation must fail closed on:

- missing, duplicate, or overlapping work-unit identities;
- unresolved ordering on the canonical recording timeline;
- incomplete or conflicting local-to-recording frame mappings;
- camera or arena substitution;
- disagreement about native full-frame dimensions;
- disagreement about acquisition-time domains;
- model or source artifacts whose pinned identities do not match the plan;
- output paths that collide with existing immutable runs; and
- downstream rowsets whose `instance_key` identities do not match their
  declared sources.

`instance_key` identifies observations and is carried between stages. It does
not establish temporal order. Canonical recording-frame indices establish
order across work-unit boundaries.

## Scientific stage fragments

Stage fragments should declare typed logical requirements and products in
addition to concrete LSF dependencies. They should not infer an upstream run by
following a mutable pointer during compute execution.

The target decomposition is:

```text
raw_detection
  requires: recording video work units, pinned model
  provides: raw detection collection

arena_geometry
  requires: recording-bound acquisition geometry and/or recording imagery
  provides: reviewed selected arena geometry

detection_arena_gate
  requires: raw detection collection, selected arena geometry
  provides: keyed detection-gate assessment

detection_postprocess
  requires: raw detections, keyed gate assessment when policy requires it
  provides: quality run, refined detections, finalized detection collection

crop_cache
  requires: finalized detection collection
  provides: stable crop lineage and reusable pixel cache

keypoints
  requires: crop lineage/cache, pinned pose model
  provides: raw and refined keypoints

subject_masks
  requires: crop lineage/cache, pinned segmentation model
  provides: raw and refined subject masks

validation/publication
  requires: the outputs selected by the workflow recipe
  provides: validated analysis and serialized registry reconciliation
```

Keypoints and subject-mask inference may run concurrently after the crop/cache
contract is satisfied. Subject-mask finalization may join the exact refined
keypoint run when component assignment requires it.

## Arena geometry and dish gating

Arena geometry is a first-class logical prerequisite, but it need not always be
a separate scheduler allocation.

Acquisition geometry import, independent Palette fitting, and raw detection can
run concurrently when they only read immutable recording inputs. Geometry
candidate publication must not silently select a candidate. A workflow that
requires a reviewed geometry authority must either:

- reuse an exact selected geometry run; or
- stop as blocked until review publishes a selection.

After selection, the detection gate consumes the raw detection collection and
the exact selected geometry. It writes a keyed, auditable assessment and does
not erase raw detections. At minimum the assessment preserves each
`instance_key`, the native-coordinate centroid, signed distance to the selected
gate, acceptance result, rejection reason, coordinate transform identity, and
geometry-run identity.

The logical stage remains distinct even if its small CPU calculation is
physically bundled with recording-order quality-source materialization. This
allows later re-gating against a different reviewed geometry without rerunning
GPU detection.

The current registry `dish_mask` stage describes tuning metadata and must not be
silently reinterpreted as this complete contract. The eventual stage-catalog
update should distinguish selected arena geometry from detection-gate
materialization, using names such as `arena_geometry` and
`detection_arena_gate` after the artifact contracts are finalized.

## Workflow recipes and standalone stages

A complete default workflow is a recipe over reusable fragments, not a special
implementation of those stages. Supported scopes should include at least:

- `detection_only`;
- `detection_through_refinement`;
- `keypoints_only`;
- `subject_masks_only`; and
- `full_analysis`.

Selecting a narrow scope must not require users to submit the complete DAG.
Standalone stage commands remain first-class interfaces for canaries, repair,
review, model comparisons, and incremental recomputation.

Registry-backed planning may satisfy a fragment requirement with a previously
validated immutable artifact. Reuse must validate exact lineage and completion;
it must not guess from the newest directory name. A reused product becomes an
external input to the composed workflow, so downstream fragments do not care
whether it was produced in the current submission or a prior one.

## Scheduler packaging

The planner chooses scheduler representation after scientific work units are
known:

- one selected work unit may use an ordinary job;
- repeated same-resource work normally uses an LSF array;
- lightweight independent CPU work may use a bounded bundle inside one
  allocation;
- stateful recording-wide reconciliation remains a finalizer; and
- shared selectors, completion markers, registry state, and collection
  metadata have one authoritative publisher after all workers validate.

An LSF array is not a scientific stage. A clip is not necessarily an array
element forever. A whole recording is not necessarily one compute task forever.
The planner may alter physical partitioning for performance only when the
result remains scientifically and byte-contract equivalent and workers own
safe non-overlapping output chunks or temporary artifacts.

For an array, `done(<array-job-id>)` is a whole-array success barrier. If a
future workflow needs successful work units to advance independently, that
requires a deliberately tested per-target dependency design; it must not assume
unsupported per-index LSF semantics.

## Migration plan

Generalization should proceed incrementally and retain clipped-production
parity at every checkpoint:

1. Define and test layout-neutral `RecordingTarget` and `VideoWorkUnit`
   contracts, including canonical-frame mapping.
2. Add adapters for the existing clipped target manifest and a one-video
   whole-recording target.
3. Split the current clipped detection module into raw-detection and
   detection-postprocess fragments without changing commands or outputs.
4. Dry-run the existing clipped campaign before and after the split and require
   equivalent jobs, dependencies, resources, commands, expected outputs, and
   fragment products.
5. Compose and dry-run a whole-recording detection workflow through the same
   fragment builders.
6. Add the selected-arena-geometry and keyed detection-gate fragment between
   raw detection and detection postprocessing.
7. Extract crop/cache, keypoint, and subject-mask fragment builders behind the
   same typed capability boundary.
8. Add named workflow recipes and registry-backed reuse of exact validated
   products.
9. Update the stage catalog only after the new artifact families and
   completion contracts are implemented.
10. Retire redundant shell-only composition where the structured planner has
    achieved behavior and evidence parity; retain thin operator wrappers where
    useful.

## Acceptance criteria

The generalized production DAG is ready when:

- the same stage builders plan both one-video and clipped recordings;
- the existing clipped dry-run remains equivalent through migration;
- detection-only and full-analysis recipes share the same detection fragments;
- selected geometry can be reused or produced independently of raw detection;
- changing selected geometry reruns gating and dependent processing without
  rerunning raw detection;
- all cross-stage observation alignment is checked by exact `instance_key`;
- all temporal reconciliation uses canonical recording frames;
- scheduler arrays and bundles remain explicit in immutable plan evidence;
- no compute job submits new jobs dynamically;
- no worker publishes shared completion or registry authority; and
- successful validation is required before pointer or registry promotion.

## Non-goals

This change does not propose:

- one generic YAML language capable of expressing every Palette command;
- moving scientific stage semantics into the LSF kernel;
- eliminating standalone stage commands;
- treating every recording as physically clipped;
- treating scheduler arrays as part of scientific provenance;
- automatically accepting an acquisition or Palette arena fit without its
  declared review policy; or
- merging the downstream analysis-workflow executor into the production
  inference scheduler.

## Related documents

- `docs/lsf_submission_framework_design.md`
- `docs/clipped_inference_dag.md`
- `docs/analysis_workflow_dag.md`
- `docs/dask_zarr_write_safety.md`
- `docs/detect_quality_collection_reconciliation_contract.md`
- `docs/recording_bound_geometry_import_and_validation_design.md`
