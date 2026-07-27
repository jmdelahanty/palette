# Layout-neutral production DAG design

- Status: accepted direction; layout/parity, whole-video raw detection,
  selected geometry, keyed gating, and clipped capability-boundary checkpoints
  implemented
- Last reviewed: 2026-07-27
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
models. Raw detection is now a separately composable module for both supported
recording layouts. Collection quality, refinement, and finalized-collection
publication remain clipped-only because they still consume the clipped
collection source and recording-frame-index contracts.

The clipped recipe no longer presents all downstream processing as one
`analysis:<target>` fragment. Its unchanged jobs are grouped behind explicit
`crop_roi_cache`, `keypoints`, `subject_mask_inference`,
`subject_mask_refinement`, and `analysis_validation` capability artifacts.
This exposes the existing fork/join behavior without duplicating a command:
keypoint and subject-mask inference both depend on crop/cache, while mask
refinement joins raw masks with the exact refined-keypoint artifact.

The downstream `fisheye.analysis_workflows` YAML DAG is a separate system. It
selects and orders persisted analysis products and normally executes them
serially inside one LSF allocation. It is appropriate for downstream derived
analysis, but it is not the production inference scheduler and should not be
extended to duplicate the structured LSF kernel.

### First implementation checkpoint

The first migration checkpoint was implemented on 2026-07-27:

- `fisheye.cluster.recording_layout` defines layout-neutral
  `RecordingTarget`, `VideoWorkUnit`, and frame-mapping contracts;
- adapters construct either a clipped collection with one shared canonical
  recording-frame index or a one-member whole-video identity collection;
- `fisheye.cluster.clipped_inference` now adapts its existing clip plan through
  that neutral contract before constructing detection work;
- the detection module exposes separate `raw_detection:<target>` and
  `detection_postprocess:<target>` fragments with a typed
  `raw_detection_work_units:<target>` artifact between them; and
- a frozen pre-split contract test proves that executable job commands,
  resources, dependencies, task envelopes, expected outputs, and typed final
  outputs did not change during the split.

### Whole-video raw-detection checkpoint

The second checkpoint binds a one-member whole-video target to
`fisheye.utils.run_detection_local_publish` through the same typed
`raw_detection_work_units:<target>` artifact exposed by clipped detection.
The publisher:

- streams the canonical acquisition video from its recording-bound PRFS path;
- builds the complete `detect_runs/<run>` candidate on node-local storage;
- validates completion, immutable sharding, model identity, provenance, and
  coordinate authority before publication;
- atomically publishes and activates the completed root detection run; and
- writes a retained result report through the shared LSF runtime envelope.

Whole-video planning fails closed unless there is exactly one identity-mapped
work unit, the output is exactly `detect_runs/<run>`, an explicit registry is
bound, and no unvalidated reuse mode is requested. Clipped raw detection still
uses its artifact build/import publisher. A frozen executable-plan hash proves
that extracting the shared raw module did not alter the existing clipped
commands, resources, dependencies, or outputs.

This checkpoint makes one whole recording independently plannable through the
general DAG kernel. `fisheye.cluster.whole_video_detection` now supplies the
cohort adapter: it discovers exact active analysis datasets and authoritative
full-frame acquisition streams from the registry, requires one content-pinned
detection model across the selected cohort, and aggregates one atomic publisher
per recording into one bounded LSF array. Whole-video quality and refinement
remain a later source-adapter checkpoint.

The cohort planner fails closed on duplicate active analysis datasets, missing
or ambiguous authoritative full-frame streams, stale video/Zarr paths, model
identity disagreement, and pre-existing output run paths. Dry runs persist the
same immutable `plan.json` and `lsf_plan.json` consumed by array elements;
`--apply` submits only the rendered `bsub` command through
`login1-citrus-poller`.

Example registry-backed dry run:

```bash
scripts/py -m fisheye.cluster.whole_video_detection \
  --run-label batman_detection \
  --run-root /groups/johnson/johnsonlab/jeremy/staging/batman_detection \
  --registry /groups/johnson/johnsonlab/jeremy/registries/palette_registry.sqlite \
  --path-contains Batman \
  --detection-set-id <set-id> \
  --detection-run-id <run-id> \
  --max-concurrent 8 \
  --dry-run --json
```

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

### Implemented selection and gate surface

The first operational implementation is intentionally two separate immutable
publications:

```text
analysis/arena_geometry_runs/<candidate>
  -> analysis/arena_geometry_selection/<selection>
  -> analysis/detection_gate_runs/<gate>
```

`publish_arena_geometry_selection` requires one exact complete candidate,
copies its candidate digest, arena binding, native coordinate authority,
physical rim, and valid gate into an immutable review record, and records the
reviewer, decision source, and reason. It never mutates the candidate and does
not yet write the legacy `analysis_metadata.attrs["dish_mask"]` projection.
The selection parent advances `latest` and `latest_complete` through the shared
owner/generation/lease activation contract only after atomic publication and
fresh source revalidation.

`materialize_registered_detection_gate` accepts either a canonical
whole-video `detect_runs/<run>` or a recording-ordered clipped
`detect_collection_sources/<run>`. Both adapters expose normalized
center-X/center-Y/width/height boxes, native dimensions, canonical frame rows,
and modern `instance_key`. The materializer streams one physical output shard
at a time and writes only:

- `instance_key` and dense `source_row_index`;
- canonical `frame_indices`;
- `detection_centroid_native_px`;
- `signed_distance_to_gate_px`;
- `inside_registered_dish_mask`;
- a versioned `gate_decision` enum; and
- canonical null-padded UTF-8 `reason_bytes`.

The table uses 16,384-row inner chunks and 131,072-row indexed shards by
default. Constant geometry and selection provenance live once at run level.
Validation requires unique keys, dense exact source-row identity, consistent
decision/reason fields, decoded output digests, exact current source-array
digests, and an unchanged selection record before selector activation. Raw
detections remain untouched.

The corresponding `fisheye.cluster.arena_geometry` builders are layout-neutral
fragments. They can be composed with raw detection, or satisfied from exact
previously published candidate/selection artifacts. Integration into the
default clipped and whole-video postprocessing recipes remains an explicit
next checkpoint: when gating policy is required, detection refinement must
consume the exact gate artifact and require identical ordered
`instance_key` values rather than following a mutable pointer.

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
5. Compose and dry-run a whole-recording raw-detection workflow through the
   same fragment builders. The atomic publisher, registry-discovered cohort
   CLI, and consolidated bounded LSF array are implemented.
6. Add the selected-arena-geometry and keyed detection-gate fragment between
   raw detection and detection postprocessing. The immutable publications and
   layout-neutral fragment builders are implemented; default-recipe insertion
   is pending.
7. Extract crop/cache, keypoint, and subject-mask fragment builders behind the
   same typed capability boundary. The clipped recipe's existing jobs now
   expose these separate logical boundaries; consolidating the remaining
   whole/clipped command renderers is pending.
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
