# Segmentation Pipeline Step TODO

<!-- todo-meta
status: active
last_updated: 2026-03-28
-->

## Goal

Introduce one broader segmentation pipeline abstraction that can run the
available segmentation methods behind a single stage family and write canonical
runtime outputs to:

- `subject_mask_runs/<run>`

instead of requiring operators and downstream code to think in terms of several
separate raw segmentation entrypoints.

The intended abstraction is not "one model for everything." It is:

- one orchestration surface
- one canonical runtime output stage
- component-specific method selection within that orchestration surface

## Why This Change Is Needed

Current segmentation authoring is fragmented:

- eye segmentation has separate `traditional`, `yolo`, and `unet` entrypoints
- subject/body segmentation has separate `traditional` and `sam3` entrypoints
- swim bladder segmentation has its own traditional entrypoint
- the core pipeline still exposes some of these as separate stage concepts
- `eye_masks_runs` remains a legacy raw write target while body/swim bladder
  already use `subject_mask_runs`

This creates three practical problems:

1. stage fragmentation
   - users need to know which CLI/module to call for each component and method
2. output fragmentation
   - runtime mask state is split between `eye_masks_runs` and
     `subject_mask_runs`
3. orchestration drift
   - `core/pipeline.py`, `run_eye_masks_batch.py`, and direct CLIs do not all
     express the same set of methods or outputs

## Verified Current Entry Points

### Eye Segmentation

- traditional: `src/fisheye/segmentation/eye_segmentation.py`
- yolo: `src/fisheye/segmentation/eye_segmentation_yolo.py`
- unet: `src/fisheye/segmentation/infer_unet_eye_masks.py`

### Subject / Body Segmentation

- traditional: `src/fisheye/segmentation/subject_segmentation.py`
- sam3: `src/fisheye/utils/run_sam_subject_masks.py`

### Swim Bladder Segmentation

- traditional: `src/fisheye/segmentation/swim_bladder_segmentation.py`

### Existing Partial Orchestration

- eye-only batch orchestration:
  `src/fisheye/utils/run_eye_masks_batch.py`
- some eye-mask pipeline wiring:
  `src/fisheye/core/pipeline.py`
- eye-to-subject projection seam:
  `src/fisheye/utils/backfill_subject_mask_runs.py`

## Design Direction

Recommended direction:

- add one segmentation orchestration surface that selects method per component
- make `subject_mask_runs` the canonical raw runtime output
- allow partial-component runs using `available_channels`
- treat eye-specific stages as compatibility/specialized stages during the
  transition

Not recommended:

- preserving separate long-term raw stage families for eye vs subject masks
- making downstream readers merge several raw segmentation stage families on
  every read
- requiring direct model-specific CLIs to become the long-term public
  abstraction

## Proposed Abstraction

One new logical pipeline step:

- `segmentation`

with component-scoped method selection, for example:

```yaml
segmentation:
  enabled: true
  run_name: null
  label_schema_id: subject_v1_union
  components:
    subject_body:
      method: sam3
    eyes:
      method: yolo
    swim_bladder:
      method: traditional
```

Interpretation:

- the step is one orchestration unit
- each component chooses its own method
- the output is one canonical `subject_mask_runs/<run>` artifact or one
  deterministic derived run per invocation
- missing components are represented through `available_channels`, not by
  switching stage families

## Important Constraint

The codebase is not yet ready for all eye outputs to default to anatomical
left/right subject-mask channels.

Current eye producers are heterogeneous:

- traditional eye segmentation is anatomical LR
- YOLO currently emits `eye_0` / `eye_1`
- U-Net may emit union or LR depending on checkpoint `label_mode`

So the safest immediate canonical runtime bridge is:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`

Longer term, once all eye methods can guarantee LR semantics, the canonical
runtime schema can be revisited.

## Required Work

### 1. Define The Pipeline Contract

We need a documented contract for the new `segmentation` step covering:

- top-level config shape
- supported components
- supported methods per component
- default method resolution
- run naming behavior
- overwrite behavior
- failure behavior for unsupported method/component combinations
- canonical output stage and schema rules

Needed deliverable:

- new contract/design doc for segmentation pipeline config and semantics

### 2. Define Supported Method Matrix

The abstraction should encode what is actually valid today.

Initial realistic method matrix:

- `subject_body`: `traditional`, `sam3`
- `eyes`: `traditional`, `yolo`, `unet`
- `swim_bladder`: `traditional`

The pipeline must fail clearly on unsupported combinations such as:

- `swim_bladder: sam3`
- `eyes: sam3`
- `subject_body: unet`

Needed deliverable:

- central method-capability table in code

### 3. Create A Shared Subject-Mask Writer / Projection Helper

We need one shared path that can take component-scoped outputs and materialize
them as canonical `subject_mask_runs/<run>`.

This probably means:

- extracting reusable logic from
  `src/fisheye/utils/backfill_subject_mask_runs.py`
- supporting direct write of eye-only outputs into `subject_mask_runs`
- preserving provenance for the true source method and source run
- standardizing attrs, stage provenance, and `available_channels`

Needed deliverables:

- reusable writer/helper module
- canonical run semantics for non-legacy runtime projection

### 4. Decide Snapshot Semantics

The new abstraction needs a clear answer to:

- does one invocation produce one coherent `subject_mask_runs/<run>` snapshot
  containing all requested components?
- or does each component still write independently and then merge?

Recommended direction:

- one orchestration invocation should produce one coherent subject-mask
  snapshot

That likely implies:

- gather component outputs first
- normalize them to the target schema
- write one subject-mask run with correct `available_channels`

This is more consistent with the existing subject-mask unification direction.

### 5. Build The New Orchestration Surface

We need a real implementation surface for the abstraction, not just docs.

Likely options:

- new utility runner, for example
  `src/fisheye/utils/run_segmentation_stage.py`
- or a more subject-mask-specific runner, for example
  `src/fisheye/utils/run_subject_mask_segmentation.py`

Responsibilities:

- parse segmentation config
- resolve component methods
- call existing method-specific producers
- normalize outputs into one subject-mask artifact
- emit clear summaries and provenance

### 6. Integrate With `core/pipeline.py`

Once the new orchestration surface exists, the core pipeline should stop
treating eye masks as a separate long-term raw segmentation concept.

Needed changes:

- add a new `_run_segmentation()` step or equivalent
- wire pipeline config to the new segmentation abstraction
- decide migration behavior for existing `eye_masks` pipeline params
- keep compatibility aliases during transition

### 7. Preserve Compatibility During Migration

We should not break existing eye-specific workflows immediately.

Near-term compatibility plan:

- keep direct model-specific eye CLIs working
- keep `eye_masks_runs` and `refined_eye_masks_runs` readable and writable
  during transition
- optionally dual-write or immediately project eye outputs into
  `subject_mask_runs`
- avoid forcing eye refinement/training consumers to migrate in the same change

Needed deliverable:

- explicit migration policy doc or section in the implementation contract

### 8. Standardize Provenance And Registry Surface

A broader segmentation step should produce consistent provenance regardless of
component method.

Needed work:

- standard run-level stage provenance for the orchestration step
- component-level provenance for body, eyes, swim bladder
- consistent source method recording
- consistent source model/checkpoint recording where applicable
- registry/status integration for the new step name and component availability

### 9. Decide What Happens To Legacy Eye-Specific Metadata

`subject_mask_runs` intentionally does not yet replace the full eye-specific
metadata surface.

We need a decision for:

- ellipses
- contours
- eye separation
- left/right assignment metadata
- eye QA/review summaries

Recommended near-term answer:

- keep these in `eye_masks_runs` / `refined_eye_masks_runs`
- do not block the raw subject-mask unification on immediate migration of these
  eye-specific details

### 10. Testing And Validation

This abstraction needs more than unit coverage of one helper.

Minimum validation surface:

- method-matrix validation tests
- projection/writer tests for eye-only, body-only, swim-bladder-only, and mixed
  runs
- provenance tests for orchestrated runs
- pipeline integration tests in `core/pipeline.py`
- compatibility tests ensuring existing eye-specific paths still work during
  migration

## Suggested Implementation Order

### Phase 1: Contract + Writer Extraction

- document the new segmentation step contract
- extract shared subject-mask projection/writer logic from current backfill code
- define canonical run semantics for runtime eye-to-subject writes

### Phase 2: Eye Bridge

- make eye orchestration surfaces write or project into `subject_mask_runs`
- keep existing eye-specific outputs intact
- validate `subject_v1_union` eye-only subject-mask snapshots

### Phase 3: New Segmentation Runner

- add the new broad segmentation runner
- implement component-method dispatch
- implement coherent run writing into `subject_mask_runs`

### Phase 4: Pipeline Integration

- integrate the runner into `core/pipeline.py`
- add compatibility aliases for existing `eye_masks` config
- update docs and operator workflows

### Phase 5: Follow-On Cleanup

- evaluate whether raw `eye_masks_runs` creation should remain normal,
  become optional, or become legacy-only
- revisit canonical runtime schema once all eye methods support trustworthy LR
  semantics
- later evaluate refined-stage unification separately

## Open Design Questions

1. Should the top-level pipeline step be called `segmentation`,
   `subject_masks`, or `subject_segmentation`?
2. Should one invocation always emit exactly one `subject_mask_runs/<run>`,
   even when only one component is requested?
3. Should eye-only writes initially be projection-based or true direct writes
   into the subject-mask writer?
4. Should direct model-specific CLIs dual-write to both legacy and canonical
   stages, or should only orchestrators perform that bridge?
5. Should the first implementation target `subject_v1_union` only, with LR kept
   as an advanced/explicit mode?
6. What registry/status step name should represent the new broad segmentation
   abstraction?

## Acceptance Criteria

This abstraction is ready when:

- operators can invoke one segmentation orchestration surface instead of
  several component-specific raw entrypoints
- the orchestration surface can run the currently supported methods by
  component
- canonical runtime outputs land in `subject_mask_runs`
- eye-specific compatibility workflows still function
- pipeline wiring no longer treats eye raw segmentation as the only first-class
  segmentation concept
- provenance and status reporting are consistent across segmentation methods

## Related Docs

- `docs/segmentation_stage_split_review.md`
- `docs/subject_mask_stage_unification_todo.md`
- `docs/subject_mask_runs_contract.md`
- `docs/refined_subject_masks_runs_contract.md`
- `docs/subject_mask_training_artifact_contract.md`
