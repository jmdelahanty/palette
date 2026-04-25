# Subject Mask Stage Unification TODO

For the active operator-facing source-of-truth contract, see
[current_pipeline_contract.md](current_pipeline_contract.md). This TODO tracks
the remaining migration from eye-specific mask stages toward that contract.

## Goal

Move future mask authoring and refinement onto one canonical stage family:

- `subject_mask_runs/<run>` for raw/component segmentation snapshots
- `refined_subject_masks_runs/<run>` for edited/refined working artifacts

while treating:

- `eye_masks_runs/<run>`
- `refined_eye_masks_runs/<run>`

as legacy compatibility stages for historical archives rather than the normal
write target for new work.

## Why This Change Is Needed

The current migration/backfill path works for provenance, but it creates two
practical problems for new work:

1. duplication
   - the same eye content may exist in both eye-specific stages and
     `subject_mask_runs`
2. awkward artifact identity
   - names like `subject_masks_from_refined_eye_masks_<run>` are acceptable for
     migration utilities, but not as the normal steady-state artifact shape

The future system should not require users or tools to think in terms of:

- "copy eye masks into the new structure"
- "which stage is the real canonical one for this component"

Instead, the canonical answer for new data should simply be:

- the current subject-mask run snapshot
- and optionally its refined subject-mask snapshot

## Design Decision

Recommended direction:

- new raw producers should target `subject_mask_runs`
- sparse multi-source assembly/finalization should target
  `refined_subject_masks_runs`
- new refinement/review should target `refined_subject_masks_runs`
- raw runs should be treated as immutable provenance snapshots
- refined runs should be treated as idempotently editable working artifacts
- run names should describe the actual subject-mask artifact, not encode the
  migration source in the run name
- per-component provenance should live in attrs/subgroups, not in the stage
  family or run name

Not recommended:

- continuing long-term authoring into `eye_masks_runs` for new recordings
- creating separate `body_runs`, `eye_mask_runs`, or `swim_bladder_runs`
  registries under `subject_mask_runs`
- treating the combined subject-mask state as something reconstructed on every
  read from multiple component registries

## Artifact Mutability Policy

The raw and refined stage families should not have identical write semantics.

### Raw runs

`subject_mask_runs/<run>` should remain snapshot-like:

- one run represents one coherent raw segmentation snapshot for one row-aligned
  ROI set
- new inference/materialization writes a new run
- `latest` points to the newest raw snapshot

### Refined runs

`refined_subject_masks_runs/<run>` should behave like a mutable working
artifact:

- an existing refined run may be cleaned up in place through tools like Crimson
  or Paintera
- save operations should be idempotent
- a no-op edit should not rewrite unrelated data or create a new run name
- `latest` may continue to point at the same refined run while that run is
  incrementally improved

Optional later behavior:

- explicit snapshot/freeze or approval actions may create a new refined run when
  stable version history is desired

This keeps raw provenance stable while matching the actual interactive edit
model of refined masks.

## Why Not Separate Per-Component Run Registries

A structure like:

```text
subject_mask_runs/
  body_runs/
  eye_mask_runs/
  swim_bladder_runs/
```

would recreate many of the problems we are trying to eliminate:

- multiple `latest` values to resolve
- row alignment has to be revalidated between component registries
- downstream readers must merge on every read
- review/approval state becomes fragmented by registry
- training/export code loses a simple canonical source artifact

The better separation is:

- one canonical run identity
- component-specific provenance and review metadata within that run

## Recommended Canonical Runtime Shape

Recommended direction for `subject_mask_runs/<run>`:

Native raw model-output runs should store probability surfaces plus
model/config/provenance. Thresholded masks are a refinement/finalization output,
not the canonical raw payload.

```text
subject_mask_runs/
  <run>/
    frame_indices
    frame_counts
    detection_indices
    detection_source
    mask_probs_roi
    available_channels
    metrics/
      prob_max
      probability_present
      area_px
      centroid_xy
      centroid_valid
      bbox_xyxy
      bbox_valid
    components/
      subject_body/
        provenance/
        metrics/...
      eye_left/
        provenance/
        metrics/...
      eye_right/
        provenance/
        metrics/...
      swim_bladder/
        provenance/
        metrics/...
```

Key point:

- `mask_probs_roi` is the canonical unified raw tensor surface
- optional `masks_roi` arrays may exist in legacy/projection runs as
  compatibility caches, but native raw model writers should not require or
  create thresholded masks as raw authority
- `refined_subject_masks_runs/<run>/masks_roi` is the canonical thresholded,
  reviewable/refined mask surface
- `components/<name>/...` holds component-scoped metadata, provenance, and any
  component-specific details
- sparse multi-source workflows should assemble directly into
  `refined_subject_masks_runs/<run>` rather than requiring an assembled raw
  `subject_mask_runs/<run>` intermediate
- the assembled unified run is only valid after subject-mask
  refinement/finalization has materialized canonical QA, reasons, and review
  metadata

Current implementation note:

- the shipped assembly/finalization path accepts source inputs readable through
  `subject_mask_runs`
- it also accepts direct `refined_eye_masks_runs` sources for canonical
  `eye_left` / `eye_right` seeding
- for legacy raw eye-stage inputs, the implemented bridge remains:
  `eye_masks_runs` or `refined_eye_masks_runs`
  -> projected/backfilled `subject_mask_runs/<run>`
  -> assembled/finalized `refined_subject_masks_runs/<run>`

## Recommended Canonical Refined Shape

Recommended direction for `refined_subject_masks_runs/<run>`:

```text
refined_subject_masks_runs/
  <run>/
    masks_roi
    available_channels
    edit_applied
    metrics/
      mask_present
      area_px
    components/
      subject_body/
        reason_bytes
        reason
        review/
        geometry/
      eye_left/
        reason_bytes
        reason
        review/
        geometry/
          ellipse_params
          ellipse_success
          eye_separation_peer
        contours/
      eye_right/
        reason_bytes
        reason
        review/
        geometry/
          ellipse_params
          ellipse_success
          eye_separation_peer
        contours/
      swim_bladder/
        reason_bytes
        reason
        review/
        geometry/
```

The eye geometry that currently lives in `refined_eye_masks_runs` should
ultimately move here as component-scoped refined metadata rather than remaining
in a separate canonical stage for new runs.

Assembly note:

- seeding this run from multiple sources is only the first half of the write
  path
- the subject-mask refinement/finalization step must always run after assembly
  so `refined_subject_masks_runs` keeps the same "materialized refined stage"
  semantics as other Palette refined artifacts
- current implementation detail: those assembly sources are still
  `subject_mask_runs`-backed today, even when the eye content originally came
  from legacy eye stages

Editing note:

- the refined run is the canonical in-place edit target
- tools such as Crimson or Paintera should not write bare pixel blocks only;
  they should eventually route through a Palette-native save helper that keeps
  sibling metadata synchronized

Implementation note:

- the direct assembler/finalizer now exists at
  [src/fisheye/refinement/assemble_refined_subject_masks.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/refinement/assemble_refined_subject_masks.py)
- the first four-component canary assembled refined run was created on
  2026-04-01 as
  `refined_subject_masks_runs/refined_subject_masks_canary_body_eyes_swim_001`

Deferred geometry note:

- parity with the current eye-mask workflow is a long-term goal for
  `subject_body` and `swim_bladder`
- that means component-scoped derived geometry, metrics, and review-time
  summaries should eventually exist for those components too
- this does **not** mean every component should adopt eye-style ellipse outputs
- body, swim bladder, and eyes should each get geometry that matches the
  anatomy and downstream use case
- the geometry/metrics parity goal is intentional, but the concrete arrays for
  body and swim bladder remain deferred until the first real refined subject
  masks are in routine use

## Canonical Label Schema Direction

Recommended runtime/refined canonical schema for new work:

- `label_schema_id = "subject_v1_lr"`
- `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

Why:

- eye refinement and geometry are inherently left/right-specific
- a union eye channel is fine for some training exports, but it is too lossy as
  the canonical runtime/refined representation
- left/right can always be collapsed later for export
- union cannot be losslessly expanded back into anatomical eyes

`subject_v1_union` should remain a valid compatibility or model-output schema,
not necessarily the preferred canonical authoring target.

## Component Update Policy

For raw runs:

- creating or recomputing raw subject masks still writes a new run

For refined runs, when only one component changes:

- update the existing refined run in place by default
- compare edited pixels against the stored component mask and no-op if they are
  identical
- update only the changed component's masks, metrics, and review/provenance
  payloads
- leave unrelated components untouched
- recompute any run-level or component-level derived summary that depends on the
  edited component state

Optional later:

- support an explicit "snapshot refined run" or "freeze refined run" action that
  clones the working refined run into a new run name

This supports workflows like:

- update only eyes
- update only swim bladder
- update body and swim bladder together

without requiring separate component-specific stage families.

## Review and Selection Policy

`latest` alone is not enough to represent readiness.

Recommended distinction:

- `latest`
  - newest raw snapshot for `subject_mask_runs`
  - current working refined artifact for `refined_subject_masks_runs`
- component review status
  - readiness for each component independently
- derived run-level review status
  - e.g. `pending`, `mixed`, `approved`

Possible later convenience attrs:

- `latest_approved`
- `latest_fully_approved`
- component-specific latest-approved pointers

But those should be selectors layered on top of one canonical run family, not
separate registries.

## Legacy Policy

For historical archives:

- keep `eye_masks_runs` and `refined_eye_masks_runs` readable
- allow explicit projection/backfill into `subject_mask_runs` when needed
- do not require destructive rewrites of historical data

For new archives:

- stop introducing new canonical dependencies on eye-specific stages
- raw eye orchestration may still emit `eye_masks_runs/<run>` for compatibility,
  but it should also materialize a same-session `subject_mask_runs/<run>`
  companion immediately so unified registry/query/operator surfaces can treat
  `subject_mask_runs` as the canonical raw subject snapshot
- if eye-specific compatibility outputs are still needed for some downstream
  consumer, treat them as derived/adapter artifacts rather than the source of
  truth
- prefer unified subject-mask component registry/query/operator surfaces for
  eye visibility, and keep legacy eye-stage visibility as diagnostic context

## Naming Policy

Preferred future naming:

- `subject_masks_<method>_<timestamp>`
- `refined_subject_masks_<purpose>_<timestamp>`

Avoid making the canonical run name encode migration ancestry like:

- `subject_masks_from_refined_eye_masks_<run>`

That pattern is fine for migration utilities, but not as the normal artifact
identity.

## Required Provenance Evolution

A single run-level `source_*_run` attr is not enough once components may come
from different sources.

We likely need component-scoped provenance such as:

```json
{
  "components": {
    "subject_body": {
      "source_stage": "subject_mask_runs",
      "source_run": "sam_subject_masks_...",
      "source_channel": "subject_body",
      "method": "sam_body_mask_inference"
    },
    "eye_left": {
      "source_stage": "refined_subject_masks_runs",
      "source_run": "refined_subject_masks_...",
      "source_channel": "eye_left",
      "method": "ellipse_refined_eye_masks"
    },
    "eye_right": {
      "source_stage": "refined_subject_masks_runs",
      "source_run": "refined_subject_masks_...",
      "source_channel": "eye_right",
      "method": "ellipse_refined_eye_masks"
    },
    "swim_bladder": {
      "source_stage": null,
      "source_run": null,
      "source_channel": "swim_bladder",
      "method": null
    }
  }
}
```

Exact naming can change, but the structure should be component-scoped.

## Immediate TODO

### 1. Update the contracts/design docs

- [x] Update `subject_mask_runs_contract.md` to describe component subgroups as
      the preferred long-term metadata home.
- [x] Update `refined_subject_masks_runs_contract.md` to describe eye geometry
      and review metadata under `components/eye_left|eye_right`.
- [x] Update docs that still describe `refined_eye_masks_runs` as the canonical
      future refined eye stage for new runs.

### 2. Define component-scoped provenance attrs

- [x] Define the canonical component provenance payload for raw runs.
- [x] Define the canonical component provenance payload for refined runs.
- [x] Specify that component-local refined edits update the existing refined run
      in place by default, leaving unchanged components untouched unless they
      participate in dependent summaries.

### 3. Define the eye migration target

- [x] Decide which eye-specific arrays move into
      `refined_subject_masks_runs/components/eye_left|eye_right`.
      See [eye_subject_mask_unification_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/eye_subject_mask_unification_design.md).
- [x] Decide whether contour storage remains per-eye component-local or also has
      a shared run-level index.
      Decision: keep contours per-eye component-local under
      `components/eye_left|eye_right/contours/`.
- [x] Decide how `eye_separation` is stored once it conceptually spans two eye
      components.
      Decision: store it under `relations/eye_pair/metrics/`, not duplicated
      into both eye components.

### 4. Define component-only edit semantics

- [x] Specify that component-only refined edits update the current working run
      in place by default rather than cloning a new refined snapshot.
- [x] Specify that unrelated component arrays/attrs remain unchanged unless they
      depend on the edited component state.
- [x] Specify that run-level and component-level summaries are recomputed only
      when they depend on the edited component state.

### 5. Keep legacy compatibility explicit

- [x] Keep backfill/projection utilities for historical eye-mask runs.
- [x] Do not require historical archives to be rewritten.
- [x] Make it explicit that legacy eye-mask stages are compatibility inputs, not
      the preferred new authoring path.
- [x] Make it explicit that legacy refined-eye manual review is a compatibility
      fallback, while canonical manual eye review now lands in
      `refined_subject_masks_runs`.

### 6. Land the direct assembled-refined canary path

- [x] Add a direct multi-source assembler that seeds
      `refined_subject_masks_runs/<run>` from component source runs and runs
      subject-mask finalization in the same command.
- [x] Validate a first four-component canary assembled refined run:
  - `subject_body` from `subject_masks_canary_sam_points_body_eyes_001`
  - `eye_left` / `eye_right` from
    `subject_masks_from_refined_eye_masks_2026-02-12_19-51-24`
  - `swim_bladder` from `traditional_swim_bladder_masks_canary_001`
  - output:
    `refined_subject_masks_canary_body_eyes_swim_001`

## Open Questions

1. Should `subject_v1_lr` become the explicit default schema for new raw and
   refined runtime authoring, while `subject_v1_union` remains mostly an export
   and compatibility schema?
2. Should a future all-in-one model be required to write all canonical channels
   into one `subject_mask_runs/<run>`, even if some channels are weak or absent,
   or should sparse `available_channels` remain normal for model-native runs?
3. Do we want a formal `latest_fully_approved` selector at the parent group
   level once component-level review becomes routine?
4. Should refined runs support an explicit freeze/snapshot operation in
   addition to in-place editing, and if so what should trigger it?
5. At what point should compatibility `refined_eye_masks_runs` materialization
   become opt-in rather than routine once `refined_subject_masks_runs` is the
   canonical eye-capable refined artifact?

## Recommended Near-Term Policy

Current near-term policy:

- use `subject_mask_runs` as the preferred canonical destination for new
  raw component-mask work
- use `refined_subject_masks_runs` as the preferred canonical refined/editable
  destination for body/swim-bladder work and sparse multi-source assembly
- use unified subject-mask component views as the preferred operator/query
  surface for eye/body/swim availability during transition
- treat eye-mask backfills as migration aids, not the final desired steady
  state
- do not require a merged raw canary run before creating a canonical refined
  multi-component working artifact
- avoid building new tooling that depends on separate per-component run
  registries
