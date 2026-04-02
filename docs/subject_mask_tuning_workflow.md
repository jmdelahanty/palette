# Subject Mask Tuning Workflow

This document describes the current operator workflow for subject-mask tuning
and canary body/swim-mask generation.

## Scope

- Traditional `subject_body` tuning and materialization.
- Batch propagation of `subject_mask_tuning` by `camera_id`.
- First-pass refinement/review of `subject_body` and `swim_bladder` masks.
- Direct multi-source assembly into `refined_subject_masks_runs`.
- Transitional eye-component sourcing from legacy eye stages when needed, while
  treating unified subject-mask component views as the preferred operator
  surface for eye/body/swim status.

This is the current practical workflow for canary training zarrs. It is not a
contract doc.

For explicit save-versus-approval semantics across detect, keypoints, eye
masks, and subject masks, see
[mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md).

## Current Stage Split

- Tuning metadata lives in:
  - `analysis_metadata.attrs["subject_mask_tuning"]`
- Raw subject masks live in:
  - `subject_mask_runs/<run>`
- Refined editable subject masks live in:
  - `refined_subject_masks_runs/<run>`
- Legacy eye-specific stages may still be present during transition:
  - `eye_masks_runs/<run>`
  - `refined_eye_masks_runs/<run>`

Operator policy:

- prefer `subject_masks` / `refined_subject_masks` component summaries when
  asking whether a recording has usable body/eye/swim masks
- treat legacy eye-specific stages as compatibility inputs or diagnostics, not
  as the preferred canonical surface for new work
- for manual eye review/edit, prefer the unified refined-subject review surface
  reached through `scripts/py -m fisheye.tune.eye_mask_review --manual`
  or directly through `scripts/py -m fisheye.tune.refined_subject_mask_review`

For the raw stage contract, see
[subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md).

## Preconditions

1. Archive contains a materialized crop run with `roi_images`.
2. Archive contains a background run.
3. Archive has dish-mask tuning if dish gating should be applied.
4. Archive is a training or analysis zarr where `analysis_metadata` attrs are
   writable.

## Recommended Procedure: Traditional `subject_body`

### 1. Tune `subject_body` interactively

```bash
scripts/py -m fisheye.tune.subject_mask_tuner <archive>.zarr --component subject_body
```

What the UI currently shows:

- cropped ROI
- ROI-local dish mask preview when present
- stored mask preview if a subject-mask run is provided
- dish-masked traditional proposal

Saved output:

- `analysis_metadata.attrs["subject_mask_tuning"].components["subject_body"]`

Notes:

- The tuner preview now applies the saved dish mask so it matches the
  materializer behavior.
- Saving writes tuning metadata only. It does not write a new
  `subject_mask_runs` run.

### 2. Optionally propagate tuning by camera

If the same camera setup is shared across recordings, you can copy the
subject-mask tuning to other zarrs with the same `camera_id`.

Dry run:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera \
  /nvme1/recordings \
  --source <source_training>.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --merge-dicts
```

Apply:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera \
  /nvme1/recordings \
  --source <source_training>.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --merge-dicts \
  --apply
```

Important behavior:

- `--keys subject_mask_tuning` copies only the subject-mask tuning payload.
- `--merge-dicts` recursively merges dict-like payloads instead of replacing the
  whole top-level attr.
- This is useful when the source only updates
  `components.subject_body`, while targets may already contain unrelated entries
  such as `components.eyes_union`.
- Without `--merge-dicts`, a top-level overwrite would replace the entire
  `subject_mask_tuning` object.
- By default, the tool only targets the same `zarr_use` as the source.

Component-scoped propagation:

- `--subject-mask-components <name>` limits propagation to selected
  `subject_mask_tuning.components[...]` entries.
- unrelated target components are preserved.
- matching target components are skipped unless `--overwrite` is given.
- with `--overwrite --merge-dicts`, the selected component subtree is merged so
  matching source fields win while unrelated target fields inside that
  component remain intact.

Example: propagate only swim-bladder tuning by camera:

```bash
scripts/py -m fisheye.utils.apply_tuning_by_camera \
  /nvme1/recordings \
  --source <source_training>.zarr \
  --recursive \
  --keys subject_mask_tuning \
  --subject-mask-components swim_bladder \
  --merge-dicts
```

### 3. Materialize a raw subject-mask run

```bash
scripts/py -m fisheye.segmentation.subject_segmentation \
  <archive>.zarr \
  --run-name traditional_subject_masks_canary_001
```

Use `--overwrite` if rerunning after retuning:

```bash
scripts/py -m fisheye.segmentation.subject_segmentation \
  <archive>.zarr \
  --run-name traditional_subject_masks_canary_001 \
  --overwrite
```

Current behavior:

- reads the saved `subject_body` tuning
- reads the selected crop run and background run
- applies dish-mask gating when dish-mask tuning exists
- writes a body-only `subject_mask_runs/<run>`
- records `run_semantics = "traditional_subject_body_inference"`

### 4. Refine / review the body masks

```bash
scripts/py -m fisheye.tune.refined_subject_mask_review \
  <archive>.zarr \
  --subject-run traditional_subject_masks_canary_001 \
  --components subject_body
```

This creates or reopens a `refined_subject_masks_runs/<run>` entry and lets the
operator paint or erase masks per ROI.

### 5. Inspect raw vs refined subject masks

```bash
scripts/py -m fisheye.visualization.subject_mask_inspector \
  <archive>.zarr \
  --subject-run traditional_subject_masks_canary_001 \
  --refined-run refined_subject_masks_001 \
  --component subject_body
```

Current inspector behavior:

- opens the crop ROI alongside raw and refined subject-mask overlays
- shows run/method/review metadata for the active component
- shows common geometry and refined component QC summaries
- draws the active component's external contour and convex hull on the raw and
  refined panels, which is especially useful when explaining solidity-driven
  flags
- supports ROI/component navigation without entering edit mode
- supports cycling the active component with `[` / `]` in addition to direct
  `1..9` selection
- supports jumping to the next flagged ROI for the active component based on
  current refined QC thresholds

This is intended as a Palette-native read-mostly inspection surface. It is not
a replacement for Paintera's heavier pixel-editing interaction model.

## Unified Eye Review During Transition

For eye components, the current canonical manual review path now goes through
the unified subject-mask editor, not directly through `refined_eye_masks_runs`.

Recommended command:

```bash
scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --manual
```

Current behavior:

- projects the selected `refined_eye_masks_runs/<run>` into a compatibility
  `subject_mask_runs/<run>`
- opens or reopens a matching `refined_subject_masks_runs/<run>` eye-component
  review surface
- uses the same per-component review payloads as body/swim refinement
- when canonical eye masks or eye component review states are saved there, the
  matching `refined_eye_masks_runs/<run>` compatibility artifact is refreshed
  for legacy eye-specific consumers

Legacy fallback:

```bash
scripts/py -m fisheye.tune.eye_mask_review <archive>.zarr --legacy-manual
```

Use the legacy mode only when the older failure-local eye UI is specifically
needed.

### 6. Recompute refined metadata in batch mode

When a refined run already has edited `masks_roi` data and the goal is to
refresh derived metadata rather than paint interactively, use the non-UI batch
apply path.

Dry run:

```bash
scripts/py -m fisheye.refinement.refine_subject_masks \
  <archive>.zarr \
  --source-run traditional_subject_masks_canary_001 \
  --run-name refined_subject_masks_001 \
  --component subject_body \
  --dry-run
```

Apply:

```bash
scripts/py -m fisheye.refinement.refine_subject_masks \
  <archive>.zarr \
  --source-run traditional_subject_masks_canary_001 \
  --run-name refined_subject_masks_001 \
  --component subject_body \
  --scheduler processes
```

Current batch behavior:

- `--dry-run` prints the resolved source run, target refined run, selected
  components, ROI count, chunk count, and scheduler without mutating the
  archive.
- `--run-name` always targets one explicit
  `refined_subject_masks_runs/<run>` entry. If that run already exists, the
  tool updates its derived metadata in place for the selected components/ROI
  rows.
- the batch apply path recomputes metrics, QC, reasons, `updated_at_utc`, and
  `dask_*` attrs from the current `masks_roi` contents
- the inspector remains the recommended read-mostly verification surface after
  batch apply

Canary validation on 2026-03-31:

- archive:
  - `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
- source run:
  - `subject_masks_canary_sam_points_body_eyes_001`
- refined run:
  - `refined_subject_masks_canary_sam_points_body_001`
- component:
  - `subject_body`
- full apply result:
  - `changed_roi_count = 0`
  - `noop_roi_count = 227`

That result means the stored `subject_body` metadata in the canary refined run
was already internally consistent with its current `masks_roi` state.

## Eye-Union Note

`subject_mask_tuner --component eyes_union` is a separate path.

Current behavior:

- uses the eye-specific threshold/Sobel method
- stores tuning under `subject_mask_tuning.components["eyes_union"]`
- mirrors `eye_mask_tuning` for compatibility with the existing traditional eye
  segmentation path

That is useful for eye workflows, but it is not the current body-mask
materialization path.

## Swim-Bladder Note

Swim bladder now has a dedicated local tuner:

```bash
scripts/py -m fisheye.tune.swim_bladder_mask_tuner <archive>.zarr
```

Current behavior:

- centers a local patch on the `swim_bladder` keypoint
- supports both:
  - `swim_bladder_patch_threshold_v1`
  - `swim_bladder_polar_boundary_v1`
- stores parameters under
  `subject_mask_tuning.components["swim_bladder"]`

For ring-like canaries, use the boundary-oriented family explicitly:

```bash
scripts/py -m fisheye.tune.swim_bladder_mask_tuner <archive>.zarr \
  --method-family polar_boundary
```

The polar family is the better fit when the useful cue is a contrast ring or
boundary rather than a uniformly dark filled interior. See
[swim_bladder_polar_boundary_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_polar_boundary_design.md).

This is intentionally separate from the ROI-wide `subject_mask_tuner` because
the operator task is patch-local and keypoint-centered rather than whole-ROI.

The saved tuning is ready for future traditional swim-bladder materialization,
and that materializer now exists:

```bash
scripts/py -m fisheye.segmentation.swim_bladder_segmentation \
  <archive>.zarr \
  --run-name traditional_swim_bladder_masks_canary_001
```

Current materializer behavior:

- reads `subject_mask_tuning.components["swim_bladder"]`
- dispatches by saved `subject_method_family`
- centers each proposal on the chosen swim-bladder keypoint run
- writes a raw `subject_mask_runs/<run>` entry with only the
  `swim_bladder` channel available
- records `run_semantics = "traditional_swim_bladder_inference"`
- snapshots the exact tuning entry it used on the produced run for auditability

## Swim-Bladder Save / Approval Policy

For the current canary phase, use the same editor surface as body masks, but
apply a stricter review interpretation:

- `s` saves ROI-local improvements and does not imply approval
- `a` on `swim_bladder` should mean the whole component on that refined run is
  training-usable, not merely that the current ROI looks acceptable
- keep empty swim-bladder masks conservative; ambiguity is not an automatic
  negative label

Detailed heuristics now live in:

- [swim_bladder_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_review_policy.md)

Short practical rule:

- save whenever a local edit is materially better than the stored mask
- approve only after the component backlog for that run has been reviewed
  end-to-end and no systematic swim-bladder failure mode remains

## Direct Multi-Source Assembly Into `refined_subject_masks_runs`

When body, eye, and swim-bladder source runs already exist, the preferred
sparse multi-source canary path is now to assemble directly into
`refined_subject_masks_runs/<run>` and finalize that run immediately.

Dry run:

```bash
scripts/py -m fisheye.refinement.assemble_refined_subject_masks \
  <archive>.zarr \
  --body-run <body_subject_run> \
  --eye-run <eye_subject_run> \
  --swim-run <swim_subject_run> \
  --run-name refined_subject_masks_<purpose>_001 \
  --dry-run
```

Apply:

```bash
scripts/py -m fisheye.refinement.assemble_refined_subject_masks \
  <archive>.zarr \
  --body-run <body_subject_run> \
  --eye-run <eye_subject_run> \
  --swim-run <swim_subject_run> \
  --run-name refined_subject_masks_<purpose>_001
```

Current assembler behavior:

- validates that the source runs are row-aligned and crop-compatible
- seeds each component from its own source run
- writes the assembled result into the unified refined subject-mask stage
- writes component-scoped provenance for the assembled refined run
- runs the normal subject-mask finalization step in the same command, so the
  output run already has canonical refined metrics, QC, reasons, and review
  scaffolding
- avoids requiring a merged raw `subject_mask_runs/<run>` intermediate for
  sparse workflows

Current implementation note:

- `--body-run`, `--eye-run`, and `--swim-run` currently name
  `subject_mask_runs/<run>` sources
- if eye content starts in `eye_masks_runs` or `refined_eye_masks_runs`, first
  project/backfill it into a compatibility `subject_mask_runs/<run>`
- direct `refined_eye_masks_runs` -> `refined_subject_masks_runs` assembly is a
  future extension, not the current CLI behavior

Canary validation on 2026-04-01:

- archive:
  - `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`
- body source:
  - `subject_masks_canary_sam_points_body_eyes_001`
- eye source:
  - `subject_masks_from_refined_eye_masks_2026-02-12_19-51-24`
- swim source:
  - `traditional_swim_bladder_masks_canary_001`
- refined output:
  - `refined_subject_masks_canary_body_eyes_swim_001`
- output schema:
  - `label_schema_id = "subject_v1_lr"`
  - `mask_labels = ["subject_body", "eye_left", "eye_right", "swim_bladder"]`

## Current Batch Propagation Recommendation

For subject-body tuning, prefer:

- source zarr from the camera you tuned interactively
- `--keys subject_mask_tuning`
- `--merge-dicts`
- dry run before `--apply`

That gives the safest behavior for mixed subject-mask payloads.

## Current Limitations

- `subject_segmentation.py` is currently single-process and canary-oriented.
- The batch copier groups by `camera_id`, not by dish design or protocol.
- `subject_mask_tuning` propagation is attr-based; it does not automatically
  rerun segmentation after copying.
- `refined_subject_masks_runs` exists, but body/swim-bladder curation is still
  early and should be treated as a canary workflow.

For deferred scaling work, see
[traditional_subject_segmentation_scaling_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/traditional_subject_segmentation_scaling_todo.md).
