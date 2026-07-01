# SAM3 Subject-Mask Canary Plan

## Purpose

Define the first concrete canary for using SAM3 to generate whole-fish masks
from existing Palette training zarr data, with outputs written back into
`subject_mask_runs`.

This plan is intentionally narrow:

- one canary training zarr
- one whole-fish/body channel
- one automatic prompt source
- one write-back target in Palette

The goal is to validate the Palette -> SAM -> `subject_mask_runs` loop before
deciding whether SAM3 should become a longer-lived dependency inside `palette`.

Observed canary results and the current Paintera editing workflow are captured
in:

- `docs/paintera_palette_subject_mask_workflow.md`

## Canary Target

Primary canary archive:

- `/nvme1/recordings/2026-01-28T22-15-03Z_arena_1_DefaultScreen/zarr/2026-01-28T22-15-03Z_arena_1_DefaultScreen_training.zarr`

Relevant existing runs in that archive:

- crop run:
  - `crop_runs/crop_2026-02-03_23-32-21`
- refined keypoint run:
  - `refined_keypoints_runs/refined_keypoints_2026-02-04_12-43-46`
- current subject-mask canary:
  - `subject_mask_runs/subject_masks_canary_001`

## Why SAM3 Is Interesting Here

The SAM3 repo notes already make the intended use clear:

- use Palette ROI crops as the segmentation image domain
- use the available refined pose keypoints as positive point prompts
- write whole-fish masks back into `subject_mask_runs`
- treat SAM3 primarily as a teacher/pseudo-label generator, not necessarily the
  final production runtime segmenter

Relevant SAM3 notes:

- [`../sam3/docs/palette_zarr_subject_segmentation_workflow.md`](/home/delahantyj@hhmi.org/gitrepos/sam3/docs/palette_zarr_subject_segmentation_workflow.md)
- [`../sam3/docs/palette_zarr_sam_integration_todo.md`](/home/delahantyj@hhmi.org/gitrepos/sam3/docs/palette_zarr_sam_integration_todo.md)

Observed runtime follow-up as of `2026-04-05`:

- the cache-backed `geometry_only` analysis path has now been validated for the
  current Palette SAM wrapper
- on the representative analysis archive, a warm-cache rerun at
  `--batch-size 64` completed in about `47:22`
- the persisted timing profile showed that ROI loading was not the bottleneck:
  - `model_predict`: about `2416.0s` (`85.2%`)
  - `output_write`: about `295.7s` (`10.4%`)
  - `roi_read`: about `29.0s` (`1.0%`)

Interpretation:

- the shared crop reader and temporary ROI cache are already adequate for the
  current SAM workflow
- remaining slowness is mostly SAM runtime cost, not Palette-side dataloader
  cost
- this is acceptable for now because SAM is being treated primarily as a
  pseudo-label / teacher path for smaller U-Net-style models, not as the
  preferred fast runtime segmenter

## First-Phase Scope

The canary should do only this:

1. read ROI crops from the canary training zarr
2. read aligned refined keypoints from the same zarr
3. use all available refined keypoints as automatic positive prompts
4. run SAM segmentation on the ROI crop
5. write a new `subject_mask_runs/<run>` containing only a `subject_body` mask
6. record enough scores/metadata to decide whether the masks are worth curating

Deliberately out of scope for phase 1:

- eye segmentation
- swim-bladder mask segmentation
- direct writing into `refined_subject_masks_runs`
- training SAM3 inside Palette
- text prompting
- broad pipeline wiring

## Recommended Prompt Policy

Use all available refined keypoints as the default positive prompt set.

Source array:

- `refined_keypoints_runs/<run>/keypoints_roi[:, :, :]`

Why:

- empirical canaries performed better with all available anatomical landmarks
  than with swim-bladder-only prompting
- multiple positive points constrain the body extent more strongly than a
  single interior anchor
- this still avoids the weaker detection-center prompt
- it preserves a fully automatic workflow

Optional later prompt additions:

- detection-derived box prompt
- negative points near ROI borders
- prompt-label ablations, including swim-bladder-only prompting, for diagnostics

Current runtime behavior:

- omitting `--positive-keypoint-labels` uses all labels available on the
  resolved keypoint run
- pass `--positive-keypoint-labels swim_bladder` only for a diagnostic ablation,
  not as the default body-mask canary policy

### Crop-Local Prompt Coordinates

SAM prompt construction for acquisition crop-video training rows is crop-local.
If the prompt policy uses keypoints, ROI-inset boxes, or fixed border/corner
negative points, full-frame dimensions are not needed for SAM inference. The
inputs already live in the decoded crop frame, for example `384x384` luma
images with `keypoints_roi` in the same pixel space.

Full-frame dimensions are only required when a prompt source must project a
full-frame detection box into ROI coordinates, such as a detector-normalized
`bbox_norm_coords` path. Early RedScare acquisition crop-video runs currently
carry crop-frame-normalized values in `bbox_norm_coords`; this is a known
contract gap, not the target schema. Until those runs are backfilled, prompt
code must reject noncanonical `bbox_norm_coords` for full-frame projection and
use crop-local prompts such as keypoints, `bbox_roi_xyxy`, ROI-inset boxes, or a
future `bbox_crop_norm_coords` instead.

Even when SAM inference is crop-local, the written `subject_mask_runs/<run>`
must preserve row and crop lineage so outputs can be placed back into the
parent recording:

- `source_crop_run`
- `source_crop_row_ids`
- `frame_indices`
- `roi_coordinates_full` / full-frame `source_crop_xywh`
- source crop-video frame indices and acquisition-local frame IDs when present

That lineage is what lets downstream readers derive full-frame pixel or
full-frame normalized geometry from ROI-local masks and keypoints.

## Recommended Output Policy

Phase-1 SAM outputs should write a new `subject_mask_runs/<run>` with:

- `label_schema_id = "subject_v1_union"`
- `mask_labels = ["subject_body", "eyes_union", "swim_bladder"]`
- only `subject_body` available
- `eyes_union` unavailable
- `swim_bladder` unavailable

That means:

- `available_channels = [true, false, false]`
- `masks_roi[:, 0]` = whole-fish binary mask
- `masks_roi[:, 1:]` = zero placeholders

This keeps the output compatible with the current Palette subject-mask contract
without inventing fake supervision for other anatomy channels.

### SAM3 Output Semantics Versus U-Net Finalization

SAM3 body-mask creation currently writes the selected SAM candidate directly.
For each eligible ROI, the wrapper asks SAM for one or more candidate masks,
selects the candidate with the highest SAM-predicted quality score, then writes:

- `masks_roi[:, subject_body] = selected_logits > 0`
- `mask_probs_roi[:, subject_body] = sigmoid(selected_logits)`
- `metrics/sam_quality_score[:, subject_body] = selected_candidate_score`

Palette does not run the U-Net smart-finalizer cleanup on this SAM3 output.
That means the SAM3 creation path does **not** apply additional Palette-side:

- probability threshold sweeps beyond SAM's selected-mask zero-logit boundary
- morphology closing
- hole filling
- keep-largest-component cleanup
- removed-mass / changed-area finalization metrics

This is deliberate for the first SAM3 body canary. The SAM-selected mask is the
artifact under evaluation, and downstream refined runs should record that
provenance rather than silently converting it into a U-Net-style finalized mask.

By contrast, U-Net `subject_mask_runs` are normally refined into
`refined_subject_masks_runs` through `smart_finalize_subject_masks_v1`, where
component-specific thresholding and cleanup policies can fill holes, close small
gaps, keep the largest component, and compute finalization QC metrics. If SAM3
masks later need the same cleanup, that should be added as an explicit
SAM3-refinement policy with recorded parameters, not hidden inside raw SAM3
creation.

## Row Eligibility Policy

Phase-1 segmentation should be fail-closed.

Segment a row only if all of the following are true:

- the ROI exists
- the refined keypoint row exists and aligns by row index
- at least one selected positive keypoint is finite and in ROI bounds
- for the preferred all-keypoint policy, the prompt-point count is recorded so
  rows with partial prompts can be audited
- the row is not otherwise marked unusable by refined keypoint validity policy

Skip or mark failed if:

- swim bladder is missing
- swim bladder is NaN or off-image
- row alignment is ambiguous

Open question for the canary:

- should interpolated rows be skipped entirely in phase 1?

Recommended initial policy:

- segment only non-interpolated rows first

That makes the first QC pass much easier to interpret.

## Runtime Entry Point

Add a small Palette-side entrypoint rather than embedding logic into the
existing `subject_mask_tuner`.

Recommended new utility:

- `src/fisheye/utils/run_sam_subject_masks.py`

Suggested CLI shape:

```bash
scripts/py -m fisheye.utils.run_sam_subject_masks \
  /path/to/recording_training.zarr \
  --crop-run crop_... \
  --keypoint-run refined_keypoints_... \
  --output-run sam_subject_masks_001 \
  --mode body-only
```

Required behaviors:

- resolve crop and keypoint runs
- verify row alignment
- convert grayscale ROI to 3-channel RGB
- call the SAM predictor
- write a Palette-compatible subject-mask run
- record run provenance and SAM checkpoint identity

## Implementation Phases

### Phase 1. Loader and alignment

- [ ] Read `roi_images` from the selected crop run.
- [ ] Read `keypoints_roi`, `frame_indices`, and `detection_indices` from the
      selected refined keypoint run.
- [ ] Verify row alignment between crop rows and keypoint rows.
- [ ] Define explicit failure behavior for rows with missing prompts.

### Phase 2. SAM adapter

- [ ] Build a minimal SAM adapter that accepts:
  - ROI image
  - positive point prompt
  - optional box prompt
- [ ] Start with the predictor path that best matches automatic point prompts.
- [ ] Return:
  - mask logits or probabilities if available
  - final selected mask
  - confidence/score metadata

### Phase 3. Palette write-back

- [ ] Create a new `subject_mask_runs/<run>`.
- [ ] Write:
  - `masks_roi`
  - `mask_probs_roi`
  - `available_channels`
  - lineage arrays
  - `metrics/prob_max`
  - `metrics/mask_present`
- [ ] Record attrs for:
  - SAM method family
  - checkpoint identifier
  - prompt policy
  - source crop run
  - source keypoint run

### Phase 4. QC and review

- [ ] Add a first-pass QC summary:
  - how many rows had usable prompts
  - how many masks were non-empty
  - mask area distribution
  - rows touching ROI borders
- [ ] Review the resulting body masks in a canary archive before broader export.
- [ ] Decide whether the masks are good enough to seed
      `refined_subject_masks_runs`.

### Phase 5. Training use

- [ ] Export curated SAM-derived `subject_body` masks as training supervision.
- [ ] Use them as teacher labels for a smaller prompt-free subject/body model.
- [ ] Keep SAM3 as a recovery / pseudo-label path unless the runtime proves
      stable enough to justify broader adoption.

## Submodule Decision

Question:

- should `palette` ship SAM3 as a git submodule?

### Recommendation

Do **not** make SAM3 a required `palette` submodule yet.

Recommended phase-1 policy:

- keep SAM3 as an external sibling checkout for now
- write the Palette adapter in `palette`
- require either:
  - an importable `sam3` Python package, or
  - a configured local path / env var pointing to the SAM3 checkout

### Why not make it a required submodule yet

1. SAM3 is not yet proven for this exact fish-body workflow.
   We should validate the canary before tightening repository coupling.

2. The dependency burden is nontrivial.
   The SAM3 repo brings its own dependency surface, checkpoint download flow,
   and Hugging Face access requirements.

3. Most Palette users probably do not need SAM3 for routine work.
   This looks more like an advanced optional segmentation path than a core
   universal dependency.

4. Palette already has one submodule (`decord`), and the repo cleanup notes
   explicitly treat vendored/external dependency choices as a maintenance cost.

5. SAM3 currently fits best as a teacher/pseudo-label generator.
   That argues for optional integration first, tighter embedding later only if
   it becomes operationally central.

### When a submodule would become reasonable

Revisit the submodule decision only if most of the following become true:

- the canary succeeds
- the lab wants a pinned, reproducible SAM3 commit inside Palette workflows
- the SAM3 runtime is used repeatedly enough to justify tighter repo coupling
- the setup/auth/checkpoint story is documented and stable
- the Palette adapter is no longer just experimental

### If a submodule is chosen later

Use it as an optional integration, not as a hard requirement for basic Palette
use.

That means:

- do not make baseline Palette commands require SAM3 to import
- gate the SAM entrypoint behind an explicit check
- keep the error message actionable when SAM3 is unavailable

## Decision Summary

Recommended now:

- implement a Palette-side SAM canary runtime
- use the existing sibling `sam3` checkout
- for cluster jobs, mirror that checkout to a compute-visible shared path such
  as `/groups/johnson/johnsonlab/jeremy/gitrepos/sam3`
- segment only `subject_body`
- use all available refined keypoints as positive prompts
- keep the result in `subject_mask_runs`
- defer the submodule decision until after the canary proves useful

## Cluster Smoke Workflow

The current cluster path is a single-job bsub wrapper around
`fisheye.utils.run_sam_subject_masks`:

```bash
scripts/submit_sam_subject_masks_bsub.sh \
  --zarr /groups/.../recording_analysis.zarr \
  --crop-run <crop_run> \
  --keypoint-group refined_keypoints_runs \
  --keypoint-run <refined_keypoints_run> \
  --output-run <planned_subject_mask_run> \
  --sam3-root /groups/johnson/johnsonlab/jeremy/gitrepos/sam3 \
  --apply \
  --apply-limit 16 \
  --profile-timings \
  --submit
```

`--apply-limit` is deliberate. The writer creates a normal
`subject_mask_runs/<run>` surface, so a full apply can touch the whole eligible
row surface. The bsub wrapper refuses `--apply` without `--apply-limit` unless
`--allow-full-apply` is passed explicitly.

SAM3 remains an external dependency. A compute-visible checkout can be mirrored
from the workstation with:

```bash
rsync -a --delete \
  --exclude outputs/ \
  --exclude .ipynb_checkpoints/ \
  /home/delahantyj@hhmi.org/gitrepos/sam3/ \
  /groups/johnson/johnsonlab/jeremy/gitrepos/sam3/
```

Prefer passing a compute-visible `--checkpoint` plus `--no-hf-download` once a
specific SAM3 checkpoint has been selected. Without that, the SAM3 runtime may
try its own default/Hugging Face checkpoint resolution.

## Related Docs

- [subject_mask_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_runs_contract.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)
- [../sam3/docs/palette_zarr_subject_segmentation_workflow.md](/home/delahantyj@hhmi.org/gitrepos/sam3/docs/palette_zarr_subject_segmentation_workflow.md)
- [../sam3/docs/palette_zarr_sam_integration_todo.md](/home/delahantyj@hhmi.org/gitrepos/sam3/docs/palette_zarr_sam_integration_todo.md)
