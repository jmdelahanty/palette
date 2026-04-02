# Swim-Bladder Review Policy

<!-- design-meta
status: active
last_verified: 2026-04-02
-->

Purpose: define the near-term save and approval heuristics for
`swim_bladder` masks in `refined_subject_masks_runs` during the current canary
body/eye/swim rollout.

This is an operator policy note, not a storage contract. The canonical save and
approval mechanics still live in
[mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md).

## Scope

This policy applies to:

- canary curation in `refined_subject_masks_runs`
- `swim_bladder` component review in
  [refined_subject_mask_review.py](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/tune/refined_subject_mask_review.py)
- both current traditional proposal families:
  - `swim_bladder_patch_threshold_v1`
  - `swim_bladder_polar_boundary_v1`

This policy does not require the proposal source itself to be perfect. It
defines when the refined component is usable enough to save or approve.

## Core Decision

For the current canary phase:

- saving remains ROI-local and incremental
- approval remains component-level on the refined run
- `approved` for `swim_bladder` should mean the reviewer believes the whole
  active component on that refined run is training-usable, not merely that the
  current ROI looks acceptable

That means:

- use `s` freely while editing individual ROIs
- use `a` only when the component-level backlog for that run is actually in a
  stable, training-usable state

## Good Enough To Save

Saving a swim-bladder edit is justified when the current ROI mask is materially
better than the stored version, even if the component is not yet ready for
approval.

Good enough to save:

- the mask is centered on the correct local anatomy near the swim-bladder
  keypoint
- obvious leakage into background, eyes, or broad body regions has been
  removed
- the saved mask better matches the intended structure than the prior stored
  mask or raw proposal
- the shape may still be imperfect, but the ROI is moving toward the intended
  label rather than away from it

Not enough to save as a "finished" ROI:

- the mask is still attached to the wrong structure
- the mask is mainly capturing body edge, eye edge, or patch artifact
- the ROI is ambiguous enough that saving would likely encode a fabricated
  training target rather than a correction

Practical rule:

- save whenever you have made a real local improvement
- do not treat save as approval

## Good Enough To Approve

For the first canary datasets, approving `swim_bladder` should be conservative.
The active component should be marked `approved` only when there is no known
systematic issue left in that component for the refined run.

Approval criteria for the component:

- reviewed ROIs no longer show a recurring failure mode such as keypoint drift,
  body-edge leakage, or persistent off-center proposals
- the component is anatomically localized consistently enough that a downstream
  trainer would learn the swim bladder rather than nearby clutter
- the component mask semantics are internally consistent across the run
  - compact filled masks are acceptable
  - polar-boundary-derived masks are also acceptable
  - the visible cue may be ring-like, but the refined label should still
    represent the enclosed swim-bladder region rather than a decorative thin
    ring
- any remaining bad ROIs are isolated cleanup items, not evidence of a method
  or policy mismatch
- the reviewer is comfortable treating the component as supervised training
  data for that run

For current canaries, prefer approving only after the reviewer has inspected
all ROIs where `swim_bladder` is intentionally available, or has otherwise
closed the known backlog for that run.

## When To Use `pending`, `needs_review`, And `rejected`

Use `pending` when:

- editing has started but the component has not yet been reviewed end-to-end
- the run is still in active cleanup

Use `needs_review` when:

- there are still uncertain ROIs that need another pass
- the reviewer cannot yet tell whether the remaining masks are biologically
  right or merely plausible
- the current source method may be usable, but the component is not ready to
  gate exports or training

Use `rejected` when:

- the component semantics for the run are wrong in a systematic way
- the proposal family or seed source is producing the wrong structure often
  enough that the run should not be treated as a curation baseline
- the reviewer would rather regenerate or retune than continue incremental
  cleanup

## Explicit-Negative Policy

Swim-bladder negatives must be deliberate.

Important rule from the training/export side:

- do not fabricate negatives for channels that were never actually supervised

Operational implication for swim bladder:

- an all-zero swim-bladder mask is acceptable only when the reviewer intends it
  as an explicit negative for that ROI
- ambiguity is not the same thing as an explicit negative

Do not approve an empty swim-bladder label merely because:

- the proposal failed
- the patch is noisy
- the keypoint is suspicious
- the anatomy is hard to see in that frame

If the ROI is ambiguous rather than confidently negative, keep the component in
`pending` or `needs_review` until the ambiguity is resolved or excluded by a
later export/filtering rule.

## Boundary-Oriented Source Policy

`swim_bladder_polar_boundary_v1` is an acceptable canary source family.

Review implication:

- the visible cue may be a partial or noisy boundary ring
- the refined label is still a binary region mask
- approval should focus on whether the final refined region encloses the right
  anatomy with acceptable leakage, not on whether the source cue looked like a
  perfect dark blob

In other words:

- ring-like image evidence is fine
- thin boundary-only labeling is not the target refined semantics

## Reviewer Notes Recommendation

When marking `swim_bladder` approved for a canary run, prefer attaching a short
session note with `--review-notes`, for example:

- `swim bladder canary approved after full ROI pass`
- `polar boundary source; explicit negatives only where confident absent`

This is not a hard guardrail, but it helps distinguish:

- careful component approval
- from a casual state toggle during exploration

## Downstream Gating

Until larger curated datasets exist:

- component-specific training/export should gate on
  `component_review_statuses["swim_bladder"].state == "approved"`
- consumers should not treat mere component availability as training readiness
- empty masks should not be interpreted as safe negatives unless they were
  intentionally curated as such

## References

- [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md)
- [mask_review_save_approval_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/mask_review_save_approval_policy.md)
- [swim_bladder_polar_boundary_design.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_polar_boundary_design.md)
- [subject_mask_training_artifact_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_training_artifact_contract.md)
