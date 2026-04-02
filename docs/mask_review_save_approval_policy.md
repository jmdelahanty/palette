# Mask Review Save And Approval Policy

<!-- design-meta
status: active
last_verified: 2026-04-02
-->

## Purpose

Document the current save and approval semantics for mask-oriented review
workflows, and compare them against the existing detect/keypoint review model.

This is not a contract for every review-status payload in the repo. Detect and
keypoint review-status schema remains defined by
[review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md).

Instead, this note answers a narrower operator question:

- when does a write count as a saved edit?
- when does a write count as an approval or review-state change?
- what is the granularity of approval?
- what attrs/pointers are updated?

## Shared Principle

Across the current refinement stack, saving edits and approving review status
are separate actions.

That means:

- saving pixels, reasons, or derived metadata does not by itself imply approval
- approval/review-state changes should write explicit review payloads
- downstream consumers should gate on review payloads, not on the mere presence
  of a refined run

## Comparison Matrix

| Workflow | Editable target | Save granularity | Approval granularity | Review payload home | Parent latest pointer |
| --- | --- | --- | --- | --- | --- |
| Detect | Manual subgroup under `refined_detect_runs/<run>/` | subgroup-level corrections | whole refined detect run | `detect_review_status` | `detect_review_status_latest` |
| Keypoints | `refined_keypoints_runs/<run>` | run-local row edits | whole refined keypoint run | `keypoint_review_status` | `keypoint_review_status_latest` |
| Eye masks | `refined_eye_masks_runs/<run>` | per-ROI manual correction | whole refined eye-mask run | `eye_mask_review_status` | `eye_mask_review_status_latest` |
| Subject masks | `refined_subject_masks_runs/<run>` | per-ROI component-aware save | per-component, with run aggregation | `component_review_statuses` + `refined_subject_mask_review_status` | `refined_subject_mask_review_status_latest` |

## Detect

Current policy:

- raw detect runs are append-only provenance
- refined detect runs are immutable stage outputs
- manual corrections live in a separate subgroup under the refined run
- approval is whole-run, not per subgroup component

Operational implication:

- detect treats "manual review" as choosing or extending the preferred refined
  subgroup, not mutating the canonical refined detect arrays in place
- approval then controls downstream crop source resolution

References:

- [detection_refinement_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/detection_refinement_workflow.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)

## Keypoints

Current policy:

- `refined_keypoints_runs/<run>` is the editable artifact
- row-level reason tags and refined outputs can be changed without implying
  approval
- approval is run-level
- both manual and algorithmic approvals are supported
- strict review-writing guardrails already exist for CLI acceptance helpers

Operational implication:

- keypoint save/edit semantics are mutable like masks, but approval remains a
  single run-level decision
- training/export gates should key off `keypoint_review_status`, not just the
  existence of refined keypoints

References:

- [keypoint_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/keypoint_review_policy.md)
- [review_status_schema_unification_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/review_status_schema_unification_contract.md)

## Eye Masks

Current policy:

- `refined_eye_masks_runs/<run>` is the editable refined artifact
- manual review saves per-ROI mask/ellipse/contour corrections in place on that
  refined run
- approval is still whole-run, not per-eye
- review-state writes happen from review UIs rather than a dedicated
  mask-specific policy doc

Current review payload:

- attr: `eye_mask_review_status`
- parent pointer: `eye_mask_review_status_latest`
- states used in UI:
  - `approved`
  - `pending`
  - `rejected`
  - `needs_review`

Important current limitation:

- eye-mask review payloads now write canonical `timestamp_utc` and mirror
  legacy `timestamp` for compatibility
- eye-mask review is still not yet covered by the shared detect/keypoint review
  status contract

Operational implication:

- eye-mask save behavior is well-established
- eye-mask approval behavior exists and is functional
- but the policy is documented mostly procedurally in the workflow doc rather
  than in a dedicated contract/policy note

References:

- [src/fisheye/docs/eye_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/docs/eye_mask_tuning_workflow.md)

## Subject Masks

Current policy:

- `refined_subject_masks_runs/<run>` is the canonical editable working artifact
- save writes are per-ROI and recompute sibling refined metadata immediately
- approval is per-component, not whole-run
- run-level review state is derived automatically from component review states

Current review payloads:

- run-level attr: `refined_subject_mask_review_status`
- component mapping attr: `component_review_statuses`
- parent pointer: `refined_subject_mask_review_status_latest`

Current component states:

- `approved`
- `pending`
- `rejected`
- `needs_review`

Current aggregation policy:

- if any component is `rejected`, run state becomes `rejected`
- else if any component is `needs_review`, run state becomes `needs_review`
- else if all components are `approved`, run state becomes `approved`
- else if any component is `pending`, run state becomes `pending`

Operational implication:

- subject-mask save/approval semantics are more expressive than eye-mask review
  because the run may legitimately be mixed:
  - body approved
  - swim bladder still pending
  - eyes seeded but not yet reviewed
- downstream gating should therefore prefer component review state when the
  consumer is component-specific

Current unresolved policy question:

- what should count as "good enough to save" or "good enough to approve" for
  early swim-bladder masks is still open and remains a workflow/data-quality
  decision rather than a storage problem

References:

- [subject_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_tuning_workflow.md)
- [refined_subject_masks_runs_contract.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/refined_subject_masks_runs_contract.md)
- [subject_mask_refinement_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/subject_mask_refinement_todo.md)

## Recommended Near-Term Policy

For mask workflows, the near-term policy should be:

1. Keep save and approval separate.
2. Keep eye-mask approval run-level during transition.
3. Keep subject-mask approval component-level.
4. Treat component review state as the canonical gate for component-specific
   subject-mask exports and registry views.
5. Continue using the shared state vocabulary:
   - `approved`
   - `pending`
   - `rejected`
   - `needs_review`

## Follow-Up Gaps

- [ ] Decide whether eye masks should eventually adopt a dedicated review
      policy/contract note rather than living only in the workflow doc.
- [ ] Define swim-bladder-specific approval heuristics for the first curated
      subject-mask datasets.
