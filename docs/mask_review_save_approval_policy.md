# Mask Review Save And Approval Policy

<!-- design-meta
status: active
last_verified: 2026-04-02
-->

## Purpose

Document the current save and approval semantics for mask-oriented review
workflows, and compare them against the existing detect/keypoint review model.
The browser checkpoint/apply layer described below is target behavior for the
web workflow unless a specific route already implements it; current direct CLI
and legacy UI saves may still write canonical Zarr arrays immediately.

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

For browser review, distinguish three user-visible states:

- checkpointed session edits: durable enough to recover the browser/session
  overlay, but not visible to ordinary canonical Zarr consumers
- applied canonical edits: written back to the target review run and visible to
  canonical consumers, while the assignment may still remain open
- approved/completed work: an explicit review-status or task-completion action
  after the saved/applied content is acceptable

Users must not need to complete an assignment before saving/applying mask edits
back to the Zarr review surface. They should be able to apply edits, leave the
assignment open, and later continue from the applied Zarr state plus any newer
unapplied session overlay.

V1 browser storage decision:

- checkpoint metadata and edit payload references should live in the web
  labeling SQLite sidecar, not in the registry SQLite and not in canonical Zarr
- for refined subject masks, the v1 apply payload should be a full replacement
  dense ROI mask for one `(row, component)` at a time, rather than stroke-delta
  replay
- stroke/lasso deltas may remain UI-local implementation details, but the
  durable checkpoint/apply payload should reconstruct the exact overlay and
  canonical replacement mask without depending on browser event replay

## Comparison Matrix

| Workflow | Editable target | Save granularity | Approval granularity | Review payload home | Parent latest pointer |
| --- | --- | --- | --- | --- | --- |
| Detect | Manual subgroup under `refined_detect_runs/<run>/` | subgroup-level corrections | whole refined detect run | `detect_review_status` | `detect_review_status_latest` |
| Keypoints | `refined_keypoints_runs/<run>` | run-local row edits | whole refined keypoint run | `keypoint_review_status` | `keypoint_review_status_latest` |
| Eye masks (legacy compat) | `refined_eye_masks_runs/<run>` for standalone historical runs; derived compat runs are read-only | per-ROI manual correction on standalone historical runs | whole refined eye-mask run | `eye_mask_review_status` | `eye_mask_review_status_latest` |
| Subject masks (canonical unified surface) | `refined_subject_masks_runs/<run>` | per-ROI component-aware save | per-component, with run aggregation | `component_review_statuses` + `refined_subject_mask_review_status` | `refined_subject_mask_review_status_latest` |

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

- canonical manual eye review authority now lives in
  `refined_subject_masks_runs/<run>`, reached via
  `scripts/py -m fisheye.tune.eye_mask_review --manual`
- `refined_eye_masks_runs/<run>` remains supported for historical archives and
  eye-specific consumers, but it is no longer the default manual review surface
- derived compatibility `refined_eye_masks_runs/<run>` artifacts refreshed from
  canonical refined-subject eye edits/review-state changes should be treated as
  read-only in legacy refined-eye viewers
- standalone historical refined-eye runs may still receive per-ROI manual
  correction in the legacy UI
- approval on the legacy refined-eye stage remains whole-run, not per-eye
- unified subject-mask component registry/query/operator surfaces should treat
  this stage as a compatibility source for eye components rather than as the
  long-term canonical refined mask surface

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

- `scripts/py -m fisheye.tune.eye_mask_review --manual` now treats
  `refined_subject_masks_runs` as the canonical manual review surface for eye
  components, seeded through a compatibility `subject_mask_runs` projection
- `scripts/py -m fisheye.tune.eye_mask_review --legacy-manual` remains
  available only for standalone historical refined-eye runs and diagnostics
- if the selected refined-eye run is a derived compatibility artifact, legacy
  viewers now redirect operators back to the canonical unified manual review
  surface rather than allowing the compat artifact to drift
- new operator summaries should prefer the unified subject-mask component view
  when asking whether eye components are available/reviewed
- but the policy is documented mostly procedurally in the workflow doc rather
  than in a dedicated contract/policy note

References:

- [src/fisheye/docs/eye_mask_tuning_workflow.md](/home/delahantyj@hhmi.org/gitrepos/palette/src/fisheye/docs/eye_mask_tuning_workflow.md)

## Subject Masks

Current policy:

- `refined_subject_masks_runs/<run>` is the canonical editable working artifact
- current direct save writes are per-ROI and recompute sibling refined metadata
  immediately
- target browser behavior should use lightweight session checkpoints for
  frequent paint/lasso changes, then explicitly apply/coalesce those edits back
  to `refined_subject_masks_runs/<run>` without requiring task completion
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
- for new manual eye review, this subject-mask component surface is now the
  default authority rather than the legacy refined-eye run-level payload
- unified subject-mask component surfaces are also the preferred operator/query
  answer for eye availability during transition, with legacy eye-stage rows
  projected in only when needed
- downstream gating should therefore prefer component review state when the
  consumer is component-specific
- if a browser session has unapplied mask edits, component approval should be
  blocked or should first require a successful apply-to-Zarr operation so the
  approved component state matches the canonical masks

Current swim-bladder policy note:

- the early canary heuristics for what is "good enough to save" versus "good
  enough to approve" now live in
  [swim_bladder_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_review_policy.md)
- that policy keeps save ROI-local, keeps approval component-level, and treats
  explicit-negative swim-bladder labels conservatively so we do not fabricate
  training negatives from ambiguous ROIs

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
5. Allow applied mask edits before assignment completion, so reviewers can
   leave and resume from canonical Zarr changes without marking the assignment
   complete.
6. Keep high-frequency browser checkpoints lightweight; coalesce and apply
   edits to Zarr at explicit save/apply boundaries rather than on every
   paint/drag operation.
7. Continue using the shared state vocabulary:
   - `approved`
   - `pending`
   - `rejected`
   - `needs_review`

## Acceptance Tests For Browser Checkpoint/Apply

Minimum tests before making the browser session layer the default save path:

- checkpointing a mask edit does not change canonical Zarr arrays or
  `edit_revision`
- reopening the same active session restores the checkpointed mask overlay
- explicit apply writes the dense replacement mask to `masks_roi`
- explicit apply increments `edit_revision` only after the canonical write
  succeeds
- retrying the same apply with the same `apply_id` is idempotent
- applying with a stale `target_edit_revision` fails without partial writes
- component approval is blocked or requires successful apply when unapplied
  session edits exist
- assignment completion is not required before a successful apply-to-Zarr

## Follow-Up Gaps

- [ ] Decide whether eye masks should eventually adopt a dedicated review
      policy/contract note rather than living only in the workflow doc.
- [x] Define swim-bladder-specific approval heuristics for the first curated
      subject-mask datasets.
  See
  [swim_bladder_review_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/swim_bladder_review_policy.md).
