# Repo-Wide Staleness Workflow Edge Checklist

<!-- design-meta
status: draft
last_updated: 2026-04-06
-->

## Purpose

Make the staleness plan concrete by enumerating workflow edges:

- which upstream correction happened
- which downstream workflow is affected
- whether the correct signal is `stale` or `missing`
- whether the repo already does this, only partially does it, or still needs
  implementation

This note complements:

- [repo_wide_staleness_policy.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_policy.md)
- [repo_wide_staleness_checklist.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_checklist.md)
- [repo_wide_staleness_gap_matrix.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_gap_matrix.md)
- [repo_wide_staleness_implementation_todo.md](/home/delahantyj@hhmi.org/gitrepos/palette/docs/repo_wide_staleness_implementation_todo.md)

## Signal Rules

Use these rules before interpreting any row in the checklist.

- `missing`: use when a new upstream run was produced, or when row identity
  broke and targeted repair is not trustworthy.
- `stale`: use when the upstream source changed inside the same editable
  lineage and the downstream artifact still exists.
- `no signal`: use when the change is only review/approval state and does not
  change the underlying source artifact.

Short rule:

- new run or identity break -> `missing`
- same-run row-stable correction -> `stale`
- review-only change -> no stale/missing transition

## Status Labels

- `implemented`: repo already has a reasonably canonical mechanism
- `partial`: some local behavior exists, but it is not yet canonical,
  registry-native, or complete
- `todo`: expected behavior is clear but not implemented
- `policy_tbd`: the repo still needs an explicit decision before implementation

## Edge Checklist

| Source event | Downstream workflow | Desired signal | Current status | Notes |
| --- | --- | --- | --- | --- |
| New `detect_runs/<run>` | `refined_detect` | `missing` | implemented | Covered by runtime cascade. |
| New `detect_runs/<run>` | `detect_quality` | `missing` | implemented | Covered by runtime cascade. |
| New `detect_runs/<run>` | `crop`, `keypoints`, `refined_keypoints`, `eye_masks`, `refined_eye_masks`, `arena_assignment`, `tracks` | `missing` | implemented | Covered transitively by runtime cascade. |
| New `detect_runs/<run>` | `subject_mask_runs`, `refined_subject_masks_runs`, swim-bladder raw/refined workflows | `missing` | todo | Current runtime cascade does not model these stage families yet. |
| New refined-detect run | `crop` and everything derived from that crop lineage | `missing` | partial | `crop` is covered by cascade, but subject-mask/swim step-family parity is not fully encoded. |
| Refined-detect row-stable bbox move/resize inside same editable lineage | `crop_runs` | row-local refresh or explicit crop-stale contract | policy_tbd | This is the core detect/crop contract gap. |
| Refined-detect row-stable bbox move/resize inside same editable lineage | `keypoints_runs`, `refined_keypoints_runs` | `stale` if row identity preserved | todo | Depends on the crop contract being defined first. |
| Refined-detect row-stable bbox move/resize inside same editable lineage | `eye_masks_runs`, `refined_eye_masks_runs` | `stale` after crop/keypoint lineage change | todo | Likely requires keypoint or crop stale propagation, not a direct one-off patch. |
| Refined-detect row-stable bbox move/resize inside same editable lineage | `subject_mask_runs`, `refined_subject_masks_runs`, swim bladder | `stale` after crop lineage change | todo | Subject/swim should eventually follow the same pattern as eye masks when lineage is stable. |
| Refined-detect add/delete/split/merge or identity-breaking correction | all downstream workflows from crop onward | `missing` | policy_tbd | Strong default should be rerun/invalidate, not row-local stale. |
| New `crop_runs/<run>` | `keypoints`, `refined_keypoints`, `eye_masks`, `refined_eye_masks` | `missing` | implemented/partial | Core chain is conceptually covered by new-run invalidation, but not all mixed-mode consumers are represented uniformly. |
| New `crop_runs/<run>` | `subject_mask_runs`, `refined_subject_masks_runs`, swim bladder | `missing` | todo | Subject/swim step-family participation in runtime cascade needs explicit implementation. |
| Crop row-stable geometry correction inside same lineage | `keypoints_runs`, `refined_keypoints_runs` | `stale` | todo | Safe only if `frame_indices`, `detection_indices`, and crop identity stay stable. |
| Crop row-stable geometry correction inside same lineage | `eye_masks_runs`, `refined_eye_masks_runs` | `stale` | todo | Should likely follow keypoint/crop stale rather than pretend nothing changed. |
| Crop row-stable geometry correction inside same lineage | `subject_mask_runs`, `refined_subject_masks_runs`, swim bladder | `stale` | todo | Same rule as above for crop-derived mask workflows. |
| New `refined_keypoints_runs/<run>` | `eye_masks`, `refined_eye_masks`, `arena_assignment`, `tracks` | `missing` | implemented | `eye_masks` and `arena_assignment` are already downstream of `refined_keypoints` in runtime cascade. |
| New `refined_keypoints_runs/<run>` | subject-mask workflows that explicitly consume keypoints | `missing` | todo | Includes swim bladder and any point-prompted subject-mask path. |
| Row-stable correction in `refined_keypoints_runs/<run>` | `eye_masks_runs` | `stale` | implemented | Canonical precedent already exists. |
| Row-stable correction in `refined_keypoints_runs/<run>` | `refined_eye_masks_runs` | `stale` | implemented | Explicit stale + explicit resolution already exist. |
| Row-stable correction in `refined_keypoints_runs/<run>` | `subject_mask_runs` rows that consume keypoints | source refresh plus downstream `stale` | partial | Local swim-bladder partial refresh exists, but the canonical subject-mask payload is missing. |
| Row-stable correction in `refined_keypoints_runs/<run>` | `refined_subject_masks_runs` | `stale` | partial | Local row queue exists for swim/subject, but top-level canonical payload is still missing. |
| Row-stable correction in `refined_keypoints_runs/<run>` | `arena_assignment_runs`, `tracking_runs` | likely `missing` for now | policy_tbd | These workflows are more global/sequence-level; targeted stale may not be worth the complexity initially. |
| New `subject_mask_runs/<run>` | `refined_subject_masks_runs` | `missing` or source-run mismatch stale | partial | Registry already detects latest-source mismatch, but runtime semantics are not yet first-class. |
| Row-stable refresh of non-curated `subject_mask_runs/<run>` rows | matching `refined_subject_masks_runs` rows | `stale` unless untouched rows auto-sync | partial | This is the subject/swim workflow that now exists locally and needs canonicalization. |
| New `eye_masks_runs/<run>` | `refined_eye_masks_runs` | `missing` | partial | The refined eye workflow exists, but repo-wide stale/missing policy is still stronger here than in subject/swim. |
| Row-stable refresh of non-curated `eye_masks_runs/<run>` rows | matching `refined_eye_masks_runs` rows | `stale` unless explicitly preserved/resolved | implemented | This is the strongest existing stale contract. |
| Manual edit in `refined_eye_masks_runs/<run>` | downstream training/export artifacts | no runtime stale; rebuild required | partial | Training artifacts are out-of-band products, not runtime stage rows. |
| Manual edit in `refined_subject_masks_runs/<run>` | downstream training/export artifacts | no runtime stale; rebuild required | partial | Same rebuild rule should apply to subject-mask exports/models. |
| Review/approval-only status change in refined eye/subject workflows | any downstream runtime workflow | no signal | todo | Should be enforced repo-wide so review changes do not masquerade as source drift. |

## First Implementation Pass

If the goal is to make staleness visible across the repo without trying to
solve every hard edge at once, the best first pass is:

1. Canonicalize `source_subject_mask_stale` for refined subject-mask workflows.
2. Project subject/swim stale into registry/query surfaces.
3. Add explicit subject-mask stale resolution.
4. Extend runtime cascade coverage for new-run subject-mask/swim stage families.
5. Decide the detect/crop row-stable versus identity-breaking contract.

## Hard Edges To Defer Until Policy Is Explicit

These should not be implemented ad hoc.

- row-stable refined-detect bbox edits feeding targeted crop/keypoint/mask stale
- crop-in-place correction semantics
- targeted stale for `arena_assignment` and `tracks`
- any attempt to treat add/delete/split/merge detection edits as safe row-local
  stale events
